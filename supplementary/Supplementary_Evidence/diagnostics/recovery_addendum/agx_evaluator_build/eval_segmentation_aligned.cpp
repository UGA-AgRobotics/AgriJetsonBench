#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace
{

class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, char const* msg) noexcept override
    {
        if (severity > Severity::kWARNING)
        {
            return;
        }
        std::cerr << "[TRT] " << msg << '\n';
    }
};

template <typename T>
struct InferDeleter
{
    void operator()(T* obj) const
    {
        delete obj;
    }
};

using RuntimePtr = std::unique_ptr<nvinfer1::IRuntime, InferDeleter<nvinfer1::IRuntime>>;
using EnginePtr = std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter<nvinfer1::ICudaEngine>>;
using ContextPtr = std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter<nvinfer1::IExecutionContext>>;

struct Sample
{
    fs::path imagePath;
    fs::path maskPath;
};

struct Args
{
    fs::path enginePath;
    fs::path datasetRoot;
    fs::path manifestPath;
    fs::path jsonOut;
    int imgsz{640};
    std::string inputName;
    std::string outputName;
    std::string evalResolution{"engine"};
};

std::string trim(std::string value)
{
    auto notSpace = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), notSpace));
    value.erase(std::find_if(value.rbegin(), value.rend(), notSpace).base(), value.end());
    return value;
}

std::vector<std::string> parseCsvLine(std::string const& line)
{
    std::vector<std::string> fields;
    std::string current;
    bool inQuotes = false;
    for (std::size_t idx = 0; idx < line.size(); ++idx)
    {
        char const ch = line[idx];
        if (ch == '"')
        {
            if (inQuotes && idx + 1 < line.size() && line[idx + 1] == '"')
            {
                current.push_back('"');
                ++idx;
            }
            else
            {
                inQuotes = !inQuotes;
            }
            continue;
        }
        if (ch == ',' && !inQuotes)
        {
            fields.push_back(current);
            current.clear();
            continue;
        }
        current.push_back(ch);
    }
    fields.push_back(current);
    return fields;
}

std::string jsonEscape(std::string const& value)
{
    std::ostringstream oss;
    for (char const ch : value)
    {
        switch (ch)
        {
        case '\\':
            oss << "\\\\";
            break;
        case '"':
            oss << "\\\"";
            break;
        case '\n':
            oss << "\\n";
            break;
        case '\r':
            oss << "\\r";
            break;
        case '\t':
            oss << "\\t";
            break;
        default:
            oss << ch;
            break;
        }
    }
    return oss.str();
}

std::vector<std::uint8_t> readBinary(fs::path const& path)
{
    std::ifstream handle(path, std::ios::binary);
    if (!handle)
    {
        throw std::runtime_error("Could not open file: " + path.string());
    }
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(handle)), std::istreambuf_iterator<char>());
}

void writeText(fs::path const& path, std::string const& text)
{
    fs::create_directories(path.parent_path());
    std::ofstream handle(path, std::ios::binary);
    if (!handle)
    {
        throw std::runtime_error("Could not write file: " + path.string());
    }
    handle << text;
}

void cudaCheck(cudaError_t code, char const* op)
{
    if (code != cudaSuccess)
    {
        throw std::runtime_error(std::string(op) + " failed: " + cudaGetErrorString(code));
    }
}

std::size_t dtypeSize(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT8:
        return 1;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kBOOL:
        return 1;
    case nvinfer1::DataType::kUINT8:
        return 1;
    case nvinfer1::DataType::kINT64:
        return 8;
    default:
        throw std::runtime_error("Unsupported TensorRT tensor dtype");
    }
}

std::int64_t volume(nvinfer1::Dims const& dims)
{
    std::int64_t total = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] < 0)
        {
            return -1;
        }
        total *= static_cast<std::int64_t>(dims.d[i]);
    }
    return total;
}

float halfToFloat(std::uint16_t value)
{
    std::uint32_t sign = (value >> 15U) & 0x1U;
    std::uint32_t exp = (value >> 10U) & 0x1FU;
    std::uint32_t mantissa = value & 0x3FFU;

    std::uint32_t outBits = 0U;
    if (exp == 0)
    {
        if (mantissa == 0)
        {
            outBits = sign << 31U;
        }
        else
        {
            exp = 1U;
            while ((mantissa & 0x400U) == 0)
            {
                mantissa <<= 1U;
                --exp;
            }
            mantissa &= 0x3FFU;
            outBits = (sign << 31U) | ((exp + 112U) << 23U) | (mantissa << 13U);
        }
    }
    else if (exp == 31U)
    {
        outBits = (sign << 31U) | 0x7F800000U | (mantissa << 13U);
    }
    else
    {
        outBits = (sign << 31U) | ((exp + 112U) << 23U) | (mantissa << 13U);
    }

    float out = 0.0F;
    std::memcpy(&out, &outBits, sizeof(out));
    return out;
}

std::vector<float> decodeTensor(std::vector<std::uint8_t> const& raw, nvinfer1::DataType dtype)
{
    std::vector<float> decoded;
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
    {
        std::size_t count = raw.size() / sizeof(float);
        decoded.resize(count);
        std::memcpy(decoded.data(), raw.data(), count * sizeof(float));
        break;
    }
    case nvinfer1::DataType::kHALF:
    {
        std::size_t count = raw.size() / sizeof(std::uint16_t);
        decoded.resize(count);
        auto const* ptr = reinterpret_cast<std::uint16_t const*>(raw.data());
        for (std::size_t idx = 0; idx < count; ++idx)
        {
            decoded[idx] = halfToFloat(ptr[idx]);
        }
        break;
    }
    case nvinfer1::DataType::kINT8:
    {
        decoded.reserve(raw.size());
        for (std::uint8_t value : raw)
        {
            decoded.push_back(static_cast<float>(static_cast<std::int8_t>(value)));
        }
        break;
    }
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kBOOL:
    {
        decoded.reserve(raw.size());
        for (std::uint8_t value : raw)
        {
            decoded.push_back(static_cast<float>(value));
        }
        break;
    }
    case nvinfer1::DataType::kINT32:
    {
        std::size_t count = raw.size() / sizeof(std::int32_t);
        decoded.resize(count);
        auto const* ptr = reinterpret_cast<std::int32_t const*>(raw.data());
        for (std::size_t idx = 0; idx < count; ++idx)
        {
            decoded[idx] = static_cast<float>(ptr[idx]);
        }
        break;
    }
    case nvinfer1::DataType::kINT64:
    {
        std::size_t count = raw.size() / sizeof(std::int64_t);
        decoded.resize(count);
        auto const* ptr = reinterpret_cast<std::int64_t const*>(raw.data());
        for (std::size_t idx = 0; idx < count; ++idx)
        {
            decoded[idx] = static_cast<float>(ptr[idx]);
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported tensor dtype while decoding output");
    }
    return decoded;
}

fs::path resolvePath(fs::path const& datasetRoot, std::string const& rawPath)
{
    fs::path path(rawPath);
    if (path.is_absolute())
    {
        return path;
    }
    return (datasetRoot / path).lexically_normal();
}

std::vector<Sample> loadManifest(fs::path const& datasetRoot, fs::path const& manifestPath)
{
    std::ifstream handle(manifestPath);
    if (!handle)
    {
        throw std::runtime_error("Could not open manifest: " + manifestPath.string());
    }

    std::string headerLine;
    if (!std::getline(handle, headerLine))
    {
        throw std::runtime_error("Manifest is empty: " + manifestPath.string());
    }

    auto const headers = parseCsvLine(headerLine);
    std::unordered_map<std::string, std::size_t> headerIndex;
    for (std::size_t idx = 0; idx < headers.size(); ++idx)
    {
        headerIndex[trim(headers[idx])] = idx;
    }
    if (!headerIndex.count("image_path") || !headerIndex.count("mask_path"))
    {
        throw std::runtime_error("Manifest must contain image_path and mask_path columns: " + manifestPath.string());
    }

    std::vector<Sample> samples;
    std::string line;
    while (std::getline(handle, line))
    {
        if (trim(line).empty())
        {
            continue;
        }
        auto fields = parseCsvLine(line);
        if (fields.size() < headers.size())
        {
            fields.resize(headers.size());
        }
        samples.push_back(
            Sample{
                resolvePath(datasetRoot, trim(fields[headerIndex["image_path"]])),
                resolvePath(datasetRoot, trim(fields[headerIndex["mask_path"]])),
            });
    }

    if (samples.empty())
    {
        throw std::runtime_error("Manifest produced zero samples: " + manifestPath.string());
    }
    return samples;
}

std::vector<float> preprocessImage(fs::path const& imagePath, int width, int height)
{
    cv::Mat bgr = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
    if (bgr.empty())
    {
        throw std::runtime_error("Could not read image: " + imagePath.string());
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat fp32;
    resized.convertTo(fp32, CV_32FC3, 1.0 / 255.0);

    constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
    constexpr float STD[3] = {0.229f, 0.224f, 0.225f};

    std::vector<float> chw(static_cast<std::size_t>(3 * width * height));
    for (int y = 0; y < height; ++y)
    {
        auto const* row = fp32.ptr<cv::Vec3f>(y);
        for (int x = 0; x < width; ++x)
        {
            auto const pixel = row[x];
            for (int c = 0; c < 3; ++c)
            {
                chw[static_cast<std::size_t>(c * width * height + y * width + x)] = (pixel[c] - MEAN[c]) / STD[c];
            }
        }
    }
    return chw;
}

std::vector<int> matToMaskVector(cv::Mat const& mask)
{
    std::vector<int> out(static_cast<std::size_t>(mask.cols * mask.rows));
    for (int y = 0; y < mask.rows; ++y)
    {
        auto const* row = mask.ptr<std::uint8_t>(y);
        for (int x = 0; x < mask.cols; ++x)
        {
            out[static_cast<std::size_t>(y * mask.cols + x)] = static_cast<int>(row[x]);
        }
    }
    return out;
}

std::vector<int> loadMask(fs::path const& maskPath, int width, int height)
{
    cv::Mat mask = cv::imread(maskPath.string(), cv::IMREAD_GRAYSCALE);
    if (mask.empty())
    {
        throw std::runtime_error("Could not read mask: " + maskPath.string());
    }
    cv::Mat resized;
    cv::resize(mask, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_NEAREST);
    return matToMaskVector(resized);
}

std::vector<int> loadMaskOriginal(fs::path const& maskPath, int& width, int& height)
{
    cv::Mat mask = cv::imread(maskPath.string(), cv::IMREAD_GRAYSCALE);
    if (mask.empty())
    {
        throw std::runtime_error("Could not read mask: " + maskPath.string());
    }
    width = mask.cols;
    height = mask.rows;
    return matToMaskVector(mask);
}

std::string dimsToString(nvinfer1::Dims const& dims)
{
    std::ostringstream oss;
    for (int idx = 0; idx < dims.nbDims; ++idx)
    {
        if (idx)
        {
            oss << 'x';
        }
        oss << dims.d[idx];
    }
    return oss.str();
}

Args parseArgs(int argc, char** argv)
{
    Args args;
    for (int idx = 1; idx < argc; ++idx)
    {
        std::string token = argv[idx];
        auto requireValue = [&](std::string const& flag) -> std::string {
            if (idx + 1 >= argc)
            {
                throw std::runtime_error("Missing value for " + flag);
            }
            return argv[++idx];
        };

        if (token == "--engine")
        {
            args.enginePath = requireValue(token);
        }
        else if (token == "--dataset-root")
        {
            args.datasetRoot = requireValue(token);
        }
        else if (token == "--manifest")
        {
            args.manifestPath = requireValue(token);
        }
        else if (token == "--json-out")
        {
            args.jsonOut = requireValue(token);
        }
        else if (token == "--imgsz")
        {
            args.imgsz = std::stoi(requireValue(token));
        }
        else if (token == "--input-name")
        {
            args.inputName = requireValue(token);
        }
        else if (token == "--output-name")
        {
            args.outputName = requireValue(token);
        }
        else if (token == "--eval-resolution")
        {
            args.evalResolution = requireValue(token);
            if (args.evalResolution != "engine" && args.evalResolution != "original")
            {
                throw std::runtime_error("--eval-resolution must be engine or original");
            }
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + token);
        }
    }

    if (args.enginePath.empty() || args.datasetRoot.empty() || args.manifestPath.empty() || args.jsonOut.empty())
    {
        throw std::runtime_error("Required arguments: --engine --dataset-root --manifest --json-out");
    }
    return args;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        Args const args = parseArgs(argc, argv);
        auto const samples = loadManifest(args.datasetRoot, args.manifestPath);

        TrtLogger logger;
        RuntimePtr runtime{nvinfer1::createInferRuntime(logger)};
        if (!runtime)
        {
            throw std::runtime_error("createInferRuntime returned nullptr");
        }

        auto const engineBlob = readBinary(args.enginePath);
        EnginePtr engine{runtime->deserializeCudaEngine(engineBlob.data(), engineBlob.size())};
        if (!engine)
        {
            throw std::runtime_error("deserializeCudaEngine returned nullptr");
        }

        ContextPtr context{engine->createExecutionContext()};
        if (!context)
        {
            throw std::runtime_error("createExecutionContext returned nullptr");
        }

        std::string inputName = args.inputName;
        std::string outputName = args.outputName;
        for (int idx = 0; idx < engine->getNbIOTensors(); ++idx)
        {
            char const* tensorName = engine->getIOTensorName(idx);
            auto const mode = engine->getTensorIOMode(tensorName);
            if (inputName.empty() && mode == nvinfer1::TensorIOMode::kINPUT)
            {
                inputName = tensorName;
            }
            if (outputName.empty() && mode == nvinfer1::TensorIOMode::kOUTPUT)
            {
                outputName = tensorName;
            }
        }
        if (inputName.empty() || outputName.empty())
        {
            throw std::runtime_error("Could not resolve engine input/output tensor names");
        }

        nvinfer1::Dims inputDims = engine->getTensorShape(inputName.c_str());
        if (inputDims.nbDims == 4 && (inputDims.d[0] < 0 || inputDims.d[2] < 0 || inputDims.d[3] < 0))
        {
            nvinfer1::Dims4 fixedShape{1, 3, args.imgsz, args.imgsz};
            if (!context->setInputShape(inputName.c_str(), fixedShape))
            {
                throw std::runtime_error("Failed to set input shape for dynamic engine");
            }
            inputDims = context->getTensorShape(inputName.c_str());
        }

        nvinfer1::Dims outputDims = context->getTensorShape(outputName.c_str());
        if (volume(outputDims) <= 0)
        {
            outputDims = engine->getTensorShape(outputName.c_str());
        }
        if (volume(outputDims) <= 0)
        {
            throw std::runtime_error("Could not resolve concrete output tensor shape");
        }

        int channels = 0;
        int height = 0;
        int width = 0;
        if (outputDims.nbDims == 4)
        {
            channels = outputDims.d[1];
            height = outputDims.d[2];
            width = outputDims.d[3];
        }
        else if (outputDims.nbDims == 3)
        {
            channels = outputDims.d[0];
            height = outputDims.d[1];
            width = outputDims.d[2];
        }
        else
        {
            throw std::runtime_error("Unsupported output rank for segmentation logits: " + dimsToString(outputDims));
        }
        if (channels <= 0 || height <= 0 || width <= 0)
        {
            throw std::runtime_error("Invalid output tensor shape: " + dimsToString(outputDims));
        }

        auto const inputCount = static_cast<std::size_t>(volume(inputDims));
        auto const outputCount = static_cast<std::size_t>(volume(outputDims));
        auto const outputDtype = engine->getTensorDataType(outputName.c_str());
        auto const inputBytes = inputCount * sizeof(float);
        auto const outputBytes = outputCount * dtypeSize(outputDtype);

        void* inputDevice = nullptr;
        void* outputDevice = nullptr;
        cudaStream_t stream{};
        cudaCheck(cudaMalloc(&inputDevice, inputBytes), "cudaMalloc(input)");
        cudaCheck(cudaMalloc(&outputDevice, outputBytes), "cudaMalloc(output)");
        cudaCheck(cudaStreamCreate(&stream), "cudaStreamCreate");

        if (!context->setTensorAddress(inputName.c_str(), inputDevice))
        {
            throw std::runtime_error("setTensorAddress failed for input tensor");
        }
        if (!context->setTensorAddress(outputName.c_str(), outputDevice))
        {
            throw std::runtime_error("setTensorAddress failed for output tensor");
        }

        std::vector<std::vector<std::uint64_t>> confusion(
            static_cast<std::size_t>(channels),
            std::vector<std::uint64_t>(static_cast<std::size_t>(channels), 0ULL));

        std::vector<std::uint8_t> outputRaw(outputBytes);
        std::size_t processed = 0;

        for (auto const& sample : samples)
        {
            auto const inputHost = preprocessImage(sample.imagePath, args.imgsz, args.imgsz);

            cudaCheck(cudaMemcpyAsync(inputDevice, inputHost.data(), inputBytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(HtoD)");
            if (!context->enqueueV3(stream))
            {
                throw std::runtime_error("enqueueV3 failed during engine inference");
            }
            cudaCheck(cudaMemcpyAsync(outputRaw.data(), outputDevice, outputBytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(DtoH)");
            cudaCheck(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

            auto const logits = decodeTensor(outputRaw, outputDtype);
            auto const spatial = static_cast<std::size_t>(height * width);
            if (logits.size() != static_cast<std::size_t>(channels) * spatial)
            {
                throw std::runtime_error("Unexpected logits size while decoding output tensor");
            }

            if (args.evalResolution == "original")
            {
                int gtWidth = 0;
                int gtHeight = 0;
                auto const gtMask = loadMaskOriginal(sample.maskPath, gtWidth, gtHeight);
                cv::Mat predSmall(height, width, CV_8UC1);
                for (int y = 0; y < height; ++y)
                {
                    auto* predRow = predSmall.ptr<std::uint8_t>(y);
                    for (int x = 0; x < width; ++x)
                    {
                        std::size_t const pixel = static_cast<std::size_t>(y * width + x);
                        int predClass = 0;
                        float bestValue = logits[pixel];
                        for (int cls = 1; cls < channels; ++cls)
                        {
                            float const candidate = logits[static_cast<std::size_t>(cls) * spatial + pixel];
                            if (candidate > bestValue)
                            {
                                bestValue = candidate;
                                predClass = cls;
                            }
                        }
                        predRow[x] = static_cast<std::uint8_t>(predClass);
                    }
                }

                cv::Mat predEval;
                cv::resize(predSmall, predEval, cv::Size(gtWidth, gtHeight), 0.0, 0.0, cv::INTER_NEAREST);
                for (int y = 0; y < gtHeight; ++y)
                {
                    auto const* predRow = predEval.ptr<std::uint8_t>(y);
                    for (int x = 0; x < gtWidth; ++x)
                    {
                        std::size_t const pixel = static_cast<std::size_t>(y * gtWidth + x);
                        int const gtClass = gtMask[pixel];
                        if (gtClass < 0 || gtClass >= channels)
                        {
                            continue;
                        }
                        int const predClass = static_cast<int>(predRow[x]);
                        confusion[static_cast<std::size_t>(gtClass)][static_cast<std::size_t>(predClass)] += 1ULL;
                    }
                }
            }
            else
            {
                auto const maskHost = loadMask(sample.maskPath, width, height);
                for (std::size_t pixel = 0; pixel < spatial; ++pixel)
                {
                    int const gtClass = maskHost[pixel];
                    if (gtClass < 0 || gtClass >= channels)
                    {
                        continue;
                    }
                    int predClass = 0;
                    float bestValue = logits[pixel];
                    for (int cls = 1; cls < channels; ++cls)
                    {
                        float const candidate = logits[static_cast<std::size_t>(cls) * spatial + pixel];
                        if (candidate > bestValue)
                        {
                            bestValue = candidate;
                            predClass = cls;
                        }
                    }
                    confusion[static_cast<std::size_t>(gtClass)][static_cast<std::size_t>(predClass)] += 1ULL;
                }
            }
            ++processed;
            if (processed == 1 || processed % 100 == 0 || processed == samples.size())
                std::cerr << "[ALIGNED] processed=" << processed << "/" << samples.size() << std::endl;
        }

        cudaCheck(cudaStreamDestroy(stream), "cudaStreamDestroy");
        cudaCheck(cudaFree(inputDevice), "cudaFree(input)");
        cudaCheck(cudaFree(outputDevice), "cudaFree(output)");

        std::vector<double> perClassIou(static_cast<std::size_t>(channels), 0.0);
        std::vector<double> perClassDice(static_cast<std::size_t>(channels), 0.0);
        std::vector<double> perClassRecall(static_cast<std::size_t>(channels), 0.0);
        std::vector<double> perClassPrecision(static_cast<std::size_t>(channels), 0.0);
        std::vector<std::uint64_t> perClassSupport(static_cast<std::size_t>(channels), 0ULL);

        double miou = 0.0;
        double mdice = 0.0;
        double fgMiou = 0.0;
        double fgMdice = 0.0;
        std::uint64_t totalPixels = 0ULL;
        std::uint64_t totalCorrect = 0ULL;

        for (int cls = 0; cls < channels; ++cls)
        {
            std::uint64_t rowSum = 0ULL;
            std::uint64_t colSum = 0ULL;
            for (int other = 0; other < channels; ++other)
            {
                rowSum += confusion[static_cast<std::size_t>(cls)][static_cast<std::size_t>(other)];
                colSum += confusion[static_cast<std::size_t>(other)][static_cast<std::size_t>(cls)];
            }
            auto const tp = confusion[static_cast<std::size_t>(cls)][static_cast<std::size_t>(cls)];
            auto const fp = colSum - tp;
            auto const fn = rowSum - tp;
            auto const unionCount = tp + fp + fn;
            auto const diceDenom = (2ULL * tp) + fp + fn;

            perClassSupport[static_cast<std::size_t>(cls)] = rowSum;
            perClassIou[static_cast<std::size_t>(cls)] = unionCount ? static_cast<double>(tp) / static_cast<double>(unionCount) : 0.0;
            perClassDice[static_cast<std::size_t>(cls)] = diceDenom ? static_cast<double>(2ULL * tp) / static_cast<double>(diceDenom) : 0.0;
            perClassRecall[static_cast<std::size_t>(cls)] = rowSum ? static_cast<double>(tp) / static_cast<double>(rowSum) : 0.0;
            perClassPrecision[static_cast<std::size_t>(cls)] = colSum ? static_cast<double>(tp) / static_cast<double>(colSum) : 0.0;

            miou += perClassIou[static_cast<std::size_t>(cls)];
            mdice += perClassDice[static_cast<std::size_t>(cls)];
            if (cls > 0)
            {
                fgMiou += perClassIou[static_cast<std::size_t>(cls)];
                fgMdice += perClassDice[static_cast<std::size_t>(cls)];
            }
            totalPixels += rowSum;
            totalCorrect += tp;
        }

        miou /= static_cast<double>(channels);
        mdice /= static_cast<double>(channels);
        if (channels > 1)
        {
            fgMiou /= static_cast<double>(channels - 1);
            fgMdice /= static_cast<double>(channels - 1);
        }
        auto const pixelAccuracy = totalPixels ? static_cast<double>(totalCorrect) / static_cast<double>(totalPixels) : 0.0;

        std::ostringstream json;
        json << std::fixed << std::setprecision(12);
        json << "{\n";
        json << "  \"engine_path\": \"" << jsonEscape(args.enginePath.string()) << "\",\n";
        json << "  \"dataset_root\": \"" << jsonEscape(args.datasetRoot.string()) << "\",\n";
        json << "  \"manifest_path\": \"" << jsonEscape(args.manifestPath.string()) << "\",\n";
        json << "  \"input_name\": \"" << jsonEscape(inputName) << "\",\n";
        json << "  \"output_name\": \"" << jsonEscape(outputName) << "\",\n";
        json << "  \"input_shape\": \"" << jsonEscape(dimsToString(inputDims)) << "\",\n";
        json << "  \"output_shape\": \"" << jsonEscape(dimsToString(outputDims)) << "\",\n";
        json << "  \"processed_samples\": " << processed << ",\n";
        json << "  \"eval_resolution\": \"" << jsonEscape(args.evalResolution) << "\",\n";
        json << "  \"preprocess\": {\n";
        json << "    \"mode\": \"rgb_imagenet_normalized_fixed_resize\",\n";
        json << "    \"image_resize\": \"bilinear\",\n";
        json << "    \"mask_resize\": \"nearest\",\n";
        json << "    \"metric_resolution\": \"" << jsonEscape(args.evalResolution) << "\",\n";
        json << "    \"mean\": [0.485000000000, 0.456000000000, 0.406000000000],\n";
        json << "    \"std\": [0.229000000000, 0.224000000000, 0.225000000000]\n";
        json << "  },\n";
        json << "  \"metrics\": {\n";
        json << "    \"miou\": " << miou << ",\n";
        json << "    \"mdice\": " << mdice << ",\n";
        json << "    \"foreground_miou\": " << fgMiou << ",\n";
        json << "    \"foreground_mdice\": " << fgMdice << ",\n";
        json << "    \"pixel_accuracy\": " << pixelAccuracy << ",\n";
        json << "    \"per_class\": {\n";
        static char const* CLASS_NAMES[] = {"background", "crop", "weed"};
        for (int cls = 0; cls < channels; ++cls)
        {
            std::string className = (cls < 3) ? CLASS_NAMES[cls] : ("class_" + std::to_string(cls));
            json << "      \"" << jsonEscape(className) << "\": {\n";
            json << "        \"iou\": " << perClassIou[static_cast<std::size_t>(cls)] << ",\n";
            json << "        \"dice\": " << perClassDice[static_cast<std::size_t>(cls)] << ",\n";
            json << "        \"recall\": " << perClassRecall[static_cast<std::size_t>(cls)] << ",\n";
            json << "        \"precision\": " << perClassPrecision[static_cast<std::size_t>(cls)] << ",\n";
            json << "        \"support_pixels\": " << perClassSupport[static_cast<std::size_t>(cls)] << "\n";
            json << "      }" << (cls + 1 < channels ? "," : "") << "\n";
        }
        json << "    },\n";
        json << "    \"confusion_matrix\": [\n";
        for (int row = 0; row < channels; ++row)
        {
            json << "      [";
            for (int col = 0; col < channels; ++col)
            {
                json << confusion[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
                if (col + 1 < channels)
                {
                    json << ", ";
                }
            }
            json << "]" << (row + 1 < channels ? "," : "") << "\n";
        }
        json << "    ]\n";
        json << "  }\n";
        json << "}\n";

        writeText(args.jsonOut, json.str());
        std::cout << json.str();
        return 0;
    }
    catch (std::exception const& exc)
    {
        std::cerr << exc.what() << '\n';
        return 1;
    }
}
