#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace
{

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, char const* msg) noexcept override
    {
        if (severity > Severity::kWARNING)
        {
            return;
        }
        std::cerr << "[TRT] " << msg << std::endl;
    }
};

template <typename T>
using UniquePtr = std::unique_ptr<T>;

struct Args
{
    fs::path onnxPath;
    fs::path outputDir;
    fs::path datasetRoot;
    std::optional<fs::path> manifestPath;
    fs::path cachePath;
    fs::path enginePath;
    fs::path metaPath;
    int imgsz{640};
    int batch{1};
    int workspaceMb{4096};
    int maxImages{256};
    bool allowFp16Fallback{true};
};

std::string jsonEscape(std::string_view value)
{
    std::ostringstream oss;
    for (char ch : value)
    {
        switch (ch)
        {
        case '\\':
            oss << "\\\\";
            break;
        case '"':
            oss << "\\\"";
            break;
        case '\b':
            oss << "\\b";
            break;
        case '\f':
            oss << "\\f";
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
            if (static_cast<unsigned char>(ch) < 0x20)
            {
                oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch);
            }
            else
            {
                oss << ch;
            }
            break;
        }
    }
    return oss.str();
}

std::string utcNowIso()
{
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&tt, &tm);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S") << '.' << std::setw(3) << std::setfill('0') << ms.count() << "Z";
    return oss.str();
}

std::string tensorrtVersionString()
{
    std::ostringstream oss;
    oss << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.' << NV_TENSORRT_PATCH;
    return oss.str();
}

std::string getenvOr(std::string const& key, std::string const& fallback)
{
    char const* value = std::getenv(key.c_str());
    if (!value || !*value)
    {
        return fallback;
    }
    return std::string(value);
}

bool hasImageExtension(fs::path const& path)
{
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

std::vector<fs::path> readManifest(fs::path const& datasetRoot, fs::path const& manifestPath)
{
    std::ifstream in(manifestPath);
    if (!in)
    {
        throw std::runtime_error("Could not open manifest: " + manifestPath.string());
    }
    std::vector<fs::path> paths;
    std::string line;
    while (std::getline(in, line))
    {
        auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#')
        {
            continue;
        }
        auto last = line.find_last_not_of(" \t\r\n");
        std::string trimmed = line.substr(first, last - first + 1);
        fs::path path = fs::path(trimmed);
        if (!path.is_absolute())
        {
            path = datasetRoot / path;
        }
        if (fs::exists(path) && hasImageExtension(path))
        {
            paths.push_back(fs::canonical(path));
        }
    }
    return paths;
}

std::vector<fs::path> collectImages(fs::path const& root)
{
    std::vector<fs::path> paths;
    for (auto const& entry : fs::recursive_directory_iterator(root))
    {
        if (entry.is_regular_file() && hasImageExtension(entry.path()))
        {
            paths.push_back(fs::canonical(entry.path()));
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

std::vector<fs::path> resolveImageList(fs::path const& datasetRoot, std::optional<fs::path> const& manifestPath)
{
    if (manifestPath && fs::exists(*manifestPath))
    {
        auto manifestImages = readManifest(datasetRoot, *manifestPath);
        if (!manifestImages.empty())
        {
            return manifestImages;
        }
    }
    for (fs::path const& candidate : {
             datasetRoot / "images" / "calibration",
             datasetRoot / "images" / "train",
             datasetRoot / "images" / "val",
             datasetRoot / "images" / "test",
         })
    {
        if (fs::is_directory(candidate))
        {
            auto images = collectImages(candidate);
            if (!images.empty())
            {
                return images;
            }
        }
    }
    throw std::runtime_error("Could not find calibration images under " + datasetRoot.string());
}

void cudaCheck(cudaError_t code, std::string const& op)
{
    if (code != cudaSuccess)
    {
        throw std::runtime_error(op + " failed: " + cudaGetErrorString(code));
    }
}

struct PreprocessedImage
{
    std::vector<float> chw;
    int origH{0};
    int origW{0};
};

PreprocessedImage preprocessImage(fs::path const& imagePath, int channels, int imgsz)
{
    cv::Mat bgr = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
    if (bgr.empty())
    {
        throw std::runtime_error("Could not read image: " + imagePath.string());
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    int srcW = rgb.cols;
    int srcH = rgb.rows;
    float scale = std::min(static_cast<float>(imgsz) / std::max(srcW, 1), static_cast<float>(imgsz) / std::max(srcH, 1));
    int newW = std::max(1, static_cast<int>(std::round(srcW * scale)));
    int newH = std::max(1, static_cast<int>(std::round(srcH * scale)));

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(newW, newH), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat canvas(imgsz, imgsz, CV_8UC3, cv::Scalar(114, 114, 114));
    int padX = (imgsz - newW) / 2;
    int padY = (imgsz - newH) / 2;
    resized.copyTo(canvas(cv::Rect(padX, padY, newW, newH)));

    cv::Mat floatImg;
    canvas.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> splitChannels;
    cv::split(floatImg, splitChannels);
    if (static_cast<int>(splitChannels.size()) != channels)
    {
        throw std::runtime_error("Unexpected channel count while preprocessing " + imagePath.string());
    }

    std::vector<float> chw(static_cast<size_t>(channels) * imgsz * imgsz);
    size_t channelStride = static_cast<size_t>(imgsz) * imgsz;
    for (int c = 0; c < channels; ++c)
    {
        std::memcpy(chw.data() + static_cast<size_t>(c) * channelStride, splitChannels[c].data, channelStride * sizeof(float));
    }
    return PreprocessedImage{std::move(chw), srcH, srcW};
}

size_t elementSizeBytes(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        return sizeof(float);
    case nvinfer1::DataType::kHALF:
        return sizeof(uint16_t);
    case nvinfer1::DataType::kINT8:
        return sizeof(int8_t);
    case nvinfer1::DataType::kINT32:
        return sizeof(int32_t);
    case nvinfer1::DataType::kINT64:
        return sizeof(int64_t);
    case nvinfer1::DataType::kBOOL:
        return sizeof(bool);
#if NV_TENSORRT_MAJOR >= 10
    case nvinfer1::DataType::kUINT8:
        return sizeof(uint8_t);
#endif
    default:
        throw std::runtime_error("Unsupported TensorRT datatype for INT8 calibration helper");
    }
}

size_t dimsElementCount(nvinfer1::Dims const& dims)
{
    if (dims.nbDims <= 0)
    {
        return 0;
    }
    size_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] <= 0)
        {
            throw std::runtime_error("Dynamic dimensions are not supported by this INT8 calibration helper without an explicit profile.");
        }
        count *= static_cast<size_t>(dims.d[i]);
    }
    return count;
}

struct InputSpec
{
    std::string name;
    nvinfer1::DataType dtype;
    nvinfer1::Dims dims;
    bool isImage{false};
    size_t elementCount{0};
    size_t byteCount{0};
    void* devicePtr{nullptr};
};

class ImageEntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    ImageEntropyCalibrator(
        std::vector<fs::path> imagePaths,
        fs::path cachePath,
        int batchSize,
        int channels,
        int imgsz,
        int maxImages,
        std::vector<InputSpec> inputSpecs)
        : mImagePaths(std::move(imagePaths))
        , mCachePath(std::move(cachePath))
        , mBatchSize(batchSize)
        , mChannels(channels)
        , mImgsz(imgsz)
        , mInputSpecs(std::move(inputSpecs))
    {
        if (maxImages > 0 && static_cast<int>(mImagePaths.size()) > maxImages)
        {
            mImagePaths.resize(static_cast<size_t>(maxImages));
        }
        mInputCount = static_cast<size_t>(mBatchSize) * mChannels * mImgsz * mImgsz;
        for (auto& spec : mInputSpecs)
        {
            spec.elementCount = spec.isImage ? mInputCount : dimsElementCount(spec.dims);
            spec.byteCount = spec.elementCount * elementSizeBytes(spec.dtype);
            cudaCheck(cudaMalloc(&spec.devicePtr, spec.byteCount), "cudaMalloc");
        }
    }

    ~ImageEntropyCalibrator() noexcept override
    {
        for (auto& spec : mInputSpecs)
        {
            if (spec.devicePtr != nullptr)
            {
                cudaFree(spec.devicePtr);
            }
        }
    }

    int32_t getBatchSize() const noexcept override
    {
        return mBatchSize;
    }

    bool getBatch(void* bindings[], char const*[], int32_t nbBindings) noexcept override
    {
        try
        {
            if (mIndex >= mImagePaths.size())
            {
                return false;
            }

            std::vector<float> hostBuffer(mInputCount, 0.0f);
            size_t imageStride = static_cast<size_t>(mChannels) * mImgsz * mImgsz;
            std::vector<std::pair<int, int>> batchOrigSizes(static_cast<size_t>(mBatchSize), {mImgsz, mImgsz});
            for (int batchIdx = 0; batchIdx < mBatchSize && mIndex < mImagePaths.size(); ++batchIdx, ++mIndex)
            {
                auto preprocessed = preprocessImage(mImagePaths[mIndex], mChannels, mImgsz);
                std::copy(preprocessed.chw.begin(), preprocessed.chw.end(), hostBuffer.begin() + static_cast<size_t>(batchIdx) * imageStride);
                batchOrigSizes[static_cast<size_t>(batchIdx)] = {preprocessed.origH, preprocessed.origW};
            }

            for (auto& spec : mInputSpecs)
            {
                if (spec.isImage)
                {
                    cudaCheck(cudaMemcpy(spec.devicePtr, hostBuffer.data(), hostBuffer.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy(images)");
                    continue;
                }

                if (spec.dtype == nvinfer1::DataType::kINT64 && spec.dims.nbDims == 2 && spec.dims.d[1] == 2)
                {
                    std::vector<int64_t> host(spec.elementCount, 0);
                    size_t rows = static_cast<size_t>(std::max<int64_t>(spec.dims.d[0], 1));
                    for (size_t row = 0; row < rows && row < batchOrigSizes.size(); ++row)
                    {
                        host[row * 2] = static_cast<int64_t>(batchOrigSizes[row].first);
                        host[row * 2 + 1] = static_cast<int64_t>(batchOrigSizes[row].second);
                    }
                    cudaCheck(cudaMemcpy(spec.devicePtr, host.data(), spec.byteCount, cudaMemcpyHostToDevice), "cudaMemcpy(aux_int64)");
                    continue;
                }

                if (spec.dtype == nvinfer1::DataType::kINT32 && spec.dims.nbDims == 2 && spec.dims.d[1] == 2)
                {
                    std::vector<int32_t> host(spec.elementCount, 0);
                    size_t rows = static_cast<size_t>(std::max<int64_t>(spec.dims.d[0], 1));
                    for (size_t row = 0; row < rows && row < batchOrigSizes.size(); ++row)
                    {
                        host[row * 2] = static_cast<int32_t>(batchOrigSizes[row].first);
                        host[row * 2 + 1] = static_cast<int32_t>(batchOrigSizes[row].second);
                    }
                    cudaCheck(cudaMemcpy(spec.devicePtr, host.data(), spec.byteCount, cudaMemcpyHostToDevice), "cudaMemcpy(aux_int32)");
                    continue;
                }

                std::vector<uint8_t> zeros(spec.byteCount, 0);
                cudaCheck(cudaMemcpy(spec.devicePtr, zeros.data(), spec.byteCount, cudaMemcpyHostToDevice), "cudaMemcpy(aux_zero)");
            }

            for (int32_t i = 0; i < nbBindings && i < static_cast<int32_t>(mInputSpecs.size()); ++i)
            {
                bindings[i] = mInputSpecs[static_cast<size_t>(i)].devicePtr;
            }
            return true;
        }
        catch (std::exception const& exc)
        {
            std::cerr << "Calibration batch failed: " << exc.what() << std::endl;
            return false;
        }
    }

    void const* readCalibrationCache(std::size_t& length) noexcept override
    {
        mCache.clear();
        if (fs::exists(mCachePath))
        {
            std::ifstream in(mCachePath, std::ios::binary);
            mCache.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        }
        length = mCache.size();
        return mCache.empty() ? nullptr : mCache.data();
    }

    void writeCalibrationCache(void const* ptr, std::size_t length) noexcept override
    {
        std::ofstream out(mCachePath, std::ios::binary);
        out.write(static_cast<char const*>(ptr), static_cast<std::streamsize>(length));
    }

private:
    std::vector<fs::path> mImagePaths;
    fs::path mCachePath;
    int mBatchSize;
    int mChannels;
    int mImgsz;
    size_t mInputCount{0};
    size_t mIndex{0};
    std::vector<InputSpec> mInputSpecs;
    std::vector<char> mCache;
};

Args parseArgs(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i)
    {
        std::string key = argv[i];
        auto requireValue = [&](std::string const& flag) -> std::string {
            if (i + 1 >= argc)
            {
                throw std::runtime_error("Missing value for " + flag);
            }
            return argv[++i];
        };

        if (key == "--onnx")
        {
            args.onnxPath = requireValue(key);
        }
        else if (key == "--output-dir")
        {
            args.outputDir = requireValue(key);
        }
        else if (key == "--dataset-root")
        {
            args.datasetRoot = requireValue(key);
        }
        else if (key == "--manifest")
        {
            args.manifestPath = fs::path(requireValue(key));
        }
        else if (key == "--imgsz")
        {
            args.imgsz = std::stoi(requireValue(key));
        }
        else if (key == "--batch")
        {
            args.batch = std::stoi(requireValue(key));
        }
        else if (key == "--workspace-mb")
        {
            args.workspaceMb = std::stoi(requireValue(key));
        }
        else if (key == "--max-images")
        {
            args.maxImages = std::stoi(requireValue(key));
        }
        else if (key == "--allow-fp16-fallback")
        {
            args.allowFp16Fallback = true;
        }
        else if (key == "--no-fp16-fallback")
        {
            args.allowFp16Fallback = false;
        }
        else if (key == "--cache-path")
        {
            args.cachePath = requireValue(key);
        }
        else if (key == "--engine-path")
        {
            args.enginePath = requireValue(key);
        }
        else if (key == "--meta-path")
        {
            args.metaPath = requireValue(key);
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }

    if (args.onnxPath.empty() || args.outputDir.empty() || args.datasetRoot.empty())
    {
        throw std::runtime_error("Required arguments: --onnx --output-dir --dataset-root");
    }

    args.onnxPath = fs::weakly_canonical(args.onnxPath);
    args.outputDir = fs::weakly_canonical(args.outputDir);
    args.datasetRoot = fs::weakly_canonical(args.datasetRoot);
    if (args.manifestPath && !args.manifestPath->empty())
    {
        *args.manifestPath = fs::weakly_canonical(*args.manifestPath);
    }
    if (args.cachePath.empty())
    {
        args.cachePath = args.outputDir / (args.onnxPath.stem().string() + "_int8_calibration.cache");
    }
    if (args.enginePath.empty())
    {
        args.enginePath = args.outputDir / (args.onnxPath.stem().string() + "_int8.engine");
    }
    if (args.metaPath.empty())
    {
        args.metaPath = args.outputDir / (args.onnxPath.stem().string() + "_int8_cache_build_meta.json");
    }
    args.cachePath = fs::weakly_canonical(args.cachePath);
    args.enginePath = fs::weakly_canonical(args.enginePath);
    args.metaPath = fs::weakly_canonical(args.metaPath);
    return args;
}

std::string dimsToJson(nvinfer1::Dims const& dims)
{
    std::ostringstream oss;
    oss << '[';
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (i)
        {
            oss << ", ";
        }
        oss << dims.d[i];
    }
    oss << ']';
    return oss.str();
}

int inferChannels(nvinfer1::Dims const& dims)
{
    if (dims.nbDims < 2)
    {
        return 3;
    }
    int value = dims.d[1];
    return value > 0 ? value : 3;
}

bool hasDynamicDims(nvinfer1::Dims const& dims)
{
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] <= 0)
        {
            return true;
        }
    }
    return false;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        Args args = parseArgs(argc, argv);
        fs::create_directories(args.outputDir);
        fs::create_directories(args.enginePath.parent_path());
        fs::create_directories(args.cachePath.parent_path());

        auto imagePaths = resolveImageList(args.datasetRoot, args.manifestPath);
        if (imagePaths.empty())
        {
            throw std::runtime_error("No calibration images found.");
        }

        Logger logger;
        UniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
        if (!builder)
        {
            throw std::runtime_error("createInferBuilder returned nullptr");
        }

        constexpr uint32_t kExplicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        UniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(kExplicitBatch)};
        if (!network)
        {
            throw std::runtime_error("createNetworkV2 returned nullptr");
        }

        UniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, logger)};
        if (!parser)
        {
            throw std::runtime_error("createParser returned nullptr");
        }
        if (!parser->parseFromFile(args.onnxPath.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
        {
            std::ostringstream oss;
            oss << "TensorRT ONNX parse failed";
            for (int i = 0; i < parser->getNbErrors(); ++i)
            {
                oss << " | " << parser->getError(i)->desc();
            }
            throw std::runtime_error(oss.str());
        }

        if (network->getNbInputs() < 1)
        {
            throw std::runtime_error("Parsed network has no inputs");
        }
        std::vector<InputSpec> inputSpecs;
        nvinfer1::ITensor* imageInputTensor = nullptr;
        for (int inputIdx = 0; inputIdx < network->getNbInputs(); ++inputIdx)
        {
            auto* tensor = network->getInput(inputIdx);
            InputSpec spec;
            spec.name = tensor->getName();
            spec.dtype = tensor->getType();
            spec.dims = tensor->getDimensions();
            spec.isImage = (spec.dtype == nvinfer1::DataType::kFLOAT && spec.dims.nbDims == 4);
            if (spec.isImage && imageInputTensor == nullptr)
            {
                imageInputTensor = tensor;
            }
            inputSpecs.push_back(spec);
        }
        if (imageInputTensor == nullptr)
        {
            throw std::runtime_error("Could not identify the image input tensor for INT8 calibration");
        }
        auto inputDims = imageInputTensor->getDimensions();
        int channels = inferChannels(inputDims);

        UniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
        if (!config)
        {
            throw std::runtime_error("createBuilderConfig returned nullptr");
        }
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, static_cast<std::size_t>(args.workspaceMb) * (1ULL << 20));
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        if (args.allowFp16Fallback)
        {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        ImageEntropyCalibrator calibrator(imagePaths, args.cachePath, args.batch, channels, args.imgsz, args.maxImages, inputSpecs);
        config->setInt8Calibrator(&calibrator);

        bool needsProfile = false;
        for (auto const& spec : inputSpecs)
        {
            if (hasDynamicDims(spec.dims))
            {
                needsProfile = true;
                break;
            }
        }

        if (needsProfile)
        {
            auto* profile = builder->createOptimizationProfile();
            if (!profile)
            {
                throw std::runtime_error("createOptimizationProfile returned nullptr");
            }
            bool ok = true;
            for (auto const& spec : inputSpecs)
            {
                nvinfer1::Dims shape = spec.dims;
                for (int dimIdx = 0; dimIdx < shape.nbDims; ++dimIdx)
                {
                    if (shape.d[dimIdx] > 0)
                    {
                        continue;
                    }
                    if (spec.isImage)
                    {
                        if (dimIdx == 0)
                        {
                            shape.d[dimIdx] = args.batch;
                        }
                        else if (dimIdx == 1)
                        {
                            shape.d[dimIdx] = channels;
                        }
                        else
                        {
                            shape.d[dimIdx] = args.imgsz;
                        }
                    }
                    else
                    {
                        shape.d[dimIdx] = (dimIdx == 0) ? args.batch : 2;
                    }
                }
                ok = ok
                    && profile->setDimensions(spec.name.c_str(), nvinfer1::OptProfileSelector::kMIN, shape)
                    && profile->setDimensions(spec.name.c_str(), nvinfer1::OptProfileSelector::kOPT, shape)
                    && profile->setDimensions(spec.name.c_str(), nvinfer1::OptProfileSelector::kMAX, shape);
            }
            if (!ok)
            {
                throw std::runtime_error("Failed to set optimization profile dimensions for calibration");
            }
            if (!config->addOptimizationProfile(profile))
            {
                throw std::runtime_error("Failed to add optimization profile");
            }
        }

        UniquePtr<nvinfer1::IHostMemory> serialized{builder->buildSerializedNetwork(*network, *config)};
        if (!serialized)
        {
            std::ostringstream oss;
            oss << "buildSerializedNetwork returned nullptr";
            if (network->getNbInputs() > 1)
            {
                oss << ". Multi-input ONNX detected; auxiliary bindings may be unsupported by this TensorRT path.";
            }
            throw std::runtime_error(oss.str());
        }

        {
            std::ofstream out(args.enginePath, std::ios::binary);
            out.write(static_cast<char const*>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
        }
        if (!fs::exists(args.cachePath))
        {
            throw std::runtime_error("Calibration cache was not written: " + args.cachePath.string());
        }

        std::ostringstream json;
        json << "{\n"
             << "  \"timestamp_utc\": \"" << jsonEscape(utcNowIso()) << "\",\n"
             << "  \"onnx_path\": \"" << jsonEscape(args.onnxPath.string()) << "\",\n"
             << "  \"engine_path\": \"" << jsonEscape(args.enginePath.string()) << "\",\n"
             << "  \"cache_path\": \"" << jsonEscape(args.cachePath.string()) << "\",\n"
             << "  \"dataset_root\": \"" << jsonEscape(args.datasetRoot.string()) << "\",\n"
             << "  \"manifest_path\": " << (args.manifestPath ? ("\"" + jsonEscape(args.manifestPath->string()) + "\"") : "null") << ",\n"
             << "  \"imgsz\": " << args.imgsz << ",\n"
             << "  \"batch_size\": " << args.batch << ",\n"
             << "  \"workspace_mb\": " << args.workspaceMb << ",\n"
             << "  \"max_images\": " << args.maxImages << ",\n"
             << "  \"allow_fp16_fallback\": " << (args.allowFp16Fallback ? "true" : "false") << ",\n"
             << "  \"calibration_image_count\": " << std::min<int>(static_cast<int>(imagePaths.size()), args.maxImages) << ",\n"
             << "  \"input_name\": \"" << jsonEscape(imageInputTensor->getName()) << "\",\n"
             << "  \"input_dims\": " << dimsToJson(inputDims) << ",\n"
             << "  \"input_count\": " << network->getNbInputs() << ",\n"
             << "  \"tensorrt_version\": \"" << jsonEscape(tensorrtVersionString()) << "\",\n"
             << "  \"tensorrt_stack_tag\": \"" << jsonEscape(getenvOr("TENSORRT_STACK_TAG", "system-trt")) << "\",\n"
             << "  \"helper\": \"cpp_tensorrt_opencv\"\n"
             << "}\n";

        std::ofstream metaOut(args.metaPath, std::ios::binary);
        metaOut << json.str();
        metaOut.close();

        std::cout << json.str();
        return 0;
    }
    catch (std::exception const& exc)
    {
        std::cerr << exc.what() << std::endl;
        return 1;
    }
}
