#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
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

std::vector<float> preprocessImage(fs::path const& imagePath, int channels, int imgsz)
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
    return chw;
}

class ImageEntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    ImageEntropyCalibrator(
        std::vector<fs::path> imagePaths, fs::path cachePath, int batchSize, int channels, int imgsz, int maxImages)
        : mImagePaths(std::move(imagePaths))
        , mCachePath(std::move(cachePath))
        , mBatchSize(batchSize)
        , mChannels(channels)
        , mImgsz(imgsz)
    {
        if (maxImages > 0 && static_cast<int>(mImagePaths.size()) > maxImages)
        {
            mImagePaths.resize(static_cast<size_t>(maxImages));
        }
        mInputCount = static_cast<size_t>(mBatchSize) * mChannels * mImgsz * mImgsz;
        cudaCheck(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)), "cudaMalloc");
    }

    ~ImageEntropyCalibrator() noexcept override
    {
        if (mDeviceInput != nullptr)
        {
            cudaFree(mDeviceInput);
        }
    }

    int32_t getBatchSize() const noexcept override
    {
        return mBatchSize;
    }

    bool getBatch(void* bindings[], char const*[], int32_t) noexcept override
    {
        try
        {
            if (mIndex >= mImagePaths.size())
            {
                return false;
            }

            std::vector<float> hostBuffer(mInputCount, 0.0f);
            size_t imageStride = static_cast<size_t>(mChannels) * mImgsz * mImgsz;
            for (int batchIdx = 0; batchIdx < mBatchSize && mIndex < mImagePaths.size(); ++batchIdx, ++mIndex)
            {
                auto chw = preprocessImage(mImagePaths[mIndex], mChannels, mImgsz);
                std::copy(chw.begin(), chw.end(), hostBuffer.begin() + static_cast<size_t>(batchIdx) * imageStride);
            }

            cudaCheck(cudaMemcpy(mDeviceInput, hostBuffer.data(), hostBuffer.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
            bindings[0] = mDeviceInput;
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
    void* mDeviceInput{nullptr};
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
        auto* inputTensor = network->getInput(0);
        auto inputDims = inputTensor->getDimensions();
        int channels = inferChannels(inputDims);

        UniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
        if (!config)
        {
            throw std::runtime_error("createBuilderConfig returned nullptr");
        }
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, static_cast<std::size_t>(args.workspaceMb) * (1ULL << 20));
        config->setFlag(nvinfer1::BuilderFlag::kINT8);

        ImageEntropyCalibrator calibrator(imagePaths, args.cachePath, args.batch, channels, args.imgsz, args.maxImages);
        config->setInt8Calibrator(&calibrator);

        if (hasDynamicDims(inputDims))
        {
            auto* profile = builder->createOptimizationProfile();
            if (!profile)
            {
                throw std::runtime_error("createOptimizationProfile returned nullptr");
            }
            nvinfer1::Dims4 shape{args.batch, channels, args.imgsz, args.imgsz};
            bool ok = profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMIN, shape)
                && profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kOPT, shape)
                && profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMAX, shape);
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
            throw std::runtime_error("buildSerializedNetwork returned nullptr");
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
             << "  \"calibration_image_count\": " << std::min<int>(static_cast<int>(imagePaths.size()), args.maxImages) << ",\n"
             << "  \"input_name\": \"" << jsonEscape(inputTensor->getName()) << "\",\n"
             << "  \"input_dims\": " << dimsToJson(inputDims) << ",\n"
             << "  \"tensorrt_version\": \"" << jsonEscape(tensorrtVersionString()) << "\",\n"
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
