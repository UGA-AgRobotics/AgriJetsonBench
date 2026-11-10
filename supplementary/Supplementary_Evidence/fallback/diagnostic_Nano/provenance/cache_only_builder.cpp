#include <NvInfer.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>

class BuildLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, char const* message) noexcept override {
        std::fprintf(stderr, "[TRT %d] %s\n", static_cast<int>(severity), message);
    }
};

class CacheOnlyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    explicit CacheOnlyCalibrator(char const* path) {
        std::ifstream stream(path, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(stream), {});
        if (bytes.empty()) {
            throw std::runtime_error("The copied calibration cache is absent or empty");
        }
    }

    int32_t getBatchSize() const noexcept override { return 1; }

    bool getBatch(void*[], char const*[], int32_t) noexcept override {
        stop("getBatch", 86);
    }

    void const* readCalibrationCache(std::size_t& length) noexcept override {
        ++reads;
        length = bytes.size();
        std::fprintf(stderr, "[CACHE ONLY] readCalibrationCache call %d, bytes %zu\n", reads, length);
        return bytes.data();
    }

    void writeCalibrationCache(void const*, std::size_t) noexcept override {
        stop("writeCalibrationCache", 87);
    }

    int reads{0};

private:
    [[noreturn]] static void stop(char const* callback, int status) noexcept {
        std::fprintf(stderr, "[CACHE ONLY] STOP: TensorRT invoked %s. Exiting with status %d.\n", callback, status);
        std::fflush(stderr);
        std::_Exit(status);
    }

    std::vector<char> bytes;
};

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            throw std::runtime_error("Usage: cache_only_builder ONNX COPIED_CACHE DIAGNOSTIC_PLAN");
        }
        // Match the historical helper's explicit-batch, ONNX parser and builder
        // configuration. The fixed input needs no optimization profile.
        BuildLogger logger;
        CacheOnlyCalibrator calibrator(argv[2]);
        std::fprintf(stderr, "[BUILD] TensorRT headers %d.%d.%d.%d, runtime version code %d\n",
                     NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, NV_TENSORRT_BUILD,
                     getInferLibVersion());
        std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
        if (!builder) throw std::runtime_error("Could not create TensorRT builder");
        constexpr auto explicitBatch = 1U << static_cast<unsigned>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicitBatch));
        if (!network) throw std::runtime_error("Could not create TensorRT network");
        std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
        if (!parser || !parser->parseFromFile(argv[1], static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            throw std::runtime_error("Could not parse the historical ONNX");
        }
        if (network->getNbInputs() != 1) throw std::runtime_error("Expected one static ONNX input");
        auto const* input = network->getInput(0);
        auto const shape = input->getDimensions();
        if (shape.nbDims != 4 || shape.d[0] != 1 || shape.d[1] != 3 ||
            shape.d[2] != 640 || shape.d[3] != 640 ||
            input->getType() != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("Expected an FP32 ONNX input of shape 1x3x640x640");
        }
        std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        if (!config) throw std::runtime_error("Could not create TensorRT builder config");
        constexpr std::size_t workspace = 4096ULL * (1ULL << 20);
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace);
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
        config->setInt8Calibrator(&calibrator);
        std::fprintf(stderr, "[BUILD] input=%s 1x3x640x640, workspace=%zu bytes, INT8=1, FP16=1, TF32 default=%d, profiling=kDETAILED\n",
                     input->getName(), config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE),
                     static_cast<int>(config->getFlag(nvinfer1::BuilderFlag::kTF32)));
        std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
        if (!plan) throw std::runtime_error("TensorRT failed to build the diagnostic plan");
        if (calibrator.reads < 1) throw std::runtime_error("TensorRT did not request the copied cache");
        std::ofstream output(argv[3], std::ios::binary);
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output.write(static_cast<char const*>(plan->data()), static_cast<std::streamsize>(plan->size()));
        output.close();
        std::fprintf(stderr, "[CACHE ONLY] BUILD SUCCESS: reads=%d, batch_requests=0, cache_write_requests=0, plan_bytes=%zu\n",
                     calibrator.reads, plan->size());
        return 0;
    } catch (std::exception const& error) {
        std::cerr << "[BUILD] " << error.what() << '\n';
        return 1;
    }
}
