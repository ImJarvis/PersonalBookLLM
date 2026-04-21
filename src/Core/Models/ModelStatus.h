#pragma once
#include <string>
#include <cstddef>

namespace LocalNotebookLLM::Core {

    struct ModelStatus {
        bool        isLoaded        = false;
        std::string modelName;
        std::string modelPath;
        size_t      modelSizeMB     = 0;
        size_t      contextWindow   = 0;   // In tokens
        size_t      kvCacheSizeMB   = 0;
        std::string kvCacheType;           // "F16", "Q8_0", "Q4_0"
        std::string device;                // "CPU", "GPU", "Auto"
        int         gpuLayers       = 0;
    };

    enum class ModelRole { Worker, Reasoner, Embedding };

    struct MemorySnapshot {
        size_t totalPhysicalMB  = 0;
        size_t availableMB      = 0;
        size_t workerModelMB    = 0;   // 0 if unloaded
        size_t reasonerModelMB  = 0;   // 0 if unloaded
        size_t embeddingModelMB = 0;   // 0 if unloaded
        size_t sqliteCacheMB    = 0;
        bool   workerOnGPU      = false;
        bool   reasonerOnGPU    = false;
    };

    struct GPUInfo {
        bool        hasDedicatedGPU = false;
        std::string gpuName;
        size_t      totalVramMB     = 0;
        size_t      freeVramMB      = 0;
    };

} // namespace LocalNotebookLLM::Core
