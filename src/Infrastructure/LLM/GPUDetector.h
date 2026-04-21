#pragma once
#include "Core/Models/ModelStatus.h"
#include <string>
#include <cstddef>

namespace LocalNotebookLLM::Infrastructure {

    /// @brief Detects GPU hardware via DXGI for VRAM-aware model placement.
    /// Used by MemoryOrchestrator to decide CPU vs GPU offloading.
    class GPUDetector {
    public:
        /// Detect the primary dedicated GPU (if any).
        [[nodiscard]]
        static Core::GPUInfo Detect();

        /// Compute recommended GPU layer count for a given model size + available VRAM.
        /// Returns 0 if CPU-only is recommended.
        [[nodiscard]]
        static int RecommendGPULayers(size_t modelSizeMB, size_t freeVramMB);
    };

} // namespace LocalNotebookLLM::Infrastructure
