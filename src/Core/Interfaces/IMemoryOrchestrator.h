#pragma once
#include "Core/Models/ModelStatus.h"
#include <expected>
#include <string>
#include <cstddef>

namespace LocalNotebookLLM::Core {

    /// @brief Manages model lifecycle across VRAM and System RAM pools.
    /// Enforces the invariant: Worker and Reasoner are never both hot in the
    /// same memory pool unless total available memory permits it.
    class IMemoryOrchestrator {
    public:
        virtual ~IMemoryOrchestrator() = default;

        /// Request a model role to be made "hot" (loaded + ready).
        /// May trigger eviction of another model to free memory.
        [[nodiscard]]
        virtual std::expected<void, std::string>
        RequestModel(ModelRole role) = 0;

        /// Release a model role to "cold" (unloaded, mmap handle retained).
        virtual void ReleaseModel(ModelRole role) = 0;

        [[nodiscard]] virtual MemorySnapshot GetSnapshot() const = 0;
        [[nodiscard]] virtual bool CanLoad(ModelRole role) const = 0;

        /// Set the hard ceiling (in MB) for total model memory.
        virtual void SetMemoryBudget(size_t maxModelMemoryMB) = 0;

        /// Detect available GPU hardware.
        [[nodiscard]] virtual GPUInfo DetectGPU() const = 0;
    };

} // namespace LocalNotebookLLM::Core
