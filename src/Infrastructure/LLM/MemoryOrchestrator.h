#pragma once
#include "Core/Interfaces/IMemoryOrchestrator.h"
#include "Core/Interfaces/ILLMProvider.h"
#include "Core/Interfaces/IEmbeddingProvider.h"
#include <memory>
#include <mutex>
#include <filesystem>

namespace LocalNotebookLLM::Infrastructure {

    /// @brief Manages model lifecycle across VRAM and System RAM pools.
    /// Enforces the invariant: Worker and Reasoner are never both hot
    /// unless total available memory permits it.
    class MemoryOrchestrator : public Core::IMemoryOrchestrator {
    public:
        struct ModelConfig {
            std::filesystem::path modelPath;
            Core::ModelParams     params;
            size_t                estimatedSizeMB = 0;
        };

        MemoryOrchestrator(
            std::shared_ptr<Core::ILLMProvider> workerProvider,
            std::shared_ptr<Core::ILLMProvider> reasonerProvider,
            std::shared_ptr<Core::IEmbeddingProvider> embeddingProvider = nullptr
        );
        ~MemoryOrchestrator() override = default;

        /// Configure model paths before use.
        void SetWorkerConfig(ModelConfig config);
        void SetReasonerConfig(ModelConfig config);
        void SetEmbeddingModelPath(const std::filesystem::path& path);

        /// Prevent the Orchestrator from evicting the embedding model during
        /// active document ingestion. Set true before GenerateEmbeddingsBackground,
        /// false after it completes to restore normal eviction behaviour.
        void SetProtectEmbedding(bool protect);

        [[nodiscard]]
        std::expected<void, std::string>
        RequestModel(Core::ModelRole role) override;

        void ReleaseModel(Core::ModelRole role) override;

        [[nodiscard]] Core::MemorySnapshot GetSnapshot() const override;
        void SetMemoryBudget(size_t maxModelMemoryMB) override;
        [[nodiscard]] Core::GPUInfo DetectGPU() const override;

    private:
        std::shared_ptr<Core::ILLMProvider>       m_workerProvider;
        std::shared_ptr<Core::ILLMProvider>       m_reasonerProvider;
        std::shared_ptr<Core::IEmbeddingProvider> m_embeddingProvider;

        ModelConfig m_workerConfig;
        ModelConfig m_reasonerConfig;
        std::filesystem::path m_embeddingModelPath;

        size_t m_memoryBudgetMB  = 8192;  // Default 8 GB budget for models
        bool   m_protectEmbedding = false; // When true, Reasoner load will NOT evict embedding
        mutable std::mutex m_mutex;

        /// Get available system RAM in MB.
        [[nodiscard]] static size_t GetAvailableSystemRAM();
        /// Get total system RAM in MB.
        [[nodiscard]] static size_t GetTotalSystemRAM();
        /// Check if a model role can be loaded within current memory budget.
        /// IMPORTANT: Must be called while holding m_mutex.
        [[nodiscard]] bool CanLoad(Core::ModelRole role) const;
    };

} // namespace LocalNotebookLLM::Infrastructure
