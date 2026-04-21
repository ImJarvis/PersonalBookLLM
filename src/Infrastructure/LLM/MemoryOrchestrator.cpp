#include "Infrastructure/LLM/MemoryOrchestrator.h"
#include "Infrastructure/LLM/GPUDetector.h"
#include "Core/Log.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace LocalNotebookLLM::Infrastructure {

    MemoryOrchestrator::MemoryOrchestrator(
        std::shared_ptr<Core::ILLMProvider> workerProvider,
        std::shared_ptr<Core::ILLMProvider> reasonerProvider,
        std::shared_ptr<Core::IEmbeddingProvider> embeddingProvider)
        : m_workerProvider(std::move(workerProvider))
        , m_reasonerProvider(std::move(reasonerProvider))
        , m_embeddingProvider(std::move(embeddingProvider))
    {
        LOG_INFO("MemOrch", "MemoryOrchestrator created (Worker + Reasoner providers)");
    }

    void MemoryOrchestrator::SetWorkerConfig(ModelConfig config) {
        std::lock_guard lock(m_mutex);
        m_workerConfig = std::move(config);
    }

    void MemoryOrchestrator::SetReasonerConfig(ModelConfig config) {
        std::lock_guard lock(m_mutex);
        m_reasonerConfig = std::move(config);
    }

    void MemoryOrchestrator::SetEmbeddingModelPath(const std::filesystem::path& path) {
        std::lock_guard lock(m_mutex);
        m_embeddingModelPath = path;
    }

    std::expected<void, std::string>
    MemoryOrchestrator::RequestModel(Core::ModelRole role) {
        std::lock_guard lock(m_mutex);

        switch (role) {
            case Core::ModelRole::Worker: {
                LOG_INFO("MemOrch", "RequestModel(Worker)");
                if (m_workerProvider->IsModelLoaded()) {
                    LOG_INFO("MemOrch", "  Worker already hot");
                    return {};
                }

                if (!CanLoad(Core::ModelRole::Worker)) {
                    LOG_INFO("MemOrch", "  Cannot fit Worker — evicting Reasoner");
                    if (m_reasonerProvider->IsModelLoaded()) {
                        m_reasonerProvider->UnloadModel();
                    }
                }

                if (m_workerConfig.modelPath.empty()) {
                    LOG_ERROR("MemOrch", "  Worker model path not configured");
                    return std::unexpected("Worker model path not configured");
                }

                LOG_INFO("MemOrch", "  Loading Worker: " + m_workerConfig.modelPath.string());
                auto result = m_workerProvider->LoadModel(
                    m_workerConfig.modelPath, m_workerConfig.params);
                if (!result) {
                    LOG_ERROR("MemOrch", "  Worker load FAILED: " + result.error().message);
                    return std::unexpected(result.error().message);
                }
                LOG_INFO("MemOrch", "  Worker loaded successfully");
                return {};
            }

            case Core::ModelRole::Reasoner: {
                LOG_INFO("MemOrch", "RequestModel(Reasoner)");
                if (m_reasonerProvider->IsModelLoaded()) {
                    LOG_INFO("MemOrch", "  Reasoner already hot");
                    return {};
                }

                LOG_INFO("MemOrch", "  Evicting Worker + Embedding for Reasoner");
                if (m_workerProvider->IsModelLoaded()) {
                    m_workerProvider->UnloadModel();
                }
                if (m_embeddingProvider && m_embeddingProvider->IsLoaded()) {
                    m_embeddingProvider->Unload();
                }

                if (m_reasonerConfig.modelPath.empty()) {
                    LOG_ERROR("MemOrch", "  Reasoner model path not configured");
                    return std::unexpected("Reasoner model path not configured");
                }

                LOG_INFO("MemOrch", "  Loading Reasoner: " + m_reasonerConfig.modelPath.string());
                auto result = m_reasonerProvider->LoadModel(
                    m_reasonerConfig.modelPath, m_reasonerConfig.params);
                if (!result) {
                    LOG_ERROR("MemOrch", "  Reasoner load FAILED: " + result.error().message);
                    return std::unexpected(result.error().message);
                }
                LOG_INFO("MemOrch", "  Reasoner loaded successfully");
                return {};
            }

            case Core::ModelRole::Embedding: {
                LOG_INFO("MemOrch", "RequestModel(Embedding)");
                if (!m_embeddingProvider) {
                    return std::unexpected("No embedding provider configured");
                }
                if (m_embeddingProvider->IsLoaded()) return {};

                auto result = m_embeddingProvider->LoadModel(m_embeddingModelPath);
                if (!result) return std::unexpected(result.error());
                return {};
            }
        }

        return std::unexpected("Unknown model role");
    }

    void MemoryOrchestrator::ReleaseModel(Core::ModelRole role) {
        std::lock_guard lock(m_mutex);

        switch (role) {
            case Core::ModelRole::Worker:
                m_workerProvider->UnloadModel();
                break;
            case Core::ModelRole::Reasoner:
                m_reasonerProvider->UnloadModel();
                break;
            case Core::ModelRole::Embedding:
                if (m_embeddingProvider) m_embeddingProvider->Unload();
                break;
        }
    }

    Core::MemorySnapshot MemoryOrchestrator::GetSnapshot() const {
        std::lock_guard lock(m_mutex);

        Core::MemorySnapshot snap;
        snap.totalPhysicalMB = GetTotalSystemRAM();
        snap.availableMB     = GetAvailableSystemRAM();

        // Estimate loaded model sizes
        snap.workerModelMB = m_workerProvider->IsModelLoaded()
                            ? m_workerConfig.estimatedSizeMB : 0;
        snap.reasonerModelMB = m_reasonerProvider->IsModelLoaded()
                              ? m_reasonerConfig.estimatedSizeMB : 0;
        snap.embeddingModelMB = (m_embeddingProvider && m_embeddingProvider->IsLoaded())
                               ? 50 : 0;  // BGE-micro-v2 is ~50 MB

        return snap;
    }

    bool MemoryOrchestrator::CanLoad(Core::ModelRole role) const {
        // Note: caller should already hold m_mutex
        size_t available = GetAvailableSystemRAM();
        size_t needed = 0;

        switch (role) {
            case Core::ModelRole::Worker:
                needed = m_workerConfig.estimatedSizeMB;
                break;
            case Core::ModelRole::Reasoner:
                needed = m_reasonerConfig.estimatedSizeMB;
                break;
            case Core::ModelRole::Embedding:
                needed = 100;  // ~50 MB model + overhead
                break;
        }

        // Require at least 1 GB headroom after loading
        return available >= (needed + 1024);
    }

    void MemoryOrchestrator::SetMemoryBudget(size_t maxModelMemoryMB) {
        std::lock_guard lock(m_mutex);
        m_memoryBudgetMB = maxModelMemoryMB;
    }

    Core::GPUInfo MemoryOrchestrator::DetectGPU() const {
        return GPUDetector::Detect();
    }

    size_t MemoryOrchestrator::GetAvailableSystemRAM() {
#ifdef _WIN32
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(memInfo);
        if (GlobalMemoryStatusEx(&memInfo)) {
            return static_cast<size_t>(memInfo.ullAvailPhys / (1024 * 1024));
        }
#endif
        return 8192;  // Fallback: assume 8 GB available
    }

    size_t MemoryOrchestrator::GetTotalSystemRAM() {
#ifdef _WIN32
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(memInfo);
        if (GlobalMemoryStatusEx(&memInfo)) {
            return static_cast<size_t>(memInfo.ullTotalPhys / (1024 * 1024));
        }
#endif
        return 16384;  // Fallback: assume 16 GB
    }

} // namespace LocalNotebookLLM::Infrastructure
