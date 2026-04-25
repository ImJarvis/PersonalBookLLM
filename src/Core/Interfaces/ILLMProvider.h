#pragma once
#include "Core/Models/ModelStatus.h"
#include <string>
#include <filesystem>
#include <functional>
#include <expected>
#include <cstddef>
#include <atomic>
#include <chrono>

namespace LocalNotebookLLM::Core {

    struct ModelParams {
        int   contextWindowSize = 4096;
        int   threadCount       = 0;        // 0 = auto-detect
        float temperature       = 0.3f;     // Low temp for factual accuracy
        float topP              = 0.9f;
        int   topK              = 40;

        // ─── Device Placement ───
        enum class PreferredDevice { CPU, GPU, Auto };
        PreferredDevice device    = PreferredDevice::Auto;
        int             gpuLayers = -1;     // -1 = auto

        // ─── KV Cache Optimization ───
        enum class KVCacheType { F16, Q8_0, Q4_0 };
        KVCacheType kvCacheTypeK    = KVCacheType::Q8_0;
        KVCacheType kvCacheTypeV    = KVCacheType::Q8_0;
        bool        useFlashAttention = true;
    };

    struct GenerationResult {
        std::string text;
        int         tokensGenerated  = 0;
        int         tokensPrompt     = 0;
        double      generationTimeMs = 0.0;
        bool        wasTruncated     = false;
    };

    struct LLMError {
        enum class Code { ModelNotFound, LoadFailed, OutOfMemory, GenerationFailed, ContextOverflow, Cancelled };
        Code code;
        std::string message;
    };

    using StreamCallback = std::function<void(const std::string& token)>;

    /// @brief Cooperative control channel between the UI thread and inference thread.
    ///
    /// Three roles:
    ///   1. **Cancellation**: UI sets `cancelled` → provider exits at next checkpoint.
    ///   2. **Heartbeat**: Provider updates `heartbeatEpochMs` after every batch decode,
    ///      proving the thread is alive even during long CPU prompt evaluations.
    ///   3. **Progress**: Provider writes `evalTokensDone` / `evalTokensTotal` so the
    ///      UI can show "Evaluating context... 384/884 tokens".
    ///
    /// All fields are lock-free atomics.  Safe to read/write from any thread.
    struct CancellationToken {
        // ─── Cancellation (user-initiated only) ───
        std::atomic<bool> cancelled{false};
        void Cancel()            { cancelled.store(true,  std::memory_order_release); }
        void Reset()             { cancelled.store(false, std::memory_order_release); }
        bool IsCancelled() const { return cancelled.load(std::memory_order_acquire); }

        // ─── Heartbeat (provider writes, watchdog reads) ───
        // Milliseconds since epoch.  Updated after every successful llama_decode().
        // If this hasn't changed in 10+ minutes, the thread is truly dead.
        std::atomic<int64_t> heartbeatEpochMs{0};
        void Heartbeat() {
            auto now = std::chrono::steady_clock::now().time_since_epoch();
            heartbeatEpochMs.store(
                std::chrono::duration_cast<std::chrono::milliseconds>(now).count(),
                std::memory_order_release);
        }
        int64_t GetHeartbeatMs() const {
            return heartbeatEpochMs.load(std::memory_order_acquire);
        }

        // ─── Prompt evaluation progress (provider writes, UI reads) ───
        std::atomic<int> evalTokensDone{0};
        std::atomic<int> evalTokensTotal{0};
        void ReportEvalProgress(int done, int total) {
            evalTokensDone.store(done,  std::memory_order_release);
            evalTokensTotal.store(total, std::memory_order_release);
        }
        void ClearEvalProgress() {
            evalTokensDone.store(0,  std::memory_order_release);
            evalTokensTotal.store(0, std::memory_order_release);
        }
    };

    /// @brief Provides LLM inference capabilities.
    /// Implementations wrap specific inference engines (llama.cpp, etc.).
    class ILLMProvider {
    public:
        virtual ~ILLMProvider() = default;

        [[nodiscard]]
        virtual std::expected<void, LLMError>
        LoadModel(const std::filesystem::path& modelPath, const ModelParams& params = {}) = 0;

        [[nodiscard]]
        virtual std::expected<GenerationResult, LLMError>
        Generate(const std::string& prompt, int maxTokens = 1024) = 0;

        [[nodiscard]]
        virtual std::expected<GenerationResult, LLMError>
        GenerateStreaming(const std::string& prompt, int maxTokens, StreamCallback callback) = 0;

        [[nodiscard]] virtual size_t GetContextWindowSize() const = 0;
        [[nodiscard]] virtual size_t EstimateTokenCount(const std::string& text) const = 0;
        [[nodiscard]] virtual bool   IsModelLoaded() const = 0;
        virtual void UnloadModel() = 0;

        /// Get the cancellation token for this provider.  Callers may set it
        /// to true to request that the current (or next) Generate call aborts.
        virtual CancellationToken& GetCancellationToken() = 0;
    };

} // namespace LocalNotebookLLM::Core
