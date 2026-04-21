#pragma once
#include "Core/Models/ModelStatus.h"
#include <string>
#include <filesystem>
#include <functional>
#include <expected>
#include <cstddef>

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
        enum class Code { ModelNotFound, LoadFailed, OutOfMemory, GenerationFailed, ContextOverflow };
        Code code;
        std::string message;
    };

    using StreamCallback = std::function<void(const std::string& token)>;

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
    };

} // namespace LocalNotebookLLM::Core
