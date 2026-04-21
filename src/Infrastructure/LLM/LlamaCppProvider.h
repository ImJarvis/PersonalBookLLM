#pragma once
#include "Core/Interfaces/ILLMProvider.h"
#include "Core/Models/ModelStatus.h"
#include <memory>
#include <string>
#include <mutex>

namespace LocalNotebookLLM::Infrastructure {

    /// @brief llama.cpp-backed LLM provider.
    /// Wraps the llama.cpp C API for model loading, prompt evaluation,
    /// and token generation with KV cache quantization support.
    class LlamaCppProvider : public Core::ILLMProvider {
    public:
        LlamaCppProvider();
        ~LlamaCppProvider() override;

        LlamaCppProvider(const LlamaCppProvider&) = delete;
        LlamaCppProvider& operator=(const LlamaCppProvider&) = delete;

        [[nodiscard]]
        std::expected<void, Core::LLMError>
        LoadModel(const std::filesystem::path& modelPath,
                  const Core::ModelParams& params = {}) override;

        [[nodiscard]]
        std::expected<Core::GenerationResult, Core::LLMError>
        Generate(const std::string& prompt, int maxTokens = 1024) override;

        [[nodiscard]]
        std::expected<Core::GenerationResult, Core::LLMError>
        GenerateStreaming(const std::string& prompt, int maxTokens,
                          Core::StreamCallback callback) override;

        [[nodiscard]] size_t GetContextWindowSize() const override;
        [[nodiscard]] size_t EstimateTokenCount(const std::string& text) const override;
        [[nodiscard]] bool   IsModelLoaded() const override;
        void UnloadModel() override;

        /// Get detailed model status.
        [[nodiscard]] Core::ModelStatus GetStatus() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
        mutable std::mutex    m_mutex;  // Serialize access to llama context
    };

} // namespace LocalNotebookLLM::Infrastructure
