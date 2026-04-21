#pragma once
#include "Core/Interfaces/IEmbeddingProvider.h"
#include <memory>
#include <mutex>

namespace LocalNotebookLLM::Infrastructure {

    /// @brief Provides embeddings using llama.cpp.
    class LlamaCppEmbeddingProvider : public Core::IEmbeddingProvider {
    public:
        LlamaCppEmbeddingProvider();
        ~LlamaCppEmbeddingProvider() override;

        [[nodiscard]]
        std::expected<void, std::string>
        LoadModel(const std::filesystem::path& modelPath) override;

        [[nodiscard]]
        std::expected<std::vector<float>, std::string>
        Embed(const std::string& text) override;

        [[nodiscard]]
        std::expected<std::vector<std::vector<float>>, std::string>
        EmbedBatch(const std::vector<std::string>& texts) override;

        [[nodiscard]] bool IsLoaded() const override;
        [[nodiscard]] size_t GetDimensions() const override;
        void Unload() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
        mutable std::mutex m_mutex;
    };

} // namespace LocalNotebookLLM::Infrastructure
