#pragma once
#include <vector>
#include <string>
#include <filesystem>
#include <expected>

namespace LocalNotebookLLM::Core {

    /// @brief Lightweight embedding model for semantic fallback search (Tier 3).
    /// Target: BGE-micro-v2 (~23M params, ~50 MB GGUF Q8_0).
    /// Only loaded when Tier 1+2 return low-confidence results.
    class IEmbeddingProvider {
    public:
        virtual ~IEmbeddingProvider() = default;

        [[nodiscard]]
        virtual std::expected<void, std::string>
        LoadModel(const std::filesystem::path& modelPath) = 0;

        [[nodiscard]]
        virtual std::expected<std::vector<float>, std::string>
        Embed(const std::string& text, bool isQuery = false) = 0;

        [[nodiscard]]
        virtual std::expected<std::vector<std::vector<float>>, std::string>
        EmbedBatch(const std::vector<std::string>& texts) = 0;

        [[nodiscard]] virtual bool   IsLoaded() const = 0;
        [[nodiscard]] virtual size_t GetDimensions() const = 0;
        virtual void Unload() = 0;
    };

} // namespace LocalNotebookLLM::Core
