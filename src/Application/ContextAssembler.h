#pragma once
#include "Core/Interfaces/IContextAssembler.h"
#include <functional>

namespace LocalNotebookLLM::Application {

    /// @brief Assembles search results into a token-budgeted, source-tagged prompt context.
    /// Each section is tagged with [SOURCE: Page X, Section "Y"] for citation extraction.
    class ContextAssembler : public Core::IContextAssembler {
    public:
        using TokenEstimator = std::function<size_t(const std::string&)>;

        /// @param estimator Function to estimate token count for a string.
        ///        Defaults to a simple chars/4 heuristic if not provided.
        explicit ContextAssembler(TokenEstimator estimator = nullptr);

        [[nodiscard]]
        Core::AssembledContext
        Assemble(const std::vector<Core::SearchResult>& results,
                 const std::string& query,
                 size_t maxTokens) override;

    private:
        TokenEstimator m_estimator;

        /// Default token estimation: ~4 characters per token (GPT-family average).
        static size_t DefaultEstimate(const std::string& text);

        /// Format a single search result as a source-tagged block.
        static std::string FormatSourceBlock(const Core::SearchResult& result);
    };

} // namespace LocalNotebookLLM::Application
