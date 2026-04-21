#include "Application/ContextAssembler.h"
#include "Core/Enums/NodeType.h"
#include <sstream>

namespace LocalNotebookLLM::Application {

    ContextAssembler::ContextAssembler(TokenEstimator estimator)
        : m_estimator(estimator ? std::move(estimator) : DefaultEstimate) {}

    size_t ContextAssembler::DefaultEstimate(const std::string& text) {
        // ~4 chars per token is a reasonable estimate for English text
        return (text.size() + 3) / 4;
    }

    std::string ContextAssembler::FormatSourceBlock(const Core::SearchResult& result) {
        std::ostringstream oss;
        oss << "[SOURCE: Page " << result.pageNumber << "]\n"
            << result.content << "\n\n";
        return oss.str();
    }

    Core::AssembledContext ContextAssembler::Assemble(
        const std::vector<Core::SearchResult>& results,
        const std::string& query,
        size_t maxTokens)
    {
        Core::AssembledContext ctx;

        // Reserve 20% of budget for the system prompt + user question
        size_t contextBudget = static_cast<size_t>(maxTokens * 0.80);
        size_t usedTokens = 0;

        std::ostringstream contextStream;
        contextStream << "=== DOCUMENT CONTEXT ===\n"
                      << "You are an analytical retrieval assistant. Your sole purpose is to retrieve facts.\n"
                      << "1. Use ONLY the following sources to answer the question.\n"
                      << "2. If the sources below DO NOT contain the exact answer, you MUST reply: \"The provided document does not contain this information.\"\n"
                      << "3. DO NOT use your own knowledge. DO NOT hallucinate.\n"
                      << "4. Cite sources meticulously using exact [Page X] notation.\n\n";

        usedTokens += m_estimator(contextStream.str());

        for (const auto& result : results) {
            std::string block = FormatSourceBlock(result);
            size_t blockTokens = m_estimator(block);

            if (usedTokens + blockTokens > contextBudget) {
                ctx.sectionsOmitted++;
                continue;
            }

            contextStream << block;
            usedTokens += blockTokens;
            ctx.sectionsIncluded++;

            // Track source for post-generation citation validation
            Core::Citation source;
            source.pageNumber  = result.pageNumber;
            source.sectionPath = result.sectionPath;
            source.citedText   = result.content.substr(0, 200); // First 200 chars for matching
            ctx.sourceMap.push_back(std::move(source));
        }

        ctx.formattedContext = contextStream.str();
        ctx.totalTokens      = static_cast<int>(usedTokens);

        return ctx;
    }

} // namespace LocalNotebookLLM::Application
