#include "Application/ContextAssembler.h"
#include "Core/Enums/NodeType.h"
#include <algorithm>
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
        // Format: [Source: filename, Page N] — enables multi-document citation tracking
        std::string sourceName = result.documentName.empty() ? "document" : result.documentName;
        oss << "[Source: " << sourceName << ", Page " << result.pageNumber << "]";
        if (!result.sectionPath.empty() && result.sectionPath != "Document Root") {
            oss << " — " << result.sectionPath;
        }
        oss << "\n" << result.content << "\n\n";
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

        // ── Sort into reading order so the LLM sees a coherent document narrative ──
        // Score-ranked order fragments the document; reading order lets the model
        // follow the author's argument from beginning to end.
        std::vector<Core::SearchResult> ordered(results);
        std::sort(ordered.begin(), ordered.end(),
            [](const Core::SearchResult& a, const Core::SearchResult& b) {
                if (a.documentId != b.documentId) return a.documentId < b.documentId;
                return a.pageNumber < b.pageNumber;
            });

        std::ostringstream contextStream;
        contextStream
            << "=== DOCUMENT CONTENT ===\n"
            << "The following passages are extracted from the document in reading order.\n"
            << "Each passage is tagged with [Source: filename, Page N] for citation.\n"
            << "Read them carefully and use them to answer the question.\n\n";

        usedTokens += m_estimator(contextStream.str());

        for (const auto& result : ordered) {
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
