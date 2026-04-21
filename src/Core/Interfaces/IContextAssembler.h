#pragma once
#include "Core/Models/SearchResult.h"
#include "Core/Models/CitedAnswer.h"
#include <vector>
#include <string>

namespace LocalNotebookLLM::Core {

    /// @brief Assembles search results into a formatted context for LLM prompts.
    /// Enforces token budget and tags each block with source citations.
    class IContextAssembler {
    public:
        virtual ~IContextAssembler() = default;

        [[nodiscard]]
        virtual AssembledContext
        Assemble(const std::vector<SearchResult>& results,
                 const std::string& query,
                 size_t maxTokens) = 0;
    };

} // namespace LocalNotebookLLM::Core
