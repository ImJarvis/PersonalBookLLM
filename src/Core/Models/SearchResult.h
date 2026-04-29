#pragma once
#include "Core/Enums/NodeType.h"
#include <string>
#include <cstdint>
#include <vector>

namespace LocalNotebookLLM::Core {

    /// A single result from an FTS5 BM25 search.
    struct SearchResult {
        int64_t     nodeId       = 0;
        int64_t     documentId   = 0;
        NodeType    nodeType     = NodeType::Unknown;
        int         pageNumber   = 0;
        std::string sectionPath;     // e.g. "Chapter 3 > 3.2 Risk Factors"
        std::string content;
        double      bm25Score    = 0.0;
        std::string documentName;    // Source filename for citation grouping
    };

    /// Context for a section: the heading + all its children.
    struct SectionContext {
        SearchResult heading;
        std::vector<SearchResult> children;
    };

    /// Search tier indicator for hybrid search results.
    enum class SearchTier { BM25, Structural, Embedding };

    /// Extended search result with tier and confidence information.
    struct HybridSearchResult : SearchResult {
        SearchTier resolvedTier   = SearchTier::BM25;
        double     confidenceScore = 0.0;
    };

} // namespace LocalNotebookLLM::Core
