#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace LocalNotebookLLM::Core {

    /// A single citation reference within an LLM answer.
    struct Citation {
        int         pageNumber  = 0;
        std::string sectionPath;
        std::string citedText;      // The specific text that was cited
        std::string documentName;   // Source filename for multi-doc citation
        bool        isVerified  = false; // Post-generation validation result
    };

    /// Complete answer from the inference pipeline, with citations.
    struct CitedAnswer {
        std::string           answerText;
        std::vector<Citation> citations;
        int                   tokensGenerated = 0;
        int                   tokensPrompt    = 0;
        double                generationTimeMs = 0.0;
        int                   sectionsSearched = 0;
        int                   sectionsOmitted  = 0;   // Due to context window limits
    };

    /// Assembled context ready for LLM prompt injection.
    struct AssembledContext {
        std::string formattedContext;  // Tagged with [SOURCE: Page X, Section "Y"]
        int         totalTokens = 0;
        int         sectionsIncluded = 0;
        int         sectionsOmitted = 0;
        std::vector<Citation> sourceMap;  // For post-generation validation
    };

} // namespace LocalNotebookLLM::Core
