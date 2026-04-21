#pragma once
#include "Core/Enums/DocumentFormat.h"
#include <string>
#include <cstdint>

namespace LocalNotebookLLM::Core {

    /// Metadata about an ingested document stored in the repository.
    struct DocumentMetadata {
        int64_t        id          = 0;
        std::string    filename;
        std::string    filepath;
        DocumentFormat format      = DocumentFormat::Unknown;
        int            pageCount   = 0;
        std::string    ingestedAt; // ISO 8601
        std::string    fileHash;   // SHA-256
        size_t         nodeCount   = 0; // Total indexed structural nodes
    };

} // namespace LocalNotebookLLM::Core
