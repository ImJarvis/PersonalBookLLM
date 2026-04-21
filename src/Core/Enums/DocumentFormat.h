#pragma once
#include <string>

namespace LocalNotebookLLM::Core {

    enum class DocumentFormat {
        PDF,
        DOCX,
        TXT,
        EPUB,
        Unknown
    };

    [[nodiscard]] inline std::string DocumentFormatToString(DocumentFormat f) {
        switch (f) {
            case DocumentFormat::PDF:  return "PDF";
            case DocumentFormat::DOCX: return "DOCX";
            case DocumentFormat::TXT:  return "TXT";
            case DocumentFormat::EPUB: return "EPUB";
            default:                   return "Unknown";
        }
    }

    [[nodiscard]] inline DocumentFormat ExtensionToFormat(const std::string& ext) {
        if (ext == ".pdf")  return DocumentFormat::PDF;
        if (ext == ".docx") return DocumentFormat::DOCX;
        if (ext == ".txt")  return DocumentFormat::TXT;
        if (ext == ".epub") return DocumentFormat::EPUB;
        return DocumentFormat::Unknown;
    }

} // namespace LocalNotebookLLM::Core
