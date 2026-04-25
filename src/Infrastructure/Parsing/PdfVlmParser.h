#pragma once
#include "Core/Interfaces/IDocumentParser.h"
#include "Infrastructure/Parsing/StructureDetector.h"
#include "Infrastructure/Parsing/PdfParser.h"
#include <filesystem>
#include <vector>
#include <string>

namespace LocalNotebookLLM::Infrastructure {

    class PdfVlmParser : public Core::IDocumentParser {
    public:
        [[nodiscard]] std::vector<std::string> GetSupportedFormats() const override {
            return { ".pdf" };
        }

        [[nodiscard]] bool CanParse(const std::filesystem::path& path) const override {
            return path.extension() == ".pdf";
        }

        [[nodiscard]]
        std::expected<Core::DocumentTree, Core::ParseError>
        ParseFile(const std::filesystem::path& filePath) const override;

    private:
        StructureDetector m_detector;
        PdfParser m_fallbackParser;
        
        std::expected<std::string, std::string> RunVlmOnImage(const std::filesystem::path& imagePath) const;
    };

} // namespace LocalNotebookLLM::Infrastructure
