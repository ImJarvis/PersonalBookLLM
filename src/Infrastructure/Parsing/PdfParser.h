#pragma once
#include "Core/Interfaces/IDocumentParser.h"
#include "Infrastructure/Parsing/StructureDetector.h"
#include <vector>

namespace LocalNotebookLLM::Infrastructure {

    /// @brief Native PDF parser using MuPDF for text+font extraction
    /// and StructureDetector for heading/paragraph classification.
    /// Extracts per-block font size, bold flags, and bounding boxes
    /// for accurate structural analysis.
    class PdfParser : public Core::IDocumentParser {
    public:
        [[nodiscard]]
        std::expected<Core::DocumentTree, Core::ParseError>
        ParseFile(const std::filesystem::path& filePath) const override;

        [[nodiscard]]
        std::vector<std::string> GetSupportedFormats() const override { return {".pdf"}; }

        [[nodiscard]]
        bool CanParse(const std::filesystem::path& p) const override {
            return p.extension() == ".pdf";
        }

    private:
        /// Extract raw text blocks with font metadata from all pages.
        [[nodiscard]]
        std::expected<std::vector<RawTextBlock>, std::string>
        ExtractTextBlocks(const std::filesystem::path& filePath) const;

        StructureDetector m_detector;
    };

} // namespace LocalNotebookLLM::Infrastructure
