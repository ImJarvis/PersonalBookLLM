#pragma once
#include "Core/Interfaces/IDocumentParser.h"
#include <unordered_map>

namespace LocalNotebookLLM::Infrastructure {

    /// @brief Native OpenXML parser for .docx files.
    /// Directly reads <w:pStyle> tags for 100% accurate heading detection.
    /// No heuristics needed — style IDs ARE the document structure.
    class DocxParser : public Core::IDocumentParser {
    public:
        [[nodiscard]]
        std::expected<Core::DocumentTree, Core::ParseError>
        ParseFile(const std::filesystem::path& filePath) const override;

        [[nodiscard]]
        std::vector<std::string> GetSupportedFormats() const override { return {".docx"}; }

        [[nodiscard]]
        bool CanParse(const std::filesystem::path& p) const override {
            return p.extension() == ".docx";
        }

    private:
        /// Extract a file from a ZIP archive into a string.
        [[nodiscard]]
        static std::expected<std::string, std::string>
        ExtractFromZip(const std::filesystem::path& zipPath, const std::string& entryName);

        /// Parse styles.xml → build style-to-NodeType map.
        [[nodiscard]]
        std::unordered_map<std::string, Core::NodeType>
        ParseStyleDefinitions(const std::string& stylesXml) const;

        /// Parse document.xml paragraphs using style map.
        [[nodiscard]]
        Core::DocumentTree
        ParseDocumentXml(const std::string& documentXml,
                         const std::unordered_map<std::string, Core::NodeType>& styleMap) const;

        /// Map common OpenXML style IDs to our NodeType enum.
        [[nodiscard]]
        static Core::NodeType MapStyleToNodeType(const std::string& styleId);
    };

} // namespace LocalNotebookLLM::Infrastructure
