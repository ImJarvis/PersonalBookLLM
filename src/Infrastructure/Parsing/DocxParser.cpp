#include "Infrastructure/Parsing/DocxParser.h"
#include <pugixml.hpp>
#include <minizip/unzip.h>
#include <sstream>
#include <algorithm>
#include <vector>

namespace LocalNotebookLLM::Infrastructure {

    Core::NodeType DocxParser::MapStyleToNodeType(const std::string& styleId) {
        // OpenXML standard style IDs + common variants
        static const std::unordered_map<std::string, Core::NodeType> s_map = {
            {"Title",          Core::NodeType::Title},
            {"Heading1",       Core::NodeType::H1},
            {"Heading2",       Core::NodeType::H2},
            {"Heading3",       Core::NodeType::H3},
            {"Heading4",       Core::NodeType::H3},  // Collapse H4+ → H3
            {"Heading5",       Core::NodeType::H3},
            {"Heading6",       Core::NodeType::H3},
            {"ListParagraph",  Core::NodeType::ListItem},
            {"TOCHeading",     Core::NodeType::Title},
            {"Subtitle",       Core::NodeType::H1},
            {"Caption",        Core::NodeType::Caption},
            // Common locale variants
            {"berschrift1",    Core::NodeType::H1},  // German: Überschrift
            {"berschrift2",    Core::NodeType::H2},
            {"berschrift3",    Core::NodeType::H3},
        };

        auto it = s_map.find(styleId);
        return (it != s_map.end()) ? it->second : Core::NodeType::Paragraph;
    }

    std::expected<std::string, std::string>
    DocxParser::ExtractFromZip(const std::filesystem::path& zipPath, const std::string& entryName) {
        unzFile zf = unzOpen(zipPath.string().c_str());
        if (!zf) {
            return std::unexpected("Failed to open ZIP: " + zipPath.string());
        }

        if (unzLocateFile(zf, entryName.c_str(), 0) != UNZ_OK) {
            unzClose(zf);
            return std::unexpected("Entry not found in ZIP: " + entryName);
        }

        unz_file_info fileInfo;
        if (unzGetCurrentFileInfo(zf, &fileInfo, nullptr, 0, nullptr, 0, nullptr, 0) != UNZ_OK) {
            unzClose(zf);
            return std::unexpected("Failed to read file info: " + entryName);
        }

        if (unzOpenCurrentFile(zf) != UNZ_OK) {
            unzClose(zf);
            return std::unexpected("Failed to open entry: " + entryName);
        }

        std::string content;
        content.resize(fileInfo.uncompressed_size);
        int bytesRead = unzReadCurrentFile(zf, content.data(), static_cast<unsigned>(content.size()));
        unzCloseCurrentFile(zf);
        unzClose(zf);

        if (bytesRead < 0) {
            return std::unexpected("Failed to read entry data: " + entryName);
        }
        content.resize(static_cast<size_t>(bytesRead));
        return content;
    }

    std::unordered_map<std::string, Core::NodeType>
    DocxParser::ParseStyleDefinitions(const std::string& stylesXml) const {
        std::unordered_map<std::string, Core::NodeType> styleMap;

        pugi::xml_document doc;
        if (!doc.load_string(stylesXml.c_str())) {
            return styleMap;  // Return empty — will fall back to standard mapping
        }

        // Walk <w:style> elements
        for (auto style : doc.select_nodes("//w:style[@w:type='paragraph']")) {
            auto node = style.node();
            std::string styleId = node.attribute("w:styleId").as_string();
            std::string styleName = node.child("w:name").attribute("w:val").as_string();

            if (styleId.empty()) continue;

            // First check by styleId
            Core::NodeType type = MapStyleToNodeType(styleId);

            // Then check by display name (more reliable for localized documents)
            if (type == Core::NodeType::Paragraph && !styleName.empty()) {
                std::string nameLower = styleName;
                std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(),
                               [](unsigned char c) { return std::tolower(c); });

                if (nameLower.find("heading 1") != std::string::npos ||
                    nameLower.find("heading1") != std::string::npos)
                    type = Core::NodeType::H1;
                else if (nameLower.find("heading 2") != std::string::npos)
                    type = Core::NodeType::H2;
                else if (nameLower.find("heading 3") != std::string::npos ||
                         nameLower.find("heading 4") != std::string::npos)
                    type = Core::NodeType::H3;
                else if (nameLower.find("title") != std::string::npos)
                    type = Core::NodeType::Title;
            }

            styleMap[styleId] = type;
        }

        return styleMap;
    }

    Core::DocumentTree DocxParser::ParseDocumentXml(
        const std::string& xml,
        const std::unordered_map<std::string, Core::NodeType>& styleMap) const
    {
        pugi::xml_document doc;
        if (!doc.load_string(xml.c_str())) {
            return Core::DocumentTree{};
        }

        Core::DocumentNode root(Core::NodeType::Document, "", 0);
        int pageEstimate = 1;
        int lineCount = 0;

        for (auto paraXPath : doc.select_nodes("//w:p")) {
            auto pNode = paraXPath.node();

            // Extract style ID
            std::string styleId = "Normal";
            auto pStyle = pNode.child("w:pPr").child("w:pStyle");
            if (pStyle) {
                styleId = pStyle.attribute("w:val").as_string("Normal");
            }

            // Concatenate all <w:t> text runs
            std::string text;
            bool hasBold = false;
            for (auto run : pNode.children("w:r")) {
                // Check run properties for bold
                auto rPr = run.child("w:rPr");
                if (rPr.child("w:b") || rPr.child("w:bCs")) {
                    hasBold = true;
                }
                for (auto t : run.children("w:t")) {
                    text += t.child_value();
                }

                // Check for page breaks
                auto br = run.child("w:br");
                if (br && std::string(br.attribute("w:type").as_string()) == "page") {
                    pageEstimate++;
                    lineCount = 0;
                }
            }

            // Also check paragraph-level page breaks
            auto sectPr = pNode.child("w:pPr").child("w:sectPr");
            if (sectPr) {
                pageEstimate++;
                lineCount = 0;
            }

            if (text.empty()) continue;

            // Map style to node type
            Core::NodeType type;
            auto it = styleMap.find(styleId);
            if (it != styleMap.end()) {
                type = it->second;
            } else {
                type = MapStyleToNodeType(styleId);
            }

            // Page estimation: ~45 lines per page
            lineCount++;
            if (lineCount > 45) { pageEstimate++; lineCount = 0; }

            Core::DocumentNode node(type, std::move(text), pageEstimate,
                                    0.0f, hasBold, 1.0f);  // 100% confidence for DOCX styles
            root.AddChild(std::move(node));
        }

        if (root.GetChildren().empty()) {
            return Core::DocumentTree{};
        }

        return Core::DocumentTree(std::move(root));
    }

    std::expected<Core::DocumentTree, Core::ParseError>
    DocxParser::ParseFile(const std::filesystem::path& filePath) const {
        if (!std::filesystem::exists(filePath)) {
            return std::unexpected(Core::ParseError{
                Core::ParseError::Code::FileNotFound,
                "File not found: " + filePath.string()
            });
        }

        if (filePath.extension() != ".docx") {
            return std::unexpected(Core::ParseError{
                Core::ParseError::Code::UnsupportedFormat,
                "Not a DOCX file: " + filePath.string()
            });
        }

        // Extract styles.xml (optional — some DOCX files omit it)
        std::unordered_map<std::string, Core::NodeType> styleMap;
        auto stylesResult = ExtractFromZip(filePath, "word/styles.xml");
        if (stylesResult) {
            styleMap = ParseStyleDefinitions(*stylesResult);
        }

        // Extract and parse document.xml (required)
        auto docResult = ExtractFromZip(filePath, "word/document.xml");
        if (!docResult) {
            return std::unexpected(Core::ParseError{
                Core::ParseError::Code::CorruptFile,
                "Failed to extract document.xml: " + docResult.error()
            });
        }

        auto tree = ParseDocumentXml(*docResult, styleMap);
        if (tree.IsEmpty()) {
            return std::unexpected(Core::ParseError{
                Core::ParseError::Code::EncodingError,
                "No text content extracted from DOCX"
            });
        }

        return tree;
    }

} // namespace LocalNotebookLLM::Infrastructure
