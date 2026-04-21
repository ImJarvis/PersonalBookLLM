#pragma once
#include <cstdint>
#include <string>

namespace LocalNotebookLLM::Core {

    /// Structural node types in a parsed document tree.
    /// Ordered by heading level for comparison (Title < H1 < H2 < H3 < leaf types).
    enum class NodeType : uint8_t {
        Document  = 0,   // Synthetic root node
        Title     = 1,   // Document title (largest font, first page)
        H1        = 2,   // Major heading / chapter
        H2        = 3,   // Sub-heading
        H3        = 4,   // Sub-sub-heading
        Paragraph = 10,  // Body text
        ListItem  = 11,  // Bulleted or numbered list entry
        Table     = 12,  // Table content
        Caption   = 13,  // Figure/table caption
        Footer    = 14,  // Page footer
        Unknown   = 255
    };

    /// Returns true if the node type represents a heading (Title, H1, H2, H3).
    [[nodiscard]] constexpr bool IsHeadingType(NodeType t) noexcept {
        return t == NodeType::Title || t == NodeType::H1 ||
               t == NodeType::H2   || t == NodeType::H3;
    }

    /// Returns the heading depth (0 = Title, 1 = H1, 2 = H2, 3 = H3, -1 = not a heading).
    [[nodiscard]] constexpr int HeadingDepth(NodeType t) noexcept {
        switch (t) {
            case NodeType::Title: return 0;
            case NodeType::H1:    return 1;
            case NodeType::H2:    return 2;
            case NodeType::H3:    return 3;
            default:              return -1;
        }
    }

    [[nodiscard]] inline std::string NodeTypeToString(NodeType t) {
        switch (t) {
            case NodeType::Document:  return "document";
            case NodeType::Title:     return "title";
            case NodeType::H1:        return "h1";
            case NodeType::H2:        return "h2";
            case NodeType::H3:        return "h3";
            case NodeType::Paragraph: return "paragraph";
            case NodeType::ListItem:  return "list_item";
            case NodeType::Table:     return "table";
            case NodeType::Caption:   return "caption";
            case NodeType::Footer:    return "footer";
            default:                  return "unknown";
        }
    }

    [[nodiscard]] inline NodeType StringToNodeType(const std::string& s) {
        if (s == "document")  return NodeType::Document;
        if (s == "title")     return NodeType::Title;
        if (s == "h1")        return NodeType::H1;
        if (s == "h2")        return NodeType::H2;
        if (s == "h3")        return NodeType::H3;
        if (s == "paragraph") return NodeType::Paragraph;
        if (s == "list_item") return NodeType::ListItem;
        if (s == "table")     return NodeType::Table;
        if (s == "caption")   return NodeType::Caption;
        if (s == "footer")    return NodeType::Footer;
        return NodeType::Unknown;
    }

} // namespace LocalNotebookLLM::Core
