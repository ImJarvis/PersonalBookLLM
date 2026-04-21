#pragma once
#include "Core/Enums/NodeType.h"
#include <string>
#include <vector>
#include <cstdint>

namespace LocalNotebookLLM::Core {

    /// Bounding box in PDF coordinate space (points, origin = bottom-left).
    struct BoundingBox {
        float x = 0.0f;
        float y = 0.0f;
        float w = 0.0f;
        float h = 0.0f;
    };

    /// A single node in the structural document tree.
    /// Each node represents a logical unit: heading, paragraph, list item, table, etc.
    class DocumentNode {
    public:
        DocumentNode() = default;

        DocumentNode(NodeType type, std::string text, int pageNumber,
                     float fontSize = 0.0f, bool bold = false,
                     float confidence = 1.0f)
            : m_type(type)
            , m_text(std::move(text))
            , m_pageNumber(pageNumber)
            , m_fontSize(fontSize)
            , m_bold(bold)
            , m_confidence(confidence) {}

        // ─── Accessors ───
        [[nodiscard]] NodeType           GetType() const noexcept { return m_type; }
        [[nodiscard]] const std::string& GetText() const noexcept { return m_text; }
        [[nodiscard]] int                GetPageNumber() const noexcept { return m_pageNumber; }
        [[nodiscard]] float              GetFontSize() const noexcept { return m_fontSize; }
        [[nodiscard]] bool               IsBold() const noexcept { return m_bold; }
        [[nodiscard]] float              GetConfidence() const noexcept { return m_confidence; }
        [[nodiscard]] const BoundingBox& GetBoundingBox() const noexcept { return m_bbox; }
        [[nodiscard]] bool               IsHeading() const noexcept { return IsHeadingType(m_type); }

        [[nodiscard]] const std::vector<DocumentNode>& GetChildren() const noexcept {
            return m_children;
        }

        [[nodiscard]] std::vector<DocumentNode>& GetChildren() noexcept {
            return m_children;
        }

        // ─── Mutators ───
        void SetType(NodeType type) noexcept { m_type = type; }
        void SetText(std::string text) { m_text = std::move(text); }
        void SetPageNumber(int page) noexcept { m_pageNumber = page; }
        void SetFontSize(float size) noexcept { m_fontSize = size; }
        void SetBold(bool bold) noexcept { m_bold = bold; }
        void SetConfidence(float c) noexcept { m_confidence = c; }
        void SetBoundingBox(BoundingBox bbox) noexcept { m_bbox = bbox; }

        void AddChild(DocumentNode child) {
            m_children.push_back(std::move(child));
        }

    private:
        NodeType                  m_type       = NodeType::Unknown;
        std::string               m_text;
        int                       m_pageNumber = 0;
        float                     m_fontSize   = 0.0f;
        bool                      m_bold       = false;
        float                     m_confidence = 1.0f;
        BoundingBox               m_bbox       = {};
        std::vector<DocumentNode> m_children;
    };

} // namespace LocalNotebookLLM::Core
