#include "Infrastructure/Parsing/StructureDetector.h"
#include <algorithm>
#include <regex>
#include <numeric>
#include <stack>

namespace LocalNotebookLLM::Infrastructure {

    Core::DocumentNode StructureDetector::Classify(const RawTextBlock& block) const {
        Core::NodeType fontType     = ClassifyByFontSize(block.fontSize, block.isBold);
        Core::NodeType patternType  = DetectContentPattern(block.text);
        float confidence            = ComputeConfidence(fontType, patternType, block.isBold);

        // Check for garbage text
        if (IsGarbageText(block.text)) {
            confidence = std::min(confidence, 0.3f);
        }

        // Determine final type: font-based classification takes priority for headings,
        // pattern-based for list items and special content
        Core::NodeType finalType;
        if (Core::IsHeadingType(fontType)) {
            finalType = fontType;
        } else if (patternType != Core::NodeType::Paragraph) {
            finalType = patternType;
        } else {
            finalType = fontType;
        }

        Core::DocumentNode node(finalType, block.text, block.pageNumber,
                                block.fontSize, block.isBold, confidence);
        node.SetBoundingBox(block.bbox);
        return node;
    }

    Core::DocumentTree StructureDetector::BuildTree(const std::vector<RawTextBlock>& blocks) const {
        if (blocks.empty()) {
            return Core::DocumentTree{};  // Empty tree
        }

        // Classify all blocks
        std::vector<Core::DocumentNode> nodes;
        nodes.reserve(blocks.size());
        for (const auto& block : blocks) {
            auto node = Classify(block);
            if (!node.GetText().empty()) {
                nodes.push_back(std::move(node));
            }
        }

        if (nodes.empty()) {
            return Core::DocumentTree{};
        }

        // Check if the first node is a title/heading — use it as root
        // Otherwise create a synthetic Document root
        Core::DocumentNode root;
        size_t startIdx = 0;

        if (nodes[0].GetType() == Core::NodeType::Title) {
            root = std::move(nodes[0]);
            startIdx = 1;
        } else {
            root = Core::DocumentNode(Core::NodeType::Document, "", 0);
        }

        // Build hierarchy using a stack of (node_ptr, depth) pairs.
        // The stack tracks the current nesting: Title(0) > H1(1) > H2(2) > H3(3)
        struct StackEntry {
            Core::DocumentNode* node;
            int depth;  // Heading depth: Title=0, H1=1, H2=2, H3=3, leaf=-1
        };

        std::stack<StackEntry> stack;
        int rootDepth = Core::HeadingDepth(root.GetType());
        stack.push({&root, rootDepth >= 0 ? rootDepth : -1});

        for (size_t i = startIdx; i < nodes.size(); ++i) {
            auto& current = nodes[i];
            int currentDepth = Core::HeadingDepth(current.GetType());

            if (currentDepth >= 0) {
                // It's a heading — pop stack until we find a parent at a shallower depth
                while (stack.size() > 1 && stack.top().depth >= currentDepth) {
                    stack.pop();
                }
                // Add as child of current parent
                stack.top().node->AddChild(std::move(current));
                auto& added = stack.top().node->GetChildren().back();
                stack.push({&added, currentDepth});
            } else {
                // It's a leaf node — add to current parent
                stack.top().node->AddChild(std::move(current));
            }
        }

        return Core::DocumentTree(std::move(root));
    }

    Core::NodeType StructureDetector::ClassifyByFontSize(float fontSize, bool isBold) const {
        if (fontSize >= TITLE_MIN_SIZE && isBold)       return Core::NodeType::Title;
        if (fontSize >= TITLE_MIN_SIZE)                  return Core::NodeType::H1;
        if (fontSize >= H1_MIN_SIZE && isBold)           return Core::NodeType::H1;
        if (fontSize >= H1_MIN_SIZE)                     return Core::NodeType::H2;
        if (fontSize >= H2_MIN_SIZE && isBold)           return Core::NodeType::H2;
        if (fontSize >= H2_MIN_SIZE)                     return Core::NodeType::H3;
        if (fontSize >= H3_MIN_SIZE && isBold)           return Core::NodeType::H3;
        return Core::NodeType::Paragraph;
    }

    Core::NodeType StructureDetector::DetectContentPattern(const std::string& text) const {
        if (text.empty()) return Core::NodeType::Paragraph;

        // Bullet patterns: •, -, *, ▪, ►, ●
        char first = text[0];
        if (first == '\xE2') {  // UTF-8 multibyte bullets (•, ▪, ►, etc.)
            return Core::NodeType::ListItem;
        }
        if (first == '-' || first == '*') {
            // Only if followed by a space (not a math expression)
            if (text.size() > 1 && text[1] == ' ')
                return Core::NodeType::ListItem;
        }

        // Numbered list: "1.", "2)", "a.", "a)", "(i)", etc.
        static const std::regex numberedPattern(R"(^\s*(\d+[\.\)]\s|[a-z][\.\)]\s|\([ivxlcdm]+\)\s))",
                                                 std::regex::icase);
        if (std::regex_search(text, numberedPattern))
            return Core::NodeType::ListItem;

        // Table-like content: contains tab separators or multiple | chars
        if (std::count(text.begin(), text.end(), '\t') >= 2 ||
            std::count(text.begin(), text.end(), '|') >= 2)
            return Core::NodeType::Table;

        // Footer detection: very short text at bottom of page, all caps page numbers
        static const std::regex footerPattern(R"(^\s*(Page\s+\d+|^\d+\s*$|\- \d+ \-))");
        if (text.size() < 30 && std::regex_search(text, footerPattern))
            return Core::NodeType::Footer;

        return Core::NodeType::Paragraph;
    }

    float StructureDetector::ComputeConfidence(Core::NodeType fontResult,
                                                Core::NodeType patternResult,
                                                bool isBold) const {
        // Signal agreement = higher confidence
        float confidence = 0.7f;

        if (fontResult == patternResult) {
            confidence = 1.0f;  // Strong agreement
        } else if (Core::IsHeadingType(fontResult) && isBold) {
            confidence = 0.9f;  // Font + bold agree on heading
        } else if (patternResult != Core::NodeType::Paragraph) {
            confidence = 0.8f;  // Pattern detection found something specific
        }

        return confidence;
    }

    bool StructureDetector::IsGarbageText(const std::string& text) const {
        if (text.empty()) return true;

        // Count "readable" ASCII characters vs total
        int readable = 0;
        int total = 0;
        for (unsigned char c : text) {
            total++;
            if ((c >= 32 && c <= 126) || c == '\n' || c == '\r' || c == '\t') {
                readable++;
            }
        }

        if (total == 0) return true;

        // If less than 50% readable ASCII, likely garbled
        float ratio = static_cast<float>(readable) / static_cast<float>(total);
        return ratio < 0.5f;
    }

} // namespace LocalNotebookLLM::Infrastructure
