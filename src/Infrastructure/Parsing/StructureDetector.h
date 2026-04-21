#pragma once
#include "Core/Models/DocumentNode.h"
#include "Core/Models/DocumentTree.h"
#include <string>
#include <vector>

namespace LocalNotebookLLM::Infrastructure {

    /// Raw text block as extracted by a PDF/document parser (pre-classification).
    struct RawTextBlock {
        std::string text;
        float       fontSize   = 0.0f;
        bool        isBold     = false;
        Core::BoundingBox bbox = {};
        int         pageNumber = 0;
    };

    /// @brief Classifies raw text blocks into structural node types
    /// and builds a hierarchical DocumentTree from them.
    ///
    /// Uses multi-signal heuristics: font size + bold + position + content patterns.
    class StructureDetector {
    public:
        /// Classify a single raw text block into a DocumentNode with type + confidence.
        [[nodiscard]]
        Core::DocumentNode Classify(const RawTextBlock& block) const;

        /// Build a full DocumentTree from an ordered sequence of raw text blocks.
        /// Maintains heading hierarchy (Title > H1 > H2 > H3 > leaf nodes).
        [[nodiscard]]
        Core::DocumentTree BuildTree(const std::vector<RawTextBlock>& blocks) const;

    private:
        /// Font-size-based heading classification.
        [[nodiscard]]
        Core::NodeType ClassifyByFontSize(float fontSize, bool isBold) const;

        /// Content-pattern detection (bullets, numbering, table-like).
        [[nodiscard]]
        Core::NodeType DetectContentPattern(const std::string& text) const;

        /// Compute confidence score based on signal agreement.
        [[nodiscard]]
        float ComputeConfidence(Core::NodeType fontResult,
                                Core::NodeType patternResult,
                                bool isBold) const;

        /// Check if text looks like garbled/corrupt extraction.
        [[nodiscard]]
        bool IsGarbageText(const std::string& text) const;

        // ─── Font size thresholds (typical PDF defaults) ───
        static constexpr float TITLE_MIN_SIZE = 20.0f;
        static constexpr float H1_MIN_SIZE    = 16.0f;
        static constexpr float H2_MIN_SIZE    = 13.0f;
        static constexpr float H3_MIN_SIZE    = 11.5f;
        static constexpr float BODY_SIZE      = 10.0f;
    };

} // namespace LocalNotebookLLM::Infrastructure
