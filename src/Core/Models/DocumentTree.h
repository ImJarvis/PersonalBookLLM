#pragma once
#include "Core/Models/DocumentNode.h"
#include <vector>
#include <functional>

namespace LocalNotebookLLM::Core {

    /// A hierarchical tree of DocumentNodes representing a parsed document.
    /// The root node is either a Title node or a synthetic Document node.
    class DocumentTree {
    public:
        DocumentTree() = default;

        explicit DocumentTree(DocumentNode root)
            : m_root(std::move(root)) {}

        // ─── Accessors ───
        [[nodiscard]] const DocumentNode& GetRoot() const noexcept { return m_root; }
        [[nodiscard]] DocumentNode&       GetRoot() noexcept { return m_root; }
        [[nodiscard]] bool                IsEmpty() const noexcept {
            return m_root.GetType() == NodeType::Unknown && m_root.GetChildren().empty();
        }

        // ─── Tree Operations ───

        /// Add a node to the tree. The tree builder decides placement based on heading hierarchy.
        void AddNode(DocumentNode node) {
            m_root.AddChild(std::move(node));
        }

        /// Flatten the tree into a depth-first ordered list of all nodes.
        [[nodiscard]] std::vector<DocumentNode> Flatten() const {
            std::vector<DocumentNode> result;
            FlattenRecursive(m_root, result);
            return result;
        }

        /// Count total nodes in the tree.
        [[nodiscard]] size_t NodeCount() const noexcept {
            return CountRecursive(m_root);
        }

        /// Get total page count (max page number across all nodes).
        [[nodiscard]] int PageCount() const noexcept {
            int maxPage = 0;
            WalkRecursive(m_root, [&maxPage](const DocumentNode& n) {
                if (n.GetPageNumber() > maxPage) maxPage = n.GetPageNumber();
            });
            return maxPage;
        }

    private:
        DocumentNode m_root{NodeType::Document, "", 0};

        static void FlattenRecursive(const DocumentNode& node,
                                      std::vector<DocumentNode>& out) {
            out.push_back(node);
            for (const auto& child : node.GetChildren()) {
                FlattenRecursive(child, out);
            }
        }

        static size_t CountRecursive(const DocumentNode& node) noexcept {
            size_t count = 1;
            for (const auto& child : node.GetChildren()) {
                count += CountRecursive(child);
            }
            return count;
        }

        static void WalkRecursive(const DocumentNode& node,
                                   const std::function<void(const DocumentNode&)>& fn) {
            fn(node);
            for (const auto& child : node.GetChildren()) {
                WalkRecursive(child, fn);
            }
        }
    };

} // namespace LocalNotebookLLM::Core
