#include "Infrastructure/Indexing/SqliteFts5Indexer.h"
#include "Core/Enums/NodeType.h"
#include "Core/Log.h"
#include <sstream>

namespace LocalNotebookLLM::Infrastructure {

    SqliteFts5Indexer::SqliteFts5Indexer(std::shared_ptr<DatabaseManager> db, std::shared_ptr<Core::IEmbeddingProvider> embedder)
        : m_db(std::move(db)), m_embedder(std::move(embedder)) {}

    std::expected<void, Core::IndexError>
    SqliteFts5Indexer::IndexDocument(int64_t documentId, const Core::DocumentTree& tree) {
        try {
            SqliteTransaction txn(m_db->GetHandle());

            // Remove previous index for re-index support
            RemoveDocument(documentId);

            std::string sectionPath;
            int sortOrder = 0;

            // ─── PERF FIX: Prepare the INSERT once, reuse it for every node ───
            // Previously this was called inside WalkAndInsert (recursive), which
            // meant sqlite3_prepare_v2 was called once per node — O(N) compilations.
            // Now we compile once and Reset()+Rebind() per node — O(1) compilations.
            auto insertStmt = m_db->Prepare(R"(
                INSERT INTO document_nodes
                    (document_id, parent_id, node_type, page_number, section_path,
                     content, font_size, is_bold, bbox_x, bbox_y, bbox_w, bbox_h,
                     confidence, sort_order, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            )");

            const auto& root = tree.GetRoot();

            // If root is a Document node, walk its children directly
            if (root.GetType() == Core::NodeType::Document) {
                for (const auto& child : root.GetChildren()) {
                    WalkAndInsert(documentId, child, std::nullopt, sectionPath, sortOrder, insertStmt);
                }
            } else {
                WalkAndInsert(documentId, root, std::nullopt, sectionPath, sortOrder, insertStmt);
            }

            // Update node count in documents table
            auto updateStmt = m_db->Prepare(
                "UPDATE documents SET node_count = ? WHERE id = ?");
            updateStmt.Bind(1, sortOrder);
            updateStmt.Bind(2, documentId);
            updateStmt.Execute();

            txn.Commit();
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(Core::IndexError{
                Core::IndexError::Code::DatabaseError,
                std::string("Indexing failed: ") + e.what()
            });
        }
    }

    void SqliteFts5Indexer::WalkAndInsert(
        int64_t docId, const Core::DocumentNode& node,
        std::optional<int64_t> parentDbId,
        std::string& sectionPath, int& sortOrder,
        SqliteStatement& stmt)  // pre-prepared INSERT, passed by reference — not re-compiled per node
    {
        // Skip empty content nodes
        if (node.GetText().empty() && !node.IsHeading()) {
            for (const auto& child : node.GetChildren()) {
                WalkAndInsert(docId, child, parentDbId, sectionPath, sortOrder, stmt);
            }
            return;
        }

        // Build section path for headings
        std::string previousPath = sectionPath;
        if (node.IsHeading()) {
            if (!sectionPath.empty()) sectionPath += " > ";
            // Truncate long headings for path readability
            std::string headingText = node.GetText().substr(0, 80);
            sectionPath += headingText;
        }

        std::string currentPath = sectionPath.empty() ? "Document Root" : sectionPath;

        // Reset the shared prepared statement and bind fresh values for this node.
        // This is the key perf fix: Reset() is O(1), vs Prepare() which is O(SQL length).
        stmt.Reset();
        stmt.Bind(1, docId);
        stmt.Bind(2, parentDbId);
        stmt.Bind(3, Core::NodeTypeToString(node.GetType()));
        stmt.Bind(4, node.GetPageNumber());
        stmt.Bind(5, currentPath);
        stmt.Bind(6, node.GetText());
        stmt.Bind(7, std::optional<double>(node.GetFontSize()));
        stmt.Bind(8, node.IsBold() ? 1 : 0);

        auto bbox = node.GetBoundingBox();
        stmt.Bind(9,  std::optional<double>(bbox.x));
        stmt.Bind(10, std::optional<double>(bbox.y));
        stmt.Bind(11, std::optional<double>(bbox.w));
        stmt.Bind(12, std::optional<double>(bbox.h));
        stmt.Bind(13, std::optional<double>(node.GetConfidence()));
        stmt.Bind(14, sortOrder++);
        stmt.BindNull(15);
        stmt.Execute();

        int64_t nodeDbId = m_db->LastInsertRowId();

        // Recurse into children
        for (const auto& child : node.GetChildren()) {
            WalkAndInsert(docId, child, nodeDbId, sectionPath, sortOrder, stmt);
        }

        // Restore section path when leaving a heading scope
        if (node.IsHeading()) {
            sectionPath = previousPath;
        }
    }

    void SqliteFts5Indexer::RemoveDocument(int64_t documentId) {
        // CASCADE deletes document_nodes → triggers clean FTS5
        auto stmt = m_db->Prepare("DELETE FROM document_nodes WHERE document_id = ?");
        stmt.Bind(1, documentId);
        stmt.Execute();
    }

    void SqliteFts5Indexer::GenerateEmbeddingsBackground(int64_t documentId, std::function<void(int, int)> progressCb) {
        if (!m_embedder || !m_embedder->IsLoaded()) return;

        // Fetch all non-empty nodes for this document missing embeddings
        auto queryStmt = m_db->Prepare("SELECT id, content FROM document_nodes WHERE document_id = ? AND embedding IS NULL AND content != ''");
        queryStmt.Bind(1, documentId);

        struct NodeCandidate { int64_t id; std::string text; };
        std::vector<NodeCandidate> candidates;
        while (queryStmt.Step()) {
            candidates.push_back({ queryStmt.GetInt64(0), queryStmt.GetString(1) });
        }

        if (candidates.empty()) return;

        // Group short texts (Semantic Chunk Aggregation) or batch them
        const size_t BATCH_SIZE = 8; 
        int totalProcessed = 0;
        int totalCount = static_cast<int>(candidates.size());

        auto updateStmt = m_db->Prepare("UPDATE document_nodes SET embedding = ? WHERE id = ?");

        for (size_t i = 0; i < candidates.size(); i += BATCH_SIZE) {
            size_t end = std::min(i + BATCH_SIZE, candidates.size());
            std::vector<std::string> texts;
            for (size_t j = i; j < end; ++j) {
                texts.push_back(candidates[j].text);
            }

            auto embResults = m_embedder->EmbedBatch(texts);
            if (embResults) {
                SqliteTransaction tx(m_db->GetHandle());
                for (size_t j = 0; j < embResults->size(); ++j) {
                    const auto& vec = (*embResults)[j];
                    updateStmt.Reset();
                    updateStmt.BindBlob(1, vec.data(), vec.size() * sizeof(float));
                    updateStmt.Bind(2, candidates[i + j].id);
                    updateStmt.Execute();
                }
                tx.Commit();
            } else {
                LOG_ERROR("SqliteFts5Indexer", "EmbedBatch FAILED during background processing: " + embResults.error());
            }

            totalProcessed += static_cast<int>(texts.size());
            if (progressCb) progressCb(totalProcessed, totalCount);
        }
    }

    std::vector<Core::SearchResult>
    SqliteFts5Indexer::Search(const std::string& query, const Core::SearchOptions& options) const {
        if (query == "*") {
            // General query (e.g. for generic summarizing/asking about document)
            std::string sql = R"(
                SELECT
                    id, document_id, node_type, page_number, section_path, content, -1.0 as rank
                FROM document_nodes
                WHERE node_type IN ('title', 'h1', 'paragraph')
            )";
            if (options.documentIdFilter >= 0) {
                sql += " AND document_id = " + std::to_string(options.documentIdFilter);
            }
            sql += " ORDER BY sort_order LIMIT ?";

            auto stmt = m_db->Prepare(sql);
            stmt.Bind(1, options.topK);

            std::vector<Core::SearchResult> results;
            while (stmt.Step()) {
                results.push_back(RowToSearchResult(stmt));
            }
            return results;
        }

        std::string ftsQuery = EscapeFts5Query(query);
        if (ftsQuery.empty()) return {};

        std::string sql = R"(
            SELECT
                dn.id,
                dn.document_id,
                dn.node_type,
                dn.page_number,
                dn.section_path,
                dn.content,
                rank
            FROM fts_index
            JOIN document_nodes dn ON dn.id = fts_index.rowid
            WHERE fts_index MATCH ?
        )";

        if (options.documentIdFilter >= 0) {
            sql += " AND dn.document_id = " + std::to_string(options.documentIdFilter);
        }

        sql += " ORDER BY rank LIMIT ?";

        auto stmt = m_db->Prepare(sql);
        stmt.Bind(1, ftsQuery);
        stmt.Bind(2, options.topK);

        std::vector<Core::SearchResult> results;
        while (stmt.Step()) {
            results.push_back(RowToSearchResult(stmt));
        }

        // Filter by minimum score if requested
        if (options.minScore > 0.0) {
            std::erase_if(results, [&](const Core::SearchResult& r) {
                return r.bm25Score < options.minScore;
            });
        }

        return results;
    }

    Core::SectionContext SqliteFts5Indexer::GetSectionContext(int64_t nodeId) const {
        Core::SectionContext ctx;

        // Get the node itself
        auto nodeStmt = m_db->Prepare(R"(
            SELECT id, document_id, node_type, page_number, section_path, content, 0.0
            FROM document_nodes WHERE id = ?
        )");
        nodeStmt.Bind(1, nodeId);
        if (nodeStmt.Step()) {
            ctx.heading = RowToSearchResult(nodeStmt);
        }

        // Find parent heading (walk up to nearest heading ancestor)
        auto parentStmt = m_db->Prepare(R"(
            WITH RECURSIVE ancestors(id, parent_id, node_type) AS (
                SELECT id, parent_id, node_type FROM document_nodes WHERE id = ?
                UNION ALL
                SELECT dn.id, dn.parent_id, dn.node_type
                FROM document_nodes dn JOIN ancestors a ON dn.id = a.parent_id
            )
            SELECT dn.id, dn.document_id, dn.node_type, dn.page_number,
                   dn.section_path, dn.content, 0.0
            FROM ancestors a
            JOIN document_nodes dn ON dn.id = a.id
            WHERE dn.node_type IN ('title','h1','h2','h3')
            AND dn.id != ?
            LIMIT 1
        )");
        parentStmt.Bind(1, nodeId);
        parentStmt.Bind(2, nodeId);
        if (parentStmt.Step()) {
            ctx.heading = RowToSearchResult(parentStmt);
        }

        // Get sibling nodes under the same parent
        auto siblingsStmt = m_db->Prepare(R"(
            SELECT id, document_id, node_type, page_number, section_path, content, 0.0
            FROM document_nodes
            WHERE parent_id = (SELECT parent_id FROM document_nodes WHERE id = ?)
            ORDER BY sort_order
        )");
        siblingsStmt.Bind(1, nodeId);
        while (siblingsStmt.Step()) {
            ctx.children.push_back(RowToSearchResult(siblingsStmt));
        }

        return ctx;
    }

    size_t SqliteFts5Indexer::GetIndexedNodeCount() const {
        auto stmt = m_db->Prepare("SELECT COUNT(*) FROM document_nodes");
        stmt.Step();
        return static_cast<size_t>(stmt.GetInt64(0));
    }

    std::string SqliteFts5Indexer::EscapeFts5Query(const std::string& query) {
        // Wrap each token in double-quotes to prevent FTS5 syntax errors
        // from special characters like hyphens and parentheses.
        std::istringstream stream(query);
        std::string word;
        std::vector<std::string> terms;

        while (stream >> word) {
            // Lowercase FTS5 boolean operators so they become plain search terms
            if (word == "AND" || word == "OR" || word == "NOT" || word == "NEAR") {
                for (auto& c : word)
                    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            terms.push_back('"' + word + '"');
        }

        if (terms.empty()) return {};

        // ── Precision strategy ──
        // Multi-word queries: implicit FTS5 AND (space-separated quoted terms).
        //   "machine learning" → "machine" "learning" — both must be present.
        //   This avoids flooding results with documents matching any single token.
        // Single-word queries: return as-is (implicit AND is identical to OR for 1 term).
        std::string escaped;
        for (size_t i = 0; i < terms.size(); ++i) {
            if (i > 0) escaped += ' ';
            escaped += terms[i];
        }
        return escaped;
    }

    Core::SearchResult SqliteFts5Indexer::RowToSearchResult(SqliteStatement& stmt) {
        Core::SearchResult r;
        r.nodeId      = stmt.GetInt64(0);
        r.documentId  = stmt.GetInt64(1);
        r.nodeType    = Core::StringToNodeType(stmt.GetString(2));
        r.pageNumber  = stmt.GetInt(3);
        r.sectionPath = stmt.GetString(4);
        r.content     = stmt.GetString(5);
        r.bm25Score   = -stmt.GetDouble(6);  // Negate for intuitive scoring (FTS5 rank is negative)
        return r;
    }

} // namespace LocalNotebookLLM::Infrastructure
