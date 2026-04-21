#pragma once
#include "Core/Models/DocumentMetadata.h"
#include <vector>
#include <string>
#include <cstdint>
#include <expected>

namespace LocalNotebookLLM::Core {

    /// @brief CRUD operations for document metadata persistence.
    class IDocumentRepository {
    public:
        virtual ~IDocumentRepository() = default;

        [[nodiscard]]
        virtual std::expected<int64_t, std::string>
        Add(const DocumentMetadata& doc) = 0;

        virtual void Remove(int64_t documentId) = 0;

        [[nodiscard]]
        virtual std::vector<DocumentMetadata> ListAll() const = 0;

        [[nodiscard]]
        virtual std::expected<DocumentMetadata, std::string>
        GetById(int64_t documentId) const = 0;

        [[nodiscard]]
        virtual bool ExistsByHash(const std::string& fileHash) const = 0;
    };

} // namespace LocalNotebookLLM::Core
