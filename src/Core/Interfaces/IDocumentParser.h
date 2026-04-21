#pragma once
#include "Core/Models/DocumentTree.h"
#include "Core/Enums/DocumentFormat.h"
#include <filesystem>
#include <vector>
#include <string>
#include <expected>

namespace LocalNotebookLLM::Core {

    struct ParseError {
        enum class Code { FileNotFound, UnsupportedFormat, CorruptFile, EncodingError, OutOfMemory };
        Code code;
        std::string message;
    };

    /// @brief Parses document files into structural trees.
    /// Implementations must be stateless and thread-safe.
    class IDocumentParser {
    public:
        virtual ~IDocumentParser() = default;

        [[nodiscard]]
        virtual std::expected<DocumentTree, ParseError>
        ParseFile(const std::filesystem::path& filePath) const = 0;

        [[nodiscard]]
        virtual std::vector<std::string> GetSupportedFormats() const = 0;

        [[nodiscard]]
        virtual bool CanParse(const std::filesystem::path& filePath) const = 0;
    };

} // namespace LocalNotebookLLM::Core
