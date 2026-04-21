#pragma once
#include "Core/Interfaces/IDocumentParser.h"
#include <vector>
#include <memory>

namespace LocalNotebookLLM::Infrastructure {

    /// @brief Routes files to the appropriate parser by extension.
    /// Implements the Composite pattern over IDocumentParser.
    class CompositeParser : public Core::IDocumentParser {
    public:
        /// Register a format-specific parser.
        void RegisterParser(std::unique_ptr<Core::IDocumentParser> parser) {
            m_parsers.push_back(std::move(parser));
        }

        [[nodiscard]]
        std::expected<Core::DocumentTree, Core::ParseError>
        ParseFile(const std::filesystem::path& path) const override {
            for (const auto& p : m_parsers) {
                if (p->CanParse(path)) return p->ParseFile(path);
            }
            return std::unexpected(Core::ParseError{
                Core::ParseError::Code::UnsupportedFormat,
                "No parser registered for: " + path.extension().string()
            });
        }

        [[nodiscard]]
        std::vector<std::string> GetSupportedFormats() const override {
            std::vector<std::string> all;
            for (const auto& p : m_parsers)
                for (const auto& f : p->GetSupportedFormats()) all.push_back(f);
            return all;
        }

        [[nodiscard]]
        bool CanParse(const std::filesystem::path& p) const override {
            for (const auto& parser : m_parsers)
                if (parser->CanParse(p)) return true;
            return false;
        }

    private:
        std::vector<std::unique_ptr<Core::IDocumentParser>> m_parsers;
    };

} // namespace LocalNotebookLLM::Infrastructure
