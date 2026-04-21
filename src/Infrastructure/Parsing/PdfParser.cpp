#include "Infrastructure/Parsing/PdfParser.h"
#include "Core/Log.h"

#ifdef HAS_MUPDF
extern "C" {
#include <mupdf/fitz.h>
}
#endif

#include <sstream>
#include <algorithm>
#include <cmath>

namespace LocalNotebookLLM::Infrastructure {

#ifdef HAS_MUPDF

    std::expected<std::vector<RawTextBlock>, std::string>
    PdfParser::ExtractTextBlocks(const std::filesystem::path& filePath) const {
        std::vector<RawTextBlock> blocks;

        fz_context* ctx = fz_new_context(nullptr, nullptr, FZ_STORE_DEFAULT);
        if (!ctx) {
            return std::unexpected("MuPDF: Failed to create context");
        }

        fz_document* doc = nullptr;
        fz_try(ctx) {
            fz_register_document_handlers(ctx);
            doc = fz_open_document(ctx, filePath.string().c_str());
        }
        fz_catch(ctx) {
            std::string err = fz_caught_message(ctx);
            fz_drop_context(ctx);
            return std::unexpected("MuPDF: Failed to open document: " + err);
        }

        int pageCount = fz_count_pages(ctx, doc);
        LOG_INFO("PdfParser", "Opened PDF: " + std::to_string(pageCount) + " pages");

        for (int pageIdx = 0; pageIdx < pageCount; ++pageIdx) {
            fz_page* page = nullptr;
            fz_stext_page* textPage = nullptr;

            fz_try(ctx) {
                page = fz_load_page(ctx, doc, pageIdx);
                fz_stext_options opts = { 0 };
                textPage = fz_new_stext_page_from_page(ctx, page, &opts);

                // Walk the structured text: blocks → lines → chars
                for (fz_stext_block* block = textPage->first_block; block; block = block->next) {
                    if (block->type != FZ_STEXT_BLOCK_TEXT) continue;

                    // Accumulate text per block, tracking font sizes
                    std::string blockText;
                    float maxFontSize = 0.0f;
                    float totalFontSize = 0.0f;
                    int charCount = 0;
                    bool hasBold = false;

                    for (fz_stext_line* line = block->u.t.first_line; line; line = line->next) {
                        if (!blockText.empty() && blockText.back() != '\n') {
                            blockText += " ";
                        }

                        for (fz_stext_char* ch = line->first_char; ch; ch = ch->next) {
                            // Get font info
                            float fontSize = ch->size;
                            if (fontSize > maxFontSize) maxFontSize = fontSize;
                            totalFontSize += fontSize;
                            charCount++;

                            // Check for bold font (name typically contains "Bold")
                            if (ch->font) {
                                const char* fontName = fz_font_name(ctx, ch->font);
                                if (fontName) {
                                    std::string fn(fontName);
                                    if (fn.find("Bold") != std::string::npos ||
                                        fn.find("bold") != std::string::npos ||
                                        fn.find("BOLD") != std::string::npos) {
                                        hasBold = true;
                                    }
                                }
                            }

                            // Append UTF-8 character
                            int c = ch->c;
                            if (c < 0x80) {
                                blockText += static_cast<char>(c);
                            } else if (c < 0x800) {
                                blockText += static_cast<char>(0xC0 | (c >> 6));
                                blockText += static_cast<char>(0x80 | (c & 0x3F));
                            } else if (c < 0x10000) {
                                blockText += static_cast<char>(0xE0 | (c >> 12));
                                blockText += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
                                blockText += static_cast<char>(0x80 | (c & 0x3F));
                            } else {
                                blockText += static_cast<char>(0xF0 | (c >> 18));
                                blockText += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
                                blockText += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
                                blockText += static_cast<char>(0x80 | (c & 0x3F));
                            }
                        }
                    }

                    // Skip empty blocks
                    // Trim whitespace
                    auto trimmed = blockText;
                    trimmed.erase(0, trimmed.find_first_not_of(" \t\n\r"));
                    if (!trimmed.empty())
                        trimmed.erase(trimmed.find_last_not_of(" \t\n\r") + 1);

                    if (trimmed.empty()) continue;

                    float avgFontSize = charCount > 0 ? totalFontSize / static_cast<float>(charCount) : 0.0f;

                    RawTextBlock raw;
                    raw.text       = std::move(trimmed);
                    raw.fontSize   = avgFontSize;
                    raw.isBold     = hasBold;
                    raw.pageNumber = pageIdx + 1;  // 1-indexed

                    // Bounding box from block
                    raw.bbox.x = block->bbox.x0;
                    raw.bbox.y = block->bbox.y0;
                    raw.bbox.w = block->bbox.x1 - block->bbox.x0;
                    raw.bbox.h = block->bbox.y1 - block->bbox.y0;

                    blocks.push_back(std::move(raw));
                }
            }
            fz_always(ctx) {
                if (textPage) fz_drop_stext_page(ctx, textPage);
                if (page) fz_drop_page(ctx, page);
            }
            fz_catch(ctx) {
                LOG_WARN("PdfParser", "Error on page " + std::to_string(pageIdx + 1) +
                         ": " + std::string(fz_caught_message(ctx)));
                // Continue with other pages
            }
        }

        fz_drop_document(ctx, doc);
        fz_drop_context(ctx);

        LOG_INFO("PdfParser", "Extracted " + std::to_string(blocks.size()) +
                 " text blocks from " + std::to_string(pageCount) + " pages");

        return blocks;
    }

    std::expected<Core::DocumentTree, Core::ParseError>
    PdfParser::ParseFile(const std::filesystem::path& filePath) const {
        if (!std::filesystem::exists(filePath)) {
            return std::unexpected(Core::ParseError{
                Core::ParseError::Code::FileNotFound,
                "File not found: " + filePath.string()
            });
        }

        if (filePath.extension() != ".pdf") {
            return std::unexpected(Core::ParseError{
                Core::ParseError::Code::UnsupportedFormat,
                "Not a PDF file: " + filePath.string()
            });
        }

        LOG_INFO("PdfParser", "ParseFile() begin: " + filePath.string());

        auto blocksResult = ExtractTextBlocks(filePath);
        if (!blocksResult) {
            return std::unexpected(Core::ParseError{
                Core::ParseError::Code::CorruptFile,
                blocksResult.error()
            });
        }

        auto& blocks = *blocksResult;
        if (blocks.empty()) {
            return std::unexpected(Core::ParseError{
                Core::ParseError::Code::EncodingError,
                "No text content extracted from PDF"
            });
        }

        // Use StructureDetector to classify blocks and build the tree
        auto tree = m_detector.BuildTree(blocks);
        if (tree.IsEmpty()) {
            return std::unexpected(Core::ParseError{
                Core::ParseError::Code::EncodingError,
                "Failed to build document tree from extracted text"
            });
        }

        LOG_INFO("PdfParser", "ParseFile() complete: " +
                 std::to_string(tree.NodeCount()) + " nodes, " +
                 std::to_string(tree.PageCount()) + " pages");

        return tree;
    }

#else
    // ─── No MuPDF fallback ───

    std::expected<std::vector<RawTextBlock>, std::string>
    PdfParser::ExtractTextBlocks(const std::filesystem::path&) const {
        return std::unexpected("PDF support requires MuPDF (libmupdf). "
                               "Add 'libmupdf' to vcpkg.json and rebuild.");
    }

    std::expected<Core::DocumentTree, Core::ParseError>
    PdfParser::ParseFile(const std::filesystem::path& filePath) const {
        LOG_ERROR("PdfParser", "PDF support not compiled — MuPDF not available");
        return std::unexpected(Core::ParseError{
            Core::ParseError::Code::UnsupportedFormat,
            "PDF support requires MuPDF library. Add 'libmupdf' to vcpkg.json and rebuild."
        });
    }

#endif // HAS_MUPDF

} // namespace LocalNotebookLLM::Infrastructure
