#include "Infrastructure/Parsing/PdfVlmParser.h"
#include "Core/Log.h"
#include <cstdlib>
#include <fstream>
#include <sstream>

#ifdef HAS_MUPDF
extern "C" {
#include <mupdf/fitz.h>
}
#endif

namespace LocalNotebookLLM::Infrastructure {

    std::expected<std::string, std::string> 
    PdfVlmParser::RunVlmOnImage(const std::filesystem::path& imagePath) const {
        // Run the python script to process the image via granite-docling
        std::string pyScript = "scripts/run_docling.py";
        std::string command = "python " + pyScript + " --image \"" + imagePath.string() + "\" > \"" + imagePath.string() + ".md\"";
        
        LOG_INFO("PdfVlmParser", "Running VLM: " + command);
        int result = std::system(command.c_str());
        
        if (result != 0) {
            return std::unexpected("VLM subprocess failed with code " + std::to_string(result));
        }
        
        std::ifstream ifs(imagePath.string() + ".md");
        if (!ifs) {
            return std::unexpected("Failed to read VLM output markdown");
        }
        
        std::stringstream buffer;
        buffer << ifs.rdbuf();
        return buffer.str();
    }

    std::expected<Core::DocumentTree, Core::ParseError>
    PdfVlmParser::ParseFile(const std::filesystem::path& filePath) const {
        // Try heuristic MuPDF parser first (fallback for simple text-only PDFs)
        auto fallbackTree = m_fallbackParser.ParseFile(filePath);
        if (fallbackTree) {
            // Check if text extraction was successful (heuristic: average nodes per page)
            int pageCount = fallbackTree->PageCount();
            size_t nodeCount = fallbackTree->NodeCount();
            
            // If we got at least 3 structural nodes per page on average, it's a good text PDF.
            // Otherwise, it's likely a scanned document or heavily image-based layout, so use VLM.
            if (pageCount > 0 && (static_cast<float>(nodeCount) / pageCount) >= 3.0f) {
                LOG_INFO("PdfVlmParser", "Heuristic parser succeeded with good text density. Skipping VLM.");
                return fallbackTree;
            } else {
                LOG_INFO("PdfVlmParser", "Heuristic parser returned low text density. Falling back to VLM for OCR/Layout.");
            }
        } else {
            LOG_WARN("PdfVlmParser", "Heuristic parser failed. Falling back to VLM.");
        }

#ifdef HAS_MUPDF
        LOG_INFO("PdfVlmParser", "Starting VLM parsing for " + filePath.string());
        
        fz_context* ctx = fz_new_context(nullptr, nullptr, FZ_STORE_DEFAULT);
        if (!ctx) return std::unexpected(Core::ParseError{Core::ParseError::Code::CorruptFile, "MuPDF ctx failed"});

        fz_document* doc = nullptr;
        fz_try(ctx) {
            fz_register_document_handlers(ctx);
            doc = fz_open_document(ctx, filePath.string().c_str());
        }
        fz_catch(ctx) {
            fz_drop_context(ctx);
            return std::unexpected(Core::ParseError{Core::ParseError::Code::CorruptFile, "Failed to open PDF"});
        }

        int pageCount = fz_count_pages(ctx, doc);
        std::string fullMarkdown;

        for (int i = 0; i < pageCount; ++i) {
            fz_page* page = nullptr;
            fz_pixmap* pix = nullptr;
            std::string tempImgPath = "temp_page_" + std::to_string(i) + ".png";
            
            fz_try(ctx) {
                page = fz_load_page(ctx, doc, i);
                fz_matrix ctm = fz_scale(2.0f, 2.0f); // 200% scale for better OCR
                pix = fz_new_pixmap_from_page(ctx, page, ctm, fz_device_rgb(ctx), 0);
                fz_save_pixmap_as_png(ctx, pix, tempImgPath.c_str());
            }
            fz_always(ctx) {
                if (pix) fz_drop_pixmap(ctx, pix);
                if (page) fz_drop_page(ctx, page);
            }
            fz_catch(ctx) {
                LOG_WARN("PdfVlmParser", "Failed to render page " + std::to_string(i));
                continue;
            }

            // Run VLM on the rendered image
            auto mdResult = RunVlmOnImage(tempImgPath);
            if (mdResult) {
                fullMarkdown += *mdResult + "\n\n";
            } else {
                LOG_WARN("PdfVlmParser", "VLM failed on page " + std::to_string(i) + ": " + mdResult.error());
            }
            
            std::filesystem::remove(tempImgPath);
            std::filesystem::remove(tempImgPath + ".md");
        }

        fz_drop_document(ctx, doc);
        fz_drop_context(ctx);
        
        // Parse the markdown output into DocumentTree via StructureDetector
        std::vector<RawTextBlock> blocks;
        std::istringstream stream(fullMarkdown);
        std::string line;
        
        while (std::getline(stream, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            if (line.empty()) continue;

            RawTextBlock block;
            block.pageNumber = 1; // Assuming sequential parsing or flatten
            block.isBold = false;
            block.fontSize = 10.0f; // BODY_SIZE

            if (line.starts_with("# ")) {
                block.text = line.substr(2);
                block.fontSize = 16.0f; // H1_MIN_SIZE
                block.isBold = true;
            } else if (line.starts_with("## ")) {
                block.text = line.substr(3);
                block.fontSize = 13.0f; // H2_MIN_SIZE
                block.isBold = true;
            } else if (line.starts_with("### ")) {
                block.text = line.substr(4);
                block.fontSize = 11.5f; // H3_MIN_SIZE
                block.isBold = true;
            } else if (line.starts_with("**") && line.ends_with("**")) {
                block.text = line.substr(2, line.length() - 4);
                block.isBold = true;
            } else {
                block.text = line;
            }
            blocks.push_back(std::move(block));
        }

        return m_detector.BuildTree(blocks);
#else
        return std::unexpected(Core::ParseError{Core::ParseError::Code::UnsupportedFormat, "Requires MuPDF"});
#endif
    }

} // namespace LocalNotebookLLM::Infrastructure
