#include "Infrastructure/Parsing/PdfVlmParser.h"
#include "Core/Log.h"
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef HAS_MUPDF
extern "C" {
#include <mupdf/fitz.h>
}
#endif

namespace LocalNotebookLLM::Infrastructure {

    static const std::filesystem::path& GetDoclingScriptPath() {
        static std::filesystem::path scriptPath = []() -> std::filesystem::path {
            std::filesystem::path relativePath = "scripts/run_docling.py";
            if (std::filesystem::exists(relativePath)) {
                return std::filesystem::absolute(relativePath);
            }
#ifdef _WIN32
            char buffer[MAX_PATH];
            if (GetModuleFileNameA(NULL, buffer, MAX_PATH)) {
                std::filesystem::path dir = std::filesystem::path(buffer).parent_path();
                for (int i = 0; i < 5 && dir.has_parent_path(); ++i) {
                    std::filesystem::path candidate = dir / "scripts" / "run_docling.py";
                    if (std::filesystem::exists(candidate)) {
                        return candidate;
                    }
                    dir = dir.parent_path();
                }
            }
#endif
            return relativePath; // fallback
        }();
        return scriptPath;
    }

    std::expected<std::string, std::string> 
    PdfVlmParser::RunVlmOnImage(const std::filesystem::path& imagePath) const {
        std::string pyScript = GetDoclingScriptPath().string();
        std::string mdPath = imagePath.string() + ".md";
        
        // Use direct process creation on Windows to avoid cmd.exe shell injection
#ifdef _WIN32
        std::string cmdArgs = "python \"" + pyScript + "\" --image \"" + imagePath.string() + "\" --output \"" + mdPath + "\"";
        
        STARTUPINFOA si;
        PROCESS_INFORMATION pi;
        ZeroMemory(&si, sizeof(si));
        si.cb = sizeof(si);
        ZeroMemory(&pi, sizeof(pi));

        std::vector<char> cmdBuffer(cmdArgs.begin(), cmdArgs.end());
        cmdBuffer.push_back('\0');

        LOG_INFO("PdfVlmParser", "Running VLM (Secure): " + cmdArgs);

        if (!CreateProcessA(NULL, cmdBuffer.data(), NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
            return std::unexpected("VLM CreateProcess failed with error " + std::to_string(GetLastError()));
        }

        WaitForSingleObject(pi.hProcess, INFINITE);

        DWORD exitCode;
        GetExitCodeProcess(pi.hProcess, &exitCode);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);

        if (exitCode != 0) {
            return std::unexpected("VLM subprocess failed with code " + std::to_string(exitCode));
        }
#else
        std::string command = "python \"" + pyScript + "\" --image \"" + imagePath.string() + "\" --output \"" + mdPath + "\"";
        LOG_INFO("PdfVlmParser", "Running VLM: " + command);
        int result = std::system(command.c_str());
        if (result != 0) {
            return std::unexpected("VLM subprocess failed with code " + std::to_string(result));
        }
#endif
        
        std::ifstream ifs(mdPath);
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
        // Split markdown by page boundaries (each page's output is separated by double newlines)
        // Track the current page number so each block gets the correct page reference.
        std::vector<RawTextBlock> blocks;
        
        // Split fullMarkdown into per-page segments using the "\n\n" separators we added above
        std::vector<std::pair<int, std::string>> pageSegments;
        {
            std::istringstream segStream(fullMarkdown);
            std::string segment;
            int currentPage = 1;
            std::string accumulated;
            std::string line;
            while (std::getline(segStream, line)) {
                if (line.empty() && !accumulated.empty()) {
                    // Check if this is a page boundary (double blank line)
                    // Each page's VLM output was separated by "\n\n"
                    pageSegments.push_back({currentPage, accumulated});
                    accumulated.clear();
                    if (currentPage < pageCount) currentPage++;
                } else if (!line.empty()) {
                    if (!accumulated.empty()) accumulated += "\n";
                    accumulated += line;
                }
            }
            if (!accumulated.empty()) {
                pageSegments.push_back({currentPage, accumulated});
            }
        }
        
        // If page splitting didn't work well, fall back to sequential assignment
        if (pageSegments.empty()) {
            std::istringstream fallbackStream(fullMarkdown);
            std::string line;
            while (std::getline(fallbackStream, line)) {
                line.erase(0, line.find_first_not_of(" \t\r\n"));
                line.erase(line.find_last_not_of(" \t\r\n") + 1);
                if (line.empty()) continue;

                RawTextBlock block;
                block.pageNumber = 1;
                block.isBold = false;
                block.fontSize = 10.0f;
                block.text = line;
                blocks.push_back(std::move(block));
            }
        } else {
            for (const auto& [pageNum, content] : pageSegments) {
                std::istringstream lineStream(content);
                std::string line;
                while (std::getline(lineStream, line)) {
                    line.erase(0, line.find_first_not_of(" \t\r\n"));
                    line.erase(line.find_last_not_of(" \t\r\n") + 1);
                    if (line.empty()) continue;

                    RawTextBlock block;
                    block.pageNumber = pageNum;  // Correct page number from VLM loop
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
            }
        }

        return m_detector.BuildTree(blocks);
#else
        return std::unexpected(Core::ParseError{Core::ParseError::Code::UnsupportedFormat, "Requires MuPDF"});
#endif
    }

} // namespace LocalNotebookLLM::Infrastructure
