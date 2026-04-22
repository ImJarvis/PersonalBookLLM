#pragma once
#include "UI/App.h"
#include "UI/Theme/AppleTheme.h"
#include <imgui.h>
#include <filesystem>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#pragma comment(lib, "comdlg32.lib")
#endif
namespace LocalNotebookLLM::UI {

    /// @brief Renders the Library page — document list + drag-and-drop ingestion.
    class LibraryPanel {
    public:
        static void Render(App& app) {
            auto& state = app.GetState();

            // ─── Header ───
            ImGui::TextColored(ThemeColors::AccentBlue, "\xef\x80\xad");  // book icon
            ImGui::SameLine();
            ImGui::Text("Document Library");
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - 120);
            ImGui::TextColored(ThemeColors::TextSecondary, "%zu documents",
                               state.documents.size());
            ImGui::Separator();
            ImGui::Spacing();

            // ─── Worker model status bar ───
            {
                ImVec4 workerBg = state.workerLoaded
                    ? ImVec4(0.05f, 0.12f, 0.05f, 0.6f)
                    : ImVec4(0.15f, 0.10f, 0.03f, 0.6f);
                ImGui::PushStyleColor(ImGuiCol_ChildBg, workerBg);
                ImGui::BeginChild("WorkerStatusBar", ImVec2(0, 28), ImGuiChildFlags_Borders);
                ImVec4 wColor = state.workerLoaded
                    ? ThemeColors::AccentGreen : ThemeColors::AccentOrange;
                ImGui::TextColored(wColor, "  Worker: %s",
                    state.workerLoaded
                        ? state.workerName.c_str()
                        : state.workerStatus.c_str());
                if (state.workerLoaded) {
                    ImGui::SameLine();
                    ImGui::TextColored(ThemeColors::TextSecondary,
                        " — Ready for document ingestion");
                }
                ImGui::EndChild();
                ImGui::PopStyleColor();
                ImGui::Spacing();
            }

            // ─── Ingestion status bar (active progress) ───
            if (state.ingesting) {
                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.05f, 0.15f, 0.05f, 0.8f));
                ImGui::BeginChild("IngestionBar", ImVec2(0, 50), ImGuiChildFlags_Borders);
                ImGui::TextColored(ThemeColors::AccentGreen, "  Ingesting...");
                ImGui::ProgressBar(state.ingestionProgress / 100.0f, ImVec2(-1, 0),
                                   state.ingestionStatus.c_str());
                ImGui::EndChild();
                ImGui::PopStyleColor();
                ImGui::Spacing();
            }

            // ─── Persistent error banner (shown even after ingesting=false) ───
            if (!state.lastIngestionError.empty()) {
                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.22f, 0.06f, 0.06f, 0.95f));
                ImGui::PushStyleColor(ImGuiCol_Border,   ImVec4(0.80f, 0.20f, 0.20f, 1.0f));
                ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.5f);
                ImGui::BeginChild("IngestionErrorBar", ImVec2(0, 52), ImGuiChildFlags_Borders);
                {
                    ImGui::Spacing();
                    ImGui::SetCursorPosX(8);
                    ImGui::TextColored(ImVec4(1.0f, 0.40f, 0.40f, 1.0f), "  Could not load document:");
                    ImGui::SameLine();
                    // Dismiss button, right-aligned
                    float dismissX = ImGui::GetContentRegionAvail().x - 60;
                    if (dismissX > 0) ImGui::SetCursorPosX(dismissX + ImGui::GetCursorPosX() - ImGui::GetContentRegionAvail().x + dismissX);
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.4f, 0.1f, 0.1f, 0.8f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.6f, 0.15f, 0.15f, 1.0f));
                    if (ImGui::SmallButton("Dismiss")) {
                        state.lastIngestionError.clear();
                    }
                    ImGui::PopStyleColor(2);
                    ImGui::SetCursorPosX(8);
                    // Truncate long error messages for display
                    std::string displayErr = state.lastIngestionError;
                    if (displayErr.size() > 120) displayErr = displayErr.substr(0, 117) + "...";
                    ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.75f, 1.0f), "  %s", displayErr.c_str());
                }
                ImGui::EndChild();
                ImGui::PopStyleVar();
                ImGui::PopStyleColor(2);
                ImGui::Spacing();
            }

            // ─── Drop zone (when no documents) ───
            if (state.documents.empty() && !state.ingesting) {
                RenderDropZone(app);
                return;
            }

            // ─── Document cards ───
            for (size_t i = 0; i < state.documents.size(); ++i) {
                RenderDocumentCard(app, state.documents[i], static_cast<int>(i));
            }

            ImGui::Spacing();
            ImGui::Spacing();

            // ─── Add more button ───
            float buttonWidth = 200.0f;
            ImGui::SetCursorPosX((ImGui::GetContentRegionAvail().x - buttonWidth) * 0.5f);
            if (ImGui::Button("+ Add Document", ImVec2(buttonWidth, 36))) {
                // Open file dialog (Win32)
                OpenFileDialog(app);
            }
        }

    private:
        static void RenderDropZone(App& app) {
            ImVec2 avail = ImGui::GetContentRegionAvail();

            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.15f, 0.15f, 0.17f, 0.5f));
            ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 2.0f);
            ImGui::PushStyleColor(ImGuiCol_Border, ThemeColors::AccentBlue);

            ImGui::BeginChild("DropZone", ImVec2(avail.x, avail.y - 20),
                              ImGuiChildFlags_Borders);
            {
                ImVec2 center = ImVec2(avail.x * 0.5f, avail.y * 0.4f);
                ImGui::SetCursorPos(center);

                // Centered text
                const char* title = "Drop PDF or DOCX files here";
                float tw = ImGui::CalcTextSize(title).x;
                ImGui::SetCursorPosX((avail.x - tw) * 0.5f);
                ImGui::TextColored(ThemeColors::TextPrimary, "%s", title);

                const char* subtitle = "or click 'Add Document' below";
                float sw = ImGui::CalcTextSize(subtitle).x;
                ImGui::SetCursorPosX((avail.x - sw) * 0.5f);
                ImGui::TextColored(ThemeColors::TextSecondary, "%s", subtitle);

                ImGui::Spacing();
                ImGui::Spacing();

                float bw = 200.0f;
                ImGui::SetCursorPosX((avail.x - bw) * 0.5f);
                if (ImGui::Button("Add Document", ImVec2(bw, 36))) {
                    OpenFileDialog(app);
                }
            }
            ImGui::EndChild();
            ImGui::PopStyleColor(2);
            ImGui::PopStyleVar();
        }

        static void RenderDocumentCard(App& app, const Core::DocumentMetadata& doc, int index) {
            ImGui::PushID(index);

            ImGui::PushStyleColor(ImGuiCol_ChildBg, ThemeColors::SurfaceElevated);
            ImGui::BeginChild(("DocCard" + std::to_string(index)).c_str(),
                              ImVec2(0, 72), ImGuiChildFlags_Borders);
            {
                // File icon + name
                const char* icon = "\xef\x87\x85";  // file icon
                ImGui::TextColored(ThemeColors::AccentBlue, "%s", icon);
                ImGui::SameLine();
                ImGui::Text("%s", doc.filename.c_str());

                // Metadata line
                ImGui::TextColored(ThemeColors::TextSecondary,
                    "  %d pages  |  %zu sections  |  %s",
                    doc.pageCount,
                    doc.nodeCount,
                    doc.ingestedAt.empty() ? "\xe2\x80\x94" : doc.ingestedAt.substr(0, 10).c_str());

                // Delete button (right-aligned)
                ImGui::SameLine(ImGui::GetContentRegionAvail().x - 30);
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.1f, 0.1f, 0.5f));
                if (ImGui::Button("X", ImVec2(24, 24))) {
                    app.RemoveDocument(doc.id);
                }
                ImGui::PopStyleColor(2);
            }
            ImGui::EndChild();
            ImGui::PopStyleColor();
            ImGui::Spacing();
            ImGui::PopID();
        }

        static void OpenFileDialog(App& app) {
#ifdef _WIN32
            // Use Win32 GetOpenFileName
            char filename[MAX_PATH] = {};
            OPENFILENAMEA ofn = {};
            ofn.lStructSize = sizeof(ofn);
            ofn.lpstrFilter = "Documents\0*.pdf;*.docx;*.txt\0PDF Files\0*.pdf\0Word Documents\0*.docx\0All Files\0*.*\0";
            ofn.lpstrFile   = filename;
            ofn.nMaxFile    = MAX_PATH;
            ofn.Flags       = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
            ofn.lpstrTitle  = "Select Document to Ingest";

            if (GetOpenFileNameA(&ofn)) {
                app.IngestDocument(std::filesystem::path(filename));
            }
#endif
        }
    };

} // namespace LocalNotebookLLM::UI
