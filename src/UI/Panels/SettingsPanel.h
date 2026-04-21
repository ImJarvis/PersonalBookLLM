#pragma once
#include "UI/App.h"
#include "UI/Theme/AppleTheme.h"
#include <imgui.h>

namespace LocalNotebookLLM::UI {

    /// @brief Renders the Settings page — dual-model selection + system info.
    class SettingsPanel {
    public:
        static void Render(App& app) {
            auto& state = app.GetState();

            ImGui::TextColored(ThemeColors::AccentBlue, "Settings");
            ImGui::Separator();
            ImGui::Spacing();

            // ═══════════════════════════════════════════════
            //  Model Pipeline Section
            // ═══════════════════════════════════════════════
            ImGui::TextColored(ThemeColors::TextPrimary, "Dual-Model Pipeline");
            ImGui::TextColored(ThemeColors::TextSecondary,
                "Qwen 2.5 Worker for indexing  |  Gemma Reasoner for Q&A");
            ImGui::Spacing();

            auto models = app.ScanAvailableModels();

            // ─── Reasoner Model Card ───
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ThemeColors::SurfaceElevated);
            ImGui::BeginChild("ReasonerCard", ImVec2(0, 140), ImGuiChildFlags_Borders);
            {
                ImGui::TextColored(ThemeColors::AccentBlue, "Reasoner Model (Q&A)");
                ImGui::TextColored(ThemeColors::TextSecondary,
                    "Answers user questions with citations — requires 4B+ model");
                ImGui::Spacing();

                // Status indicator
                ImVec4 statusColor = state.reasonerLoaded
                    ? ThemeColors::AccentGreen : ThemeColors::AccentOrange;
                const char* statusText = state.reasonerLoaded ? "LOADED" : "NOT LOADED";

                // Show loading animation
                if (state.reasonerStatus.find("Loading") != std::string::npos ||
                    state.reasonerStatus.find("Auto-loading") != std::string::npos) {
                    statusColor = ThemeColors::AccentBlue;
                    statusText = "LOADING...";
                }

                ImGui::TextColored(statusColor, "%s", statusText);
                ImGui::SameLine();
                ImGui::TextColored(ThemeColors::TextSecondary, "  %s",
                    state.reasonerStatus.c_str());
                ImGui::Spacing();

                if (state.reasonerLoaded) {
                    ImGui::TextColored(ThemeColors::AccentGreen, "  %s", state.reasonerName.c_str());
                } else if (models.empty()) {
                    ImGui::TextColored(ThemeColors::AccentRed,
                        "No .gguf models found. Re-run CMake to download models.");
                } else {
                    // Find reasoner candidates and show load buttons
                    for (size_t i = 0; i < models.size(); ++i) {
                        bool isReasoner = models[i].find("gemma") != std::string::npos ||
                                          models[i].find("e2b") != std::string::npos ||
                                          models[i].find("e4b") != std::string::npos ||
                                          models[i].find("4b") != std::string::npos;
                        if (isReasoner || models.size() == 1) {
                            if (ImGui::Button(("Load " + models[i]).c_str(), ImVec2(400, 28))) {
                                app.LoadReasonerModel(app.GetModelsDir() / models[i]);
                            }
                            break;
                        }
                    }
                }
            }
            ImGui::EndChild();
            ImGui::PopStyleColor();

            ImGui::Spacing();

            // ─── Worker Model Card ───
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ThemeColors::SurfaceElevated);
            ImGui::BeginChild("WorkerCard", ImVec2(0, 140), ImGuiChildFlags_Borders);
            {
                ImGui::TextColored(ThemeColors::AccentOrange, "Worker Model (Indexing)");
                ImGui::TextColored(ThemeColors::TextSecondary,
                    "Enriches document structure during ingestion — 1B model");
                ImGui::Spacing();

                // Status indicator
                ImVec4 statusColor = state.workerLoaded
                    ? ThemeColors::AccentGreen : ThemeColors::AccentOrange;
                const char* statusText = state.workerLoaded ? "LOADED" : "NOT LOADED";

                if (state.workerStatus.find("Loading") != std::string::npos ||
                    state.workerStatus.find("Auto-loading") != std::string::npos) {
                    statusColor = ThemeColors::AccentBlue;
                    statusText = "LOADING...";
                }

                ImGui::TextColored(statusColor, "%s", statusText);
                ImGui::SameLine();
                ImGui::TextColored(ThemeColors::TextSecondary, "  %s",
                    state.workerStatus.c_str());
                ImGui::Spacing();

                if (state.workerLoaded) {
                    ImGui::TextColored(ThemeColors::AccentGreen, "  %s", state.workerName.c_str());
                } else {
                    // Find worker candidates and show load buttons
                    bool workerFound = false;
                    for (const auto& m : models) {
                        if (m.find("qwen") != std::string::npos ||
                            m.find("1.5b") != std::string::npos ||
                            m.find("1b") != std::string::npos) {
                            if (ImGui::Button(("Load " + m).c_str(), ImVec2(400, 28))) {
                                app.LoadWorkerModel(app.GetModelsDir() / m);
                            }
                            workerFound = true;
                            break;
                        }
                    }
                    if (!workerFound) {
                        ImGui::TextColored(ThemeColors::TextSecondary,
                            "  Not found — ingestion will use heuristics only");
                    }
                }
            }
            ImGui::EndChild();
            ImGui::PopStyleColor();

            ImGui::Spacing();

            // ─── All Available Models ───
            if (!models.empty()) {
                ImGui::TextColored(ThemeColors::TextPrimary, "All Available Models");
                ImGui::Spacing();

                ImGui::PushStyleColor(ImGuiCol_ChildBg, ThemeColors::SurfaceElevated);
                ImGui::BeginChild("AllModels", ImVec2(0, 30.0f * models.size() + 16),
                                  ImGuiChildFlags_Borders);
                for (const auto& m : models) {
                    // Identify role
                    bool isWorker   = m.find("qwen") != std::string::npos || m.find("1.5b") != std::string::npos || m.find("1b") != std::string::npos;
                    bool isReasoner = m.find("gemma") != std::string::npos ||
                                     m.find("e2b") != std::string::npos ||
                                     m.find("e4b") != std::string::npos ||
                                     m.find("4b") != std::string::npos;

                    // Show loaded indicator
                    bool loaded = (isReasoner && state.reasonerLoaded && state.reasonerName == m) ||
                                  (isWorker && state.workerLoaded && state.workerName == m);

                    ImVec4 tagColor = isReasoner ? ThemeColors::AccentBlue
                                   : isWorker   ? ThemeColors::AccentOrange
                                                : ThemeColors::TextSecondary;
                    const char* tag = isReasoner ? "[Reasoner]"
                                   : isWorker   ? "[Worker]"
                                                : "[Other]";

                    if (loaded) {
                        ImGui::TextColored(ThemeColors::AccentGreen, "%s", "\xe2\x9c\x93");  // ✓
                        ImGui::SameLine();
                    }
                    ImGui::TextColored(tagColor, "%s", tag);
                    ImGui::SameLine();
                    ImGui::Text("%s", m.c_str());
                }
                ImGui::EndChild();
                ImGui::PopStyleColor();
            }

            ImGui::Spacing();
            ImGui::Spacing();

            // ═══════════════════════════════════════════════
            //  About Section
            // ═══════════════════════════════════════════════
            ImGui::TextColored(ThemeColors::TextPrimary, "About");
            ImGui::Spacing();

            ImGui::PushStyleColor(ImGuiCol_ChildBg, ThemeColors::SurfaceElevated);
            ImGui::BeginChild("AboutSection", ImVec2(0, 100), ImGuiChildFlags_Borders);
            {
                ImGui::Text("LocalNotebookLLM  v0.1.0");
                ImGui::TextColored(ThemeColors::TextSecondary,
                    "Offline Document Intelligence for Windows");
                ImGui::Spacing();
                ImGui::TextColored(ThemeColors::TextSecondary,
                    "RAG:  Structural-Hybrid (BM25 + FTS5 + Heading-Boost)");
                ImGui::TextColored(ThemeColors::TextSecondary,
                    "LLM:  llama.cpp  |  UI: Dear ImGui + DX11 + Mica");
            }
            ImGui::EndChild();
            ImGui::PopStyleColor();
        }
    };

} // namespace LocalNotebookLLM::UI
