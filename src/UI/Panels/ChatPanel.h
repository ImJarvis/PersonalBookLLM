#pragma once
#include "UI/App.h"
#include "UI/Theme/AppleTheme.h"
#include <imgui.h>
#include <cstdio>
#include <cmath>
#include <string>

namespace LocalNotebookLLM::UI {

    /// @brief Renders the chat page — message history + query input.
    class ChatPanel {
    public:
        static void Render(App& app) {
            auto& state = app.GetState();

            // ─── Header bar ───
            ImGui::PushFont(nullptr);
            ImGui::TextColored(ThemeColors::AccentBlue, "\xef\x81\xad");
            ImGui::SameLine();
            ImGui::Text("Ask Your Documents");
            ImGui::PopFont();
            ImGui::Separator();
            ImGui::Spacing();

            // ─── Model status bar ───
            if (!state.reasonerLoaded) {
                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.15f, 0.08f, 0.02f, 0.8f));
                ImGui::BeginChild("ModelWarning", ImVec2(0, 42), ImGuiChildFlags_Borders);
                ImGui::TextColored(ThemeColors::AccentOrange, "  Reasoner not loaded.");
                ImGui::SameLine();
                if (state.reasonerStatus.find("Loading") != std::string::npos) {
                    ImGui::TextColored(ThemeColors::TextSecondary, "%s", state.reasonerStatus.c_str());
                } else {
                    ImGui::TextColored(ThemeColors::TextSecondary,
                        "Go to Settings to load a model, or wait for auto-load.");
                }
                ImGui::EndChild();
                ImGui::PopStyleColor();
                ImGui::Spacing();
            }

            // ─── Chat history (scrollable) ───
            float inputHeight = 66.0f;
            ImGui::BeginChild("ChatHistory", ImVec2(0, -inputHeight), ImGuiChildFlags_None);
            {
                if (state.chatHistory.empty() && !state.generating) {
                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    ImGui::SetCursorPosY(avail.y * 0.35f);
                    ImGui::PushStyleColor(ImGuiCol_Text, ThemeColors::TextSecondary);
                    const char* emptyMsg = "Drop a PDF into the Library, then ask questions here.";
                    float textWidth = ImGui::CalcTextSize(emptyMsg).x;
                    ImGui::SetCursorPosX((avail.x - textWidth) * 0.5f);
                    ImGui::TextWrapped("%s", emptyMsg);
                    ImGui::PopStyleColor();
                } else {
                    for (size_t i = 0; i < state.chatHistory.size(); ++i) {
                        RenderMessage(state.chatHistory[i], static_cast<int>(i));
                    }

                    // ─── Live generation bubble ───
                    // Shown as soon as generating=true, before any tokens arrive.
                    // Provides: stage progress bar, elapsed timer, Cancel button,
                    // animated waiting dots, streaming tokens with a pulsing cursor.
                    if (state.generating) {
                        ImGui::Spacing();
                        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.12f, 0.15f, 0.12f, 1.0f));
                        ImGui::BeginChild("StreamingMsg", ImVec2(0, 0),
                                          ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Borders);

                        // Stage progress: Searching → Assembling → Generating → Validating
                        RenderStageBar(state.generationStatus);

                        // ─── Elapsed timer + Cancel button (same line) ───
                        {
                            int elapsed = static_cast<int>(state.elapsedGenerationSec);
                            int mins = elapsed / 60;
                            int secs = elapsed % 60;
                            char timeBuf[32];
                            if (mins > 0)
                                std::snprintf(timeBuf, sizeof(timeBuf), "%dm %02ds", mins, secs);
                            else
                                std::snprintf(timeBuf, sizeof(timeBuf), "%ds", secs);

                            ImGui::TextColored(ThemeColors::TextSecondary,
                                "  Elapsed: %s", timeBuf);
                            ImGui::SameLine();

                            // Cancel button — right-aligned, red accent
                            float cancelW = 70.0f;
                            float availW = ImGui::GetContentRegionAvail().x;
                            if (availW > cancelW + 10) {
                                ImGui::SameLine(ImGui::GetCursorPosX() + availW - cancelW);
                            }
                            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.12f, 0.12f, 0.9f));
                            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.7f, 0.18f, 0.18f, 1.0f));
                            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.9f, 0.2f, 0.2f, 1.0f));
                            if (ImGui::SmallButton("Cancel")) {
                                app.CancelGeneration();
                            }
                            ImGui::PopStyleColor(3);
                        }
                        ImGui::Spacing();

                        if (!state.streamingAnswer.empty()) {
                            // Live token stream with pulsing block cursor
                            ImGui::TextWrapped("%s", state.streamingAnswer.c_str());
                            ImGui::SameLine(0, 0);
                            // Flash at ~1 Hz using sine
                            float t = static_cast<float>(
                                std::fmod(ImGui::GetTime() * 2.0, 2.0));
                            if (t < 1.0f) {
                                ImGui::TextColored(ThemeColors::AccentGreen, "\xe2\x96\x8b"); // ▋
                            } else {
                                ImGui::TextUnformatted(" ");
                            }
                        } else {
                            // No tokens yet — show detailed eval progress or generic waiting
                            const auto& status = state.generationStatus;
                            bool isEvaluating = (status.find("Evaluating") != std::string::npos);

                            if (isEvaluating) {
                                // Show a progress bar during context evaluation
                                ImGui::TextColored(ThemeColors::AccentBlue,
                                    "  %s", status.c_str());
                                ImGui::Spacing();
                                ImGui::TextColored(ThemeColors::TextSecondary,
                                    "  CPU inference — this may take a few minutes on first query.");
                                ImGui::TextColored(ThemeColors::TextSecondary,
                                    "  Follow-up questions will be much faster (KV cache reuse).");
                            } else {
                                // Animate waiting dots
                                int dots = static_cast<int>(
                                    std::fmod(ImGui::GetTime() * 1.5, 4.0));
                                std::string waiting = "Thinking";
                                for (int d = 0; d < dots; ++d) waiting += '.';
                                ImGui::TextColored(ThemeColors::TextSecondary, "%s", waiting.c_str());
                            }
                        }

                        ImGui::EndChild();
                        ImGui::PopStyleColor();

                        // Always keep the live bubble visible
                        ImGui::SetScrollHereY(1.0f);
                    } else {
                        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 20.0f) {
                            ImGui::SetScrollHereY(1.0f);
                        }
                    }
                }
            }
            ImGui::EndChild();

            // ─── Input bar ───
            ImGui::Separator();
            ImGui::Spacing();

            bool canSend = state.reasonerLoaded && !state.generating && !state.documents.empty();
            if (!canSend) ImGui::BeginDisabled();

            ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 90);
            bool enterPressed = ImGui::InputText("##QueryInput", state.queryBuffer,
                sizeof(state.queryBuffer),
                ImGuiInputTextFlags_EnterReturnsTrue);
            ImGui::PopItemWidth();

            ImGui::SameLine();
            bool sendClicked = ImGui::Button("Send", ImVec2(75, 0));

            if ((enterPressed || sendClicked) && state.queryBuffer[0] != '\0') {
                app.AskQuestion(std::string(state.queryBuffer));
                state.queryBuffer[0] = '\0';
            }

            if (!canSend) ImGui::EndDisabled();
        }

    private:
        // ─── Stage progress bar ───────────────────────────────────────────────────
        // Shows all 4 pipeline stages in one line:
        //   ✓ Searching  →  ✓ Assembling  →  [Generating]  →  Validating
        //
        // - Completed stages: dim green with ✓ checkmark
        // - Active stage:     bright pulsing green
        // - Pending stages:   dim grey
        static void RenderStageBar(const std::string& statusStr) {
            enum Stage { S_Searching, S_Assembling, S_Generating, S_Validating, S_Count };
            struct StageInfo { const char* label; const char* prefix; };
            static const StageInfo stages[S_Count] = {
                { "Searching",  "Searching" },
                { "Assembling", "Assembl"   },
                { "Generating", "Generat"   },
                { "Validating", "Validat"   },
            };

            // Walk backwards to find the latest matching stage
            int activeStage = S_Searching;
            for (int i = S_Count - 1; i >= 0; --i) {
                if (statusStr.find(stages[i].prefix) != std::string::npos) {
                    activeStage = i;
                    break;
                }
            }

            // Pulse amplitude for the active chip
            float pulse = 0.6f + 0.4f * static_cast<float>(std::sin(ImGui::GetTime() * 4.0));

            ImGui::TextColored(ThemeColors::AccentGreen, "AI");
            ImGui::SameLine();
            ImGui::TextColored(ThemeColors::TextSecondary, "|");

            for (int i = 0; i < S_Count; ++i) {
                ImGui::SameLine();
                if (i == activeStage) {
                    ImGui::TextColored(
                        ImVec4(0.3f * pulse, 0.9f * pulse, 0.4f * pulse, 1.0f),
                        "%s", stages[i].label);
                } else if (i < activeStage) {
                    // Completed stage — dim green tick
                    ImGui::TextColored(
                        ImVec4(0.25f, 0.65f, 0.3f, 0.7f),
                        "\xe2\x9c\x93 %s", stages[i].label);  // ✓
                } else {
                    // Pending stage — grey
                    ImGui::TextColored(
                        ImVec4(0.4f, 0.4f, 0.4f, 0.5f),
                        "%s", stages[i].label);
                }
                if (i < S_Count - 1) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(0.3f, 0.3f, 0.3f, 0.7f),
                                       "\xe2\x86\x92");  // →
                }
            }
        }

        // ─── Message bubble ───────────────────────────────────────────────────────
        static void RenderMessage(const ChatMessage& msg, int index) {
            ImGui::PushID(index);
            ImGui::Spacing();

            ImVec4 roleColor;
            const char* roleLabel;
            switch (msg.role) {
                case ChatMessage::Role::User:
                    roleColor = ThemeColors::AccentBlue;
                    roleLabel = "You";
                    break;
                case ChatMessage::Role::Assistant:
                    roleColor = ThemeColors::AccentGreen;
                    roleLabel = "AI";
                    break;
                case ChatMessage::Role::System:
                    roleColor = ThemeColors::AccentRed;
                    roleLabel = "System";
                    break;
            }

            ImVec4 bgColor = (msg.role == ChatMessage::Role::User)
                ? ImVec4(0.10f, 0.20f, 0.35f, 0.6f)
                : ImVec4(0.14f, 0.14f, 0.16f, 1.0f);

            ImGui::PushStyleColor(ImGuiCol_ChildBg, bgColor);
            ImGui::BeginChild(("Msg" + std::to_string(index)).c_str(),
                              ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Borders);

            ImGui::TextColored(roleColor, "%s", roleLabel);

            if (msg.role == ChatMessage::Role::Assistant && msg.generationTimeMs > 0) {
                ImGui::SameLine();
                ImGui::TextColored(ThemeColors::TextSecondary, "| %.1fs | %d sources",
                    msg.generationTimeMs / 1000.0, msg.sectionsSearched);
            }

            ImGui::TextWrapped("%s", msg.text.c_str());

            if (!msg.citations.empty()) {
                ImGui::Spacing();
                ImGui::TextColored(ThemeColors::TextSecondary, "Sources:");
                for (const auto& cite : msg.citations) {
                    ImGui::BulletText("Page %d \xe2\x80\x94 %s", cite.pageNumber,
                                      cite.sectionPath.c_str());
                }
            }

            ImGui::EndChild();
            ImGui::PopStyleColor();
            ImGui::PopID();
        }
    };

} // namespace LocalNotebookLLM::UI
