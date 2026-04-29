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
                        if (state.chatHistory[i].isHidden) continue;
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
                            if (state.generationCancelled) {
                                ImGui::TextColored(ThemeColors::TextSecondary, "Cancelling...");
                            } else {
                                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.12f, 0.12f, 0.9f));
                                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.7f, 0.18f, 0.18f, 1.0f));
                                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.9f, 0.2f, 0.2f, 1.0f));
                                if (ImGui::SmallButton("Cancel")) {
                                    app.CancelGeneration();
                                }
                                ImGui::PopStyleColor(3);
                            }
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

        // ─── Custom Markdown & Wrapped Text Renderer ──────────────────────────────
        static void RenderMarkdown(const std::string& text) {
            float start_x = ImGui::GetCursorPosX();
            float wrap_width = start_x + ImGui::GetContentRegionAvail().x;
            
            bool in_bold = false;
            float oldSpacing = ImGui::GetStyle().ItemSpacing.y;
            ImGui::GetStyle().ItemSpacing.y = 8.0f; // Add breathing room between paragraphs
            
            auto renderWord = [&](const std::string& word, bool isBold, bool isCitation, bool hasSpaceAfter) {
                if (word.empty()) return;
                ImVec2 textSize = ImGui::CalcTextSize(word.c_str());
                if (ImGui::GetCursorPosX() + textSize.x > wrap_width && ImGui::GetCursorPosX() > start_x) {
                    ImGui::NewLine();
                    ImGui::SetCursorPosX(start_x);
                }
                
                ImVec4 color = ThemeColors::TextPrimary;
                if (isBold) color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // Bright white for bold
                if (isCitation) color = ThemeColors::AccentBlue;

                ImVec2 p_min = ImGui::GetCursorScreenPos();
                ImGui::TextColored(color, "%s", word.c_str());
                ImVec2 p_max = ImVec2(p_min.x + textSize.x, p_min.y + textSize.y);
                
                if (isCitation) {
                    // Underline
                    ImGui::GetWindowDrawList()->AddLine(
                        ImVec2(p_min.x, p_max.y), ImVec2(p_max.x, p_max.y),
                        ImGui::ColorConvertFloat4ToU32(color), 1.0f);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                        ImGui::SetTooltip("Citation reference");
                    }
                }
                
                ImGui::SameLine(0, 0);
                if (hasSpaceAfter) {
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::CalcTextSize(" ").x);
                }
            };

            size_t i = 0;
            while (i < text.size()) {
                if (text[i] == '\n') {
                    ImGui::NewLine();
                    if (i + 1 < text.size() && text[i+1] == '\n') {
                        ImGui::Spacing();
                        i++;
                    }
                    ImGui::SetCursorPosX(start_x);
                    i++;
                    continue;
                }
                
                if (text[i] == '*' && i + 1 < text.size() && text[i+1] == '*') {
                    in_bold = !in_bold;
                    i += 2;
                    continue;
                }
                
                std::string word;
                bool isCitation = false;
                if (text[i] == '[') {
                    size_t end = text.find(']', i);
                    if (end != std::string::npos) {
                        word = text.substr(i, end - i + 1);
                        isCitation = true;
                        i = end + 1;
                    } else {
                        word += text[i++];
                    }
                } else {
                    while (i < text.size() && text[i] != ' ' && text[i] != '\n' && text[i] != '[' && !(text[i] == '*' && i+1 < text.size() && text[i+1] == '*')) {
                        word += text[i++];
                    }
                }
                
                bool hasSpace = false;
                if (i < text.size() && text[i] == ' ') {
                    hasSpace = true;
                    while(i < text.size() && text[i] == ' ') i++; // skip multiple spaces
                }
                
                renderWord(word, in_bold, isCitation, hasSpace);
            }
            ImGui::NewLine();
            ImGui::GetStyle().ItemSpacing.y = oldSpacing;
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

            if (msg.role == ChatMessage::Role::Assistant) {
                if (msg.generationTimeMs > 0) {
                    ImGui::SameLine();
                    ImGui::TextColored(ThemeColors::TextSecondary, "| %.1fs | %d sources",
                        msg.generationTimeMs / 1000.0, msg.sectionsSearched);
                }
                
                // Copy Button
                ImGui::SameLine(ImGui::GetContentRegionAvail().x - 50.0f);
                if (ImGui::Button("Copy", ImVec2(50, 0))) {
                    ImGui::SetClipboardText(msg.text.c_str());
                }
            }

            ImGui::Spacing();
            
            // Render text with improved paragraph spacing to reduce density
            RenderMarkdown(msg.text);

            ImGui::EndChild();
            ImGui::PopStyleColor();
            ImGui::PopID();
        }
    };

} // namespace LocalNotebookLLM::UI
