#pragma once
#include "UI/App.h"
#include <imgui.h>

namespace LocalNotebookLLM::UI {

    /// @brief Renders the chat page — message history + query input.
    class ChatPanel {
    public:
        /// Render the full chat panel into the current ImGui region.
        static void Render(App& app) {
            auto& state = app.GetState();

            // ─── Header bar ───
            ImGui::PushFont(nullptr);  // Will use default; swap to heading font if available
            ImGui::TextColored(ThemeColors::AccentBlue, "\xef\x81\xad");  // FontAwesome comments icon
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
                if (state.chatHistory.empty()) {
                    // Empty state
                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    ImGui::SetCursorPosY(avail.y * 0.35f);
                    ImGui::PushStyleColor(ImGuiCol_Text, ThemeColors::TextSecondary);

                    // Center the text
                    const char* emptyMsg = "Drop a PDF into the Library, then ask questions here.";
                    float textWidth = ImGui::CalcTextSize(emptyMsg).x;
                    ImGui::SetCursorPosX((avail.x - textWidth) * 0.5f);
                    ImGui::TextWrapped("%s", emptyMsg);
                    ImGui::PopStyleColor();
                } else {
                    for (size_t i = 0; i < state.chatHistory.size(); ++i) {
                        RenderMessage(state.chatHistory[i], static_cast<int>(i));
                    }

                    // Show streaming answer in progress
                    if (state.generating && !state.streamingAnswer.empty()) {
                        ImGui::Spacing();
                        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.14f, 0.14f, 0.16f, 1.0f));
                        ImGui::BeginChild("StreamingMsg", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY);
                        ImGui::PushStyleColor(ImGuiCol_Text, ThemeColors::AccentGreen);
                        ImGui::TextUnformatted("AI");
                        ImGui::PopStyleColor();
                        ImGui::SameLine();
                        ImGui::TextColored(ThemeColors::TextSecondary, "| %s",
                                           state.generationStatus.c_str());
                        ImGui::TextWrapped("%s", state.streamingAnswer.c_str());
                        ImGui::EndChild();
                        ImGui::PopStyleColor();
                    }

                    // Auto-scroll to bottom
                    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 20.0f) {
                        ImGui::SetScrollHereY(1.0f);
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

            if (state.generating) {
                ImGui::SameLine();
                ImGui::TextColored(ThemeColors::TextSecondary, "%s",
                                   state.generationStatus.c_str());
            }
        }

    private:
        static void RenderMessage(const ChatMessage& msg, int index) {
            ImGui::PushID(index);
            ImGui::Spacing();

            // Choose color/label by role
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

            // Message bubble
            ImVec4 bgColor = (msg.role == ChatMessage::Role::User)
                ? ImVec4(0.10f, 0.20f, 0.35f, 0.6f)
                : ImVec4(0.14f, 0.14f, 0.16f, 1.0f);

            ImGui::PushStyleColor(ImGuiCol_ChildBg, bgColor);
            ImGui::BeginChild(("Msg" + std::to_string(index)).c_str(),
                              ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Borders);

            // Role label
            ImGui::TextColored(roleColor, "%s", roleLabel);

            // Time/stats for assistant
            if (msg.role == ChatMessage::Role::Assistant && msg.generationTimeMs > 0) {
                ImGui::SameLine();
                ImGui::TextColored(ThemeColors::TextSecondary, "| %.1fs | %d sources",
                    msg.generationTimeMs / 1000.0, msg.sectionsSearched);
            }

            // Message text
            ImGui::TextWrapped("%s", msg.text.c_str());

            // Citations
            if (!msg.citations.empty()) {
                ImGui::Spacing();
                ImGui::TextColored(ThemeColors::TextSecondary, "Sources:");
                for (const auto& cite : msg.citations) {
                    ImGui::BulletText("Page %d — %s", cite.pageNumber,
                                      cite.sectionPath.c_str());
                }
            }

            ImGui::EndChild();
            ImGui::PopStyleColor();
            ImGui::PopID();
        }
    };

} // namespace LocalNotebookLLM::UI
