#pragma once
#include <imgui.h>

namespace LocalNotebookLLM::UI {

    struct ThemeColors {
        // Dark mode (primary) — Apple-inspired color tokens
        static constexpr ImVec4 Surface         = {0.110f, 0.110f, 0.118f, 0.85f}; // #1C1C1E, 85% for Mica bleed
        static constexpr ImVec4 SurfaceElevated  = {0.173f, 0.173f, 0.180f, 1.0f};  // #2C2C2E
        static constexpr ImVec4 SurfaceOverlay   = {0.227f, 0.227f, 0.235f, 1.0f};  // #3A3A3C
        static constexpr ImVec4 TextPrimary      = {0.960f, 0.960f, 0.968f, 1.0f};  // #F5F5F7
        static constexpr ImVec4 TextSecondary    = {0.596f, 0.596f, 0.616f, 1.0f};  // #98989D
        static constexpr ImVec4 AccentBlue       = {0.161f, 0.592f, 1.000f, 1.0f};  // #2997FF
        static constexpr ImVec4 AccentGreen      = {0.188f, 0.820f, 0.345f, 1.0f};  // #30D158
        static constexpr ImVec4 AccentOrange     = {1.000f, 0.624f, 0.039f, 1.0f};  // #FF9F0A
        static constexpr ImVec4 AccentRed        = {1.000f, 0.271f, 0.227f, 1.0f};  // #FF453A
    };

    inline void ApplyAppleTheme() {
        ImGuiStyle& s = ImGui::GetStyle();

        // Geometry — generous rounding, Apple-like spacing
        s.WindowRounding    = 12.0f;
        s.FrameRounding     = 10.0f;
        s.GrabRounding      = 10.0f;
        s.TabRounding       = 8.0f;
        s.ScrollbarRounding = 10.0f;
        s.ChildRounding     = 10.0f;
        s.PopupRounding     = 10.0f;

        s.WindowPadding     = {20.0f, 20.0f};
        s.FramePadding      = {12.0f, 8.0f};
        s.ItemSpacing       = {10.0f, 8.0f};
        s.ScrollbarSize     = 10.0f;
        s.WindowBorderSize  = 0.0f;
        s.FrameBorderSize   = 0.0f;

        s.AntiAliasedLines  = true;
        s.AntiAliasedFill   = true;

        auto* c = s.Colors;
        c[ImGuiCol_WindowBg]          = ThemeColors::Surface;
        c[ImGuiCol_ChildBg]           = {0, 0, 0, 0};  // Transparent for Mica
        c[ImGuiCol_FrameBg]           = ThemeColors::SurfaceElevated;
        c[ImGuiCol_FrameBgHovered]    = ThemeColors::SurfaceOverlay;
        c[ImGuiCol_FrameBgActive]     = ThemeColors::SurfaceOverlay;
        c[ImGuiCol_TitleBg]           = ThemeColors::Surface;
        c[ImGuiCol_TitleBgActive]     = ThemeColors::Surface;
        c[ImGuiCol_Text]              = ThemeColors::TextPrimary;
        c[ImGuiCol_TextDisabled]      = ThemeColors::TextSecondary;
        c[ImGuiCol_Button]            = ThemeColors::AccentBlue;
        c[ImGuiCol_ButtonHovered]     = {0.20f, 0.63f, 1.0f, 1.0f};
        c[ImGuiCol_ButtonActive]      = {0.12f, 0.55f, 0.95f, 1.0f};
        c[ImGuiCol_Header]            = ThemeColors::SurfaceElevated;
        c[ImGuiCol_HeaderHovered]     = ThemeColors::SurfaceOverlay;
        c[ImGuiCol_HeaderActive]      = ThemeColors::SurfaceOverlay;
        c[ImGuiCol_Separator]         = {1.0f, 1.0f, 1.0f, 0.06f};
        c[ImGuiCol_ScrollbarBg]       = {0, 0, 0, 0};
        c[ImGuiCol_ScrollbarGrab]     = {1.0f, 1.0f, 1.0f, 0.15f};
        c[ImGuiCol_ScrollbarGrabHovered] = {1.0f, 1.0f, 1.0f, 0.25f};
        c[ImGuiCol_ScrollbarGrabActive]  = {1.0f, 1.0f, 1.0f, 0.35f};
        c[ImGuiCol_Tab]               = ThemeColors::SurfaceElevated;
        c[ImGuiCol_TabHovered]        = ThemeColors::AccentBlue;
        c[ImGuiCol_TabSelected]       = ThemeColors::AccentBlue;
        c[ImGuiCol_PopupBg]           = {0.12f, 0.12f, 0.13f, 0.95f};
        c[ImGuiCol_Border]            = {1.0f, 1.0f, 1.0f, 0.05f};
        c[ImGuiCol_CheckMark]         = ThemeColors::AccentBlue;
        c[ImGuiCol_SliderGrab]        = ThemeColors::AccentBlue;
        c[ImGuiCol_SliderGrabActive]  = {0.12f, 0.55f, 0.95f, 1.0f};
    }

} // namespace LocalNotebookLLM::UI
