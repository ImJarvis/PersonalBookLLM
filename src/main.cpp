// ╔══════════════════════════════════════════════════════════════════╗
// ║  LocalNotebookLLM — Main Application Entry Point                ║
// ║  Win32 + DirectX 11 + Dear ImGui + DWM Mica                   ║
// ╚══════════════════════════════════════════════════════════════════╝

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d11.h>
#include <tchar.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

#include <imgui.h>
#include <imgui_impl_win32.h>
#include <imgui_impl_dx11.h>

#include "UI/App.h"
#include "UI/Theme/AppleTheme.h"
#include "UI/Platform/WindowSetup.h"
#include "UI/Panels/ChatPanel.h"
#include "UI/Panels/LibraryPanel.h"
#include "UI/Panels/SettingsPanel.h"
#include "Core/Log.h"

#include <filesystem>
#include <string>

// Forward-declare the ImGui Win32 message handler
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(
    HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// ─── DirectX 11 globals ───
static ID3D11Device*           g_pd3dDevice        = nullptr;
static ID3D11DeviceContext*    g_pd3dDeviceContext  = nullptr;
static IDXGISwapChain*         g_pSwapChain        = nullptr;
static ID3D11RenderTargetView* g_mainRenderTarget   = nullptr;
static UINT                    g_ResizeWidth        = 0;
static UINT                    g_ResizeHeight       = 0;

// ─── Forward declarations ───
static bool  CreateDeviceD3D(HWND hWnd);
static void  CleanupDeviceD3D();
static void  CreateRenderTarget();
static void  CleanupRenderTarget();
static LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// ─── Application instance ───
static LocalNotebookLLM::UI::App g_app;

// ─── Drag & Drop support ───
static void HandleDroppedFiles(HDROP hDrop) {
    UINT count = DragQueryFileW(hDrop, 0xFFFFFFFF, nullptr, 0);
    LOG_INFO("Main", "Drag-and-drop: " + std::to_string(count) + " file(s) dropped");
    for (UINT i = 0; i < count; ++i) {
        wchar_t path[MAX_PATH] = {};
        DragQueryFileW(hDrop, i, path, MAX_PATH);
        // Convert wchar_t to string for logging
        char mbPath[MAX_PATH] = {};
        WideCharToMultiByte(CP_UTF8, 0, path, -1, mbPath, MAX_PATH, nullptr, nullptr);
        LOG_INFO("Main", "  Ingesting dropped file: " + std::string(mbPath));
        g_app.IngestDocument(std::filesystem::path(path));
    }
    DragFinish(hDrop);
}

// ═══════════════════════════════════════════════════════════════════
//  Main render function — renders the full application UI
// ═══════════════════════════════════════════════════════════════════
static void RenderUI() {
    using namespace LocalNotebookLLM::UI;
    auto& state = g_app.GetState();

    // Poll async operations
    g_app.Update();

    // ─── Full-window ImGui viewport ───
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

    ImGuiWindowFlags mainFlags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize   | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    ImGui::Begin("##MainWindow", nullptr, mainFlags);
    ImGui::PopStyleVar();

    // ─── Sidebar navigation (left column) ───
    float sidebarWidth = 220.0f;

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ThemeColors::SurfaceElevated);
    ImGui::BeginChild("Sidebar", ImVec2(sidebarWidth, 0), ImGuiChildFlags_None);
    {
        ImGui::Spacing();
        ImGui::Spacing();

        // App title
        ImGui::SetCursorPosX(20);
        ImGui::TextColored(ThemeColors::TextPrimary, "LocalNotebookLLM");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Spacing();

        // Navigation buttons
        auto NavButton = [&](const char* label, AppState::ActivePage page) {
            bool selected = (state.activePage == page);
            if (selected) {
                ImGui::PushStyleColor(ImGuiCol_Button, ThemeColors::AccentBlue);
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
            }
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                selected ? ImVec4(0.20f, 0.63f, 1.0f, 1.0f) : ThemeColors::SurfaceOverlay);

            if (ImGui::Button(label, ImVec2(sidebarWidth - 20, 40))) {
                state.activePage = page;
            }

            ImGui::PopStyleColor(2);
        };

        ImGui::SetCursorPosX(10);
        NavButton("  Chat", AppState::ActivePage::Chat);
        ImGui::SetCursorPosX(10);
        NavButton("  Library", AppState::ActivePage::Library);
        ImGui::SetCursorPosX(10);
        NavButton("  Settings", AppState::ActivePage::Settings);

        // ─── Status at bottom ───
        float bottomY = ImGui::GetContentRegionAvail().y - 110;
        if (bottomY > 0) ImGui::SetCursorPosY(bottomY);

        ImGui::Separator();
        ImGui::Spacing();

        // Document count
        ImGui::TextColored(ThemeColors::TextSecondary, "  %zu docs indexed",
                           state.documents.size());

        // Reasoner model status
        ImVec4 reasonerColor = state.reasonerLoaded
            ? ThemeColors::AccentGreen
            : ThemeColors::AccentOrange;
        ImGui::TextColored(reasonerColor, "  R: %s",
            state.reasonerLoaded ? state.reasonerName.c_str() : "No reasoner");

        // Worker model status
        ImVec4 workerColor = state.workerLoaded
            ? ThemeColors::AccentGreen
            : ThemeColors::AccentOrange;
        ImGui::TextColored(workerColor, "  W: %s",
            state.workerLoaded ? state.workerName.c_str() : "No worker");
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();

    // ─── Content area (right side) ───
    ImGui::SameLine();
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(24, 20));
    ImGui::BeginChild("Content", ImVec2(0, 0), ImGuiChildFlags_None);
    {
        switch (state.activePage) {
            case AppState::ActivePage::Chat:
                ChatPanel::Render(g_app);
                break;
            case AppState::ActivePage::Library:
                LibraryPanel::Render(g_app);
                break;
            case AppState::ActivePage::Settings:
                SettingsPanel::Render(g_app);
                break;
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();

    ImGui::End();
}

// ═══════════════════════════════════════════════════════════════════
//  WinMain — Application Entry Point
// ═══════════════════════════════════════════════════════════════════
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    // ─── Initialize Logger FIRST ─── 
    std::filesystem::path logDir = std::filesystem::current_path() / "data";
    LocalNotebookLLM::Core::Logger::Instance().Initialize(logDir);
    LOG_INFO("Main", "=== WinMain Entry ===");
    LOG_INFO("Main", "CWD: " + std::filesystem::current_path().string());

    // ─── Create window class ───
    WNDCLASSEXW wc = {};
    wc.cbSize        = sizeof(wc);
    wc.style         = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc   = WndProc;
    wc.hInstance     = hInstance;
    wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = L"LocalNotebookLLM";
    RegisterClassExW(&wc);
    LOG_INFO("Main", "Window class registered");

    // ─── Create window ───
    HWND hwnd = CreateWindowExW(
        WS_EX_ACCEPTFILES,  // Accept drag & drop
        wc.lpszClassName,
        L"LocalNotebookLLM \u2014 Offline Document Intelligence",
        WS_OVERLAPPEDWINDOW,
        100, 50, 1400, 900,
        nullptr, nullptr, hInstance, nullptr);
    LOG_INFO("Main", "Window created: " + std::to_string(reinterpret_cast<uintptr_t>(hwnd)));

    // ─── DirectX 11 init ───
    if (!CreateDeviceD3D(hwnd)) {
        LOG_ERROR("Main", "DirectX 11 device creation FAILED");
        CleanupDeviceD3D();
        UnregisterClassW(wc.lpszClassName, hInstance);
        return 1;
    }
    LOG_INFO("Main", "DirectX 11 device created successfully");

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    // ─── Enable Windows 11 Mica backdrop ───
    bool micaOk = LocalNotebookLLM::UI::EnableMicaBackdrop(hwnd);
    LOG_INFO("Main", std::string("Mica backdrop: ") + (micaOk ? "enabled" : "not available"));

    // ─── ImGui initialization ───
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    LOG_INFO("Main", "ImGui context created");

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr;  // Don't save imgui.ini

    // Load system font
    const char* fontPath = "C:\\Windows\\Fonts\\SegUIVar.ttf";
    if (GetFileAttributesA(fontPath) != INVALID_FILE_ATTRIBUTES) {
        io.Fonts->AddFontFromFileTTF(fontPath, 15.0f);
        LOG_INFO("Main", "System font loaded: SegUIVar.ttf");
    } else {
        LOG_WARN("Main", "System font not found, using ImGui default");
    }

    // Apply theme
    LocalNotebookLLM::UI::ApplyAppleTheme();
    LOG_INFO("Main", "Apple dark theme applied");

    // Init platform/renderer backends
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);
    LOG_INFO("Main", "ImGui backends initialized (Win32 + DX11)");

    // ─── Initialize application ───
    std::filesystem::path dataDir = std::filesystem::current_path() / "data";
    LOG_INFO("Main", "Initializing App with data dir: " + dataDir.string());
    g_app.Initialize(dataDir);
    LOG_INFO("Main", "App initialization complete — entering main loop");

    // ─── Main loop ───
    bool running = true;
    while (running) {
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT) running = false;
        }
        if (!running) break;

        // Handle resize
        if (g_ResizeWidth != 0 && g_ResizeHeight != 0) {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, g_ResizeWidth, g_ResizeHeight,
                                         DXGI_FORMAT_UNKNOWN, 0);
            g_ResizeWidth = g_ResizeHeight = 0;
            CreateRenderTarget();
        }

        // Start ImGui frame
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        // Render application UI
        RenderUI();

        // Present
        ImGui::Render();
        const float clearColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Transparent for Mica
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTarget, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTarget, clearColor);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        // VSync on
        g_pSwapChain->Present(1, 0);
    }

    // ─── Cleanup ───
    LOG_INFO("Main", "Main loop exited — cleaning up");
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    DestroyWindow(hwnd);
    UnregisterClassW(wc.lpszClassName, hInstance);

    LOG_INFO("Main", "Shutdown complete");
    LocalNotebookLLM::Core::Logger::Instance().Shutdown();

    return 0;
}

// ═══════════════════════════════════════════════════════════════════
//  DirectX 11 helpers
// ═══════════════════════════════════════════════════════════════════
static bool CreateDeviceD3D(HWND hWnd) {
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount        = 2;
    sd.BufferDesc.Width   = 0;
    sd.BufferDesc.Height  = 0;
    sd.BufferDesc.Format  = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator   = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags              = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage        = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow       = hWnd;
    sd.SampleDesc.Count   = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed           = TRUE;
    sd.SwapEffect         = DXGI_SWAP_EFFECT_DISCARD;

    UINT createFlags = 0;
#ifdef _DEBUG
    createFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0
    };

    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        createFlags, featureLevels, 2,
        D3D11_SDK_VERSION, &sd,
        &g_pSwapChain, &g_pd3dDevice,
        &featureLevel, &g_pd3dDeviceContext);

    if (FAILED(hr)) return false;

    CreateRenderTarget();
    return true;
}

static void CleanupDeviceD3D() {
    CleanupRenderTarget();
    if (g_pSwapChain)       { g_pSwapChain->Release();       g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice)       { g_pd3dDevice->Release();       g_pd3dDevice = nullptr; }
}

static void CreateRenderTarget() {
    ID3D11Texture2D* backBuffer = nullptr;
    g_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D),
                             reinterpret_cast<void**>(&backBuffer));
    g_pd3dDevice->CreateRenderTargetView(backBuffer, nullptr, &g_mainRenderTarget);
    backBuffer->Release();
}

static void CleanupRenderTarget() {
    if (g_mainRenderTarget) { g_mainRenderTarget->Release(); g_mainRenderTarget = nullptr; }
}

static LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg) {
        case WM_SIZE:
            if (wParam == SIZE_MINIMIZED) return 0;
            g_ResizeWidth  = LOWORD(lParam);
            g_ResizeHeight = HIWORD(lParam);
            return 0;
        case WM_DROPFILES:
            HandleDroppedFiles(reinterpret_cast<HDROP>(wParam));
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hWnd, msg, wParam, lParam);
}
