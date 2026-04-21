#pragma once
#include <dwmapi.h>
#pragma comment(lib, "dwmapi.lib")

#ifndef DWMWA_SYSTEMBACKDROP_TYPE
#define DWMWA_SYSTEMBACKDROP_TYPE 38
#endif

namespace LocalNotebookLLM::UI {

    enum DWM_SYSTEMBACKDROP_TYPE_LOCAL {
        DWMSBT_AUTO            = 0,
        DWMSBT_NONE            = 1,
        DWMSBT_MAINWINDOW      = 2,  // Mica
        DWMSBT_TRANSIENTWINDOW = 3,  // Acrylic
        DWMSBT_TABBEDWINDOW    = 4   // Tabbed Mica
    };

    inline bool EnableMicaBackdrop(HWND hwnd) {
        // Enable immersive dark mode
        BOOL darkMode = TRUE;
        DwmSetWindowAttribute(hwnd, 20 /*DWMWA_USE_IMMERSIVE_DARK_MODE*/,
                              &darkMode, sizeof(darkMode));

        // Enable Mica
        auto backdrop = DWMSBT_MAINWINDOW;
        HRESULT hr = DwmSetWindowAttribute(
            hwnd, DWMWA_SYSTEMBACKDROP_TYPE,
            &backdrop, sizeof(backdrop));

        return SUCCEEDED(hr);
    }

} // namespace LocalNotebookLLM::UI
