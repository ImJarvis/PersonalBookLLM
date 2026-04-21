#include "Infrastructure/LLM/GPUDetector.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#pragma comment(lib, "dxgi.lib")
using Microsoft::WRL::ComPtr;
#endif

#include <algorithm>

namespace LocalNotebookLLM::Infrastructure {

    Core::GPUInfo GPUDetector::Detect() {
        Core::GPUInfo info;

#ifdef _WIN32
        ComPtr<IDXGIFactory1> factory;
        HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1),
                                         reinterpret_cast<void**>(factory.GetAddressOf()));
        if (FAILED(hr)) return info;

        ComPtr<IDXGIAdapter1> adapter;
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc;
            adapter->GetDesc1(&desc);

            // Skip software adapters (Microsoft Basic Render Driver)
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
                adapter.Reset();
                continue;
            }

            // DedicatedVideoMemory > 0 = dedicated GPU
            if (desc.DedicatedVideoMemory > 0) {
                info.hasDedicatedGPU = true;

                // Convert wide string to narrow
                char name[256] = {};
                wcstombs(name, desc.Description, sizeof(name) - 1);
                info.gpuName = name;

                info.totalVramMB = desc.DedicatedVideoMemory / (1024 * 1024);

                // Query VRAM usage via IDXGIAdapter3
                ComPtr<IDXGIAdapter3> adapter3;
                hr = adapter->QueryInterface(__uuidof(IDXGIAdapter3),
                                              reinterpret_cast<void**>(adapter3.GetAddressOf()));
                if (SUCCEEDED(hr) && adapter3) {
                    DXGI_QUERY_VIDEO_MEMORY_INFO memInfo = {};
                    hr = adapter3->QueryVideoMemoryInfo(
                            0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memInfo);
                    if (SUCCEEDED(hr)) {
                        size_t usedMB = static_cast<size_t>(memInfo.CurrentUsage / (1024 * 1024));
                        info.freeVramMB = info.totalVramMB > usedMB
                                        ? info.totalVramMB - usedMB : 0;
                    } else {
                        info.freeVramMB = info.totalVramMB * 80 / 100;
                    }
                } else {
                    info.freeVramMB = info.totalVramMB * 80 / 100;
                }

                break;  // Use the first dedicated GPU found
            }
            adapter.Reset();
        }
#endif

        return info;
    }

    int GPUDetector::RecommendGPULayers(size_t modelSizeMB, size_t freeVramMB) {
        if (freeVramMB == 0 || modelSizeMB == 0) return 0;

        // Reserve 500 MB for KV cache + OS overhead
        const size_t reserveMB = 500;
        if (freeVramMB <= reserveMB) return 0;

        size_t usableVram = freeVramMB - reserveMB;

        // If we can fit the whole model, offload everything (-1)
        if (usableVram >= modelSizeMB) return -1;

        // Partial offload: estimate layers proportionally (typical 32-layer model)
        const int estimatedTotalLayers = 32;
        double fraction = static_cast<double>(usableVram) / static_cast<double>(modelSizeMB);
        int layers = static_cast<int>(fraction * estimatedTotalLayers);

        return (std::max)(0, layers);
    }

} // namespace LocalNotebookLLM::Infrastructure
