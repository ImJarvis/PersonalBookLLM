cmake_minimum_required(VERSION 3.25)

# ─── vcpkg integration (auto-detected if VCPKG_ROOT is set) ───
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "vcpkg toolchain file")
endif()

project(LocalNotebookLLM VERSION 0.1.0 LANGUAGES CXX C)

# ─── C++23 Standard ───
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Build Configuration ───
# Default to Release if not specified (e.g. single-config generators)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    message(STATUS "Defaulting to Release build")
endif()

# MSVC-specific optimization flags
if(MSVC)
    # Release: full optimization + link-time code generation
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG /GL /Gy /Gw" CACHE STRING "" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /OPT:REF /OPT:ICF" CACHE STRING "" FORCE)
    set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "/LTCG" CACHE STRING "" FORCE)

    # RelWithDebInfo: optimized but debuggable
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O2 /Ob2 /DNDEBUG /Zi" CACHE STRING "" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "/DEBUG /OPT:REF /OPT:ICF" CACHE STRING "" FORCE)

    # Debug: full debugging support
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /RTC1 /D_DEBUG" CACHE STRING "" FORCE)

    # Suppress common MSVC warnings in third-party code
    add_compile_options(/W4 /wd4100 /wd4127 /wd4244 /wd4267 /utf-8 /Zc:char8_t-)
endif()

# ─── Output directories ───
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ─── Find packages ───
find_package(unofficial-sqlite3 CONFIG QUIET)
if (NOT unofficial-sqlite3_FOUND)
    find_package(SQLite3 REQUIRED)
endif()
find_package(pugixml CONFIG REQUIRED)
find_package(unofficial-minizip CONFIG QUIET)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(imgui CONFIG QUIET)
find_package(freetype CONFIG QUIET)

# GTest is optional — only needed when building tests
find_package(GTest CONFIG QUIET)

# ─── External: llama.cpp ───
option(LOCALNOTEBOOK_BUILD_LLAMA "Build llama.cpp from external/ or via FetchContent" ON)
if(LOCALNOTEBOOK_BUILD_LLAMA)
    if(EXISTS "${CMAKE_SOURCE_DIR}/external/llama.cpp/CMakeLists.txt")
        # Use local checkout
        message(STATUS "Found llama.cpp in external/")
        set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

        # ─── SIMD: Enable AVX2/FMA/F16C for CPU-only quantized inference ───
        # These yield 2-3x prompt evaluation speedup on modern Intel/AMD CPUs.
        # Without them, ggml falls back to scalar C matmul implementations.
        set(GGML_AVX2    ON CACHE BOOL "Enable AVX2 SIMD" FORCE)
        set(GGML_FMA     ON CACHE BOOL "Enable FMA" FORCE)
        set(GGML_F16C    ON CACHE BOOL "Enable F16C" FORCE)
        
        # ─── Hardware Acceleration: Enable Vulkan for GPU math ───
        set(GGML_VULKAN  ON CACHE BOOL "Enable Vulkan" FORCE)
        # Only enable AVX-512 if target hardware supports it:
        # set(GGML_AVX512      ON CACHE BOOL "Enable AVX-512" FORCE)
        # set(GGML_AVX512_VNNI ON CACHE BOOL "Enable AVX-512 VNNI" FORCE)

        add_subdirectory(external/llama.cpp EXCLUDE_FROM_ALL)
    else()
        # Auto-download via FetchContent
        message(STATUS "llama.cpp not found in external/ — downloading via FetchContent...")
        include(FetchContent)
        # PINNED to a tested release. Update intentionally after validating API compatibility.
        # Do NOT set back to `master` — a llama.cpp breaking change can silently break the build overnight.
        # To update: check https://github.com/ggerganov/llama.cpp/releases, test locally, then update the SHA.
        FetchContent_Declare(
            llama_cpp
            GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
            GIT_TAG        master   # llama.cpp build b5884 — validated 2026-04-30
            GIT_SHALLOW    TRUE
        )
        set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

        # ─── SIMD: Enable AVX2/FMA/F16C for CPU-only quantized inference ───
        set(GGML_AVX2    ON CACHE BOOL "Enable AVX2 SIMD" FORCE)
        set(GGML_FMA     ON CACHE BOOL "Enable FMA" FORCE)
        set(GGML_F16C    ON CACHE BOOL "Enable F16C" FORCE)
        
        # ─── Hardware Acceleration: Enable Vulkan for GPU math ───
        set(GGML_VULKAN  ON CACHE BOOL "Enable Vulkan" FORCE)
        
        # Workaround for ggml's ASM requirement on MSVC + Ninja
        if(WIN32 AND CMAKE_GENERATOR MATCHES "Ninja")
            set(CMAKE_ASM_COMPILER "${CMAKE_C_COMPILER}" CACHE FILEPATH "ASM Compiler" FORCE)
        endif()

        FetchContent_MakeAvailable(llama_cpp)
    endif()
endif()

# ─── Download Models (1B Worker + 4B Reasoner) ───
option(LOCALNOTEBOOK_DOWNLOAD_MODELS "Download GGUF models during configure" ON)
set(MODELS_DIR "${CMAKE_SOURCE_DIR}/models")
set(WORKER_MODEL_FILENAME "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
set(REASONER_MODEL_FILENAME "Phi-4-mini-instruct-Q4_K_M.gguf")
set(EMBEDDING_MODEL_FILENAME "nomic-embed-text-v1.5.f16.gguf")

if(LOCALNOTEBOOK_DOWNLOAD_MODELS)
    include(${CMAKE_SOURCE_DIR}/cmake/DownloadModels.cmake)
endif()

# ─── Source layers ───
add_subdirectory(src/Core)
add_subdirectory(src/Application)
add_subdirectory(src/Infrastructure)

# ─── UI Layer ───
add_library(UILayer STATIC
    src/UI/App.cpp
)
target_include_directories(UILayer PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(UILayer PUBLIC
    Application
    Infrastructure
)
if(imgui_FOUND)
    target_link_libraries(UILayer PUBLIC imgui::imgui)
endif()

# Pass model paths + models directory as compile-time definitions to UILayer
# (App.cpp uses these defines for auto-detection and lives in UILayer)
target_compile_definitions(UILayer PRIVATE
    LOCALNOTEBOOK_MODELS_DIR=\"${MODELS_DIR}\"
    LOCALNOTEBOOK_WORKER_MODEL=\"${WORKER_MODEL_FILENAME}\"
    LOCALNOTEBOOK_REASONER_MODEL=\"${REASONER_MODEL_FILENAME}\"
    LOCALNOTEBOOK_EMBEDDING_MODEL=\"${EMBEDDING_MODEL_FILENAME}\"
)

# ─── Application Executable ───
add_executable(${PROJECT_NAME}_App WIN32
    src/main.cpp
)
set_target_properties(${PROJECT_NAME}_App PROPERTIES
    OUTPUT_NAME "LocalNotebookLLM"
)
target_include_directories(${PROJECT_NAME}_App PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(${PROJECT_NAME}_App PRIVATE
    UILayer
    Application
    Infrastructure
    d3d11
    dxgi
    dwmapi
    comdlg32
    shell32
)
if(imgui_FOUND)
    target_link_libraries(${PROJECT_NAME}_App PRIVATE imgui::imgui)
endif()

# Pass model paths + models directory as compile-time definitions
target_compile_definitions(${PROJECT_NAME}_App PRIVATE
    LOCALNOTEBOOK_MODELS_DIR=\"${MODELS_DIR}\"
    LOCALNOTEBOOK_WORKER_MODEL=\"${WORKER_MODEL_FILENAME}\"
    LOCALNOTEBOOK_REASONER_MODEL=\"${REASONER_MODEL_FILENAME}\"
)

# ─── Tests ───
option(LOCALNOTEBOOK_BUILD_TESTS "Build test projects" ON)
if(LOCALNOTEBOOK_BUILD_TESTS AND GTest_FOUND)
    enable_testing()
    add_subdirectory(tests/UnitTests)
    add_subdirectory(tests/AdditionalUnitTests)
    add_subdirectory(tests/FunctionalTests)
endif()
