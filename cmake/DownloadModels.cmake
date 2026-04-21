# ─── Automatic GGUF Model Download ───
# Downloads the dual-model pipeline:
#   Worker:   Qwen 2.5 1.5B Instruct Q4_K_M   (~0.9 GB) — indexing / structured extraction
#   Reasoner: Gemma 4 E2B Instruct Q4_K_M      (~1.3 GB) — Q&A generation & reasoning
#
# If models already exist on disk, download is skipped.

set(MODELS_DIR "${CMAKE_SOURCE_DIR}/models")

# ─── Worker Model: Qwen 2.5 1.5B ───
set(WORKER_MODEL_FILENAME "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
set(WORKER_MODEL_URL "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
set(WORKER_MODEL_PATH "${MODELS_DIR}/${WORKER_MODEL_FILENAME}")

if(NOT EXISTS "${WORKER_MODEL_PATH}")
    message(STATUS "")
    message(STATUS "╔══════════════════════════════════════════════════════════════╗")
    message(STATUS "║  Downloading Worker Model: Qwen 2.5 1.5B Q4_K_M            ║")
    message(STATUS "║  Size: ~0.9 GB  |  Role: Indexing & Structured Extraction   ║")
    message(STATUS "╚══════════════════════════════════════════════════════════════╝")
    file(MAKE_DIRECTORY "${MODELS_DIR}")
    file(DOWNLOAD
        "${WORKER_MODEL_URL}"
        "${WORKER_MODEL_PATH}"
        SHOW_PROGRESS
        STATUS WORKER_DL_STATUS
        TLS_VERIFY ON
    )
    list(GET WORKER_DL_STATUS 0 WORKER_DL_CODE)
    if(NOT WORKER_DL_CODE EQUAL 0)
        list(GET WORKER_DL_STATUS 1 WORKER_DL_MSG)
        message(WARNING "Worker model download failed: ${WORKER_DL_MSG}")
        message(WARNING "Manually download from: ${WORKER_MODEL_URL}")
        message(WARNING "Place at: ${WORKER_MODEL_PATH}")
        file(REMOVE "${WORKER_MODEL_PATH}")
    else()
        message(STATUS "Worker model downloaded: ${WORKER_MODEL_FILENAME}")
    endif()
else()
    message(STATUS "Worker model found: ${WORKER_MODEL_FILENAME}")
endif()

# ─── Reasoner Model: Gemma 4 E2B (fallback to Gemma 3 4B if present) ───
# Auto-detect: prefer gemma-4-e2b, fallback to gemma-3-4b if already on disk
set(REASONER_MODEL_FILENAME "")

# Check for existing models in priority order
foreach(_candidate
    "gemma-3-4b-it-Q4_K_M.gguf"
    "gemma-4-e2b-it-Q4_K_M.gguf"
    
)
    if(EXISTS "${MODELS_DIR}/${_candidate}")
        set(REASONER_MODEL_FILENAME "${_candidate}")
        message(STATUS "Reasoner model found: ${_candidate}")
        break()
    endif()
endforeach()

# If no existing model found, download Gemma 4 E2B
if(REASONER_MODEL_FILENAME STREQUAL "")
    set(REASONER_MODEL_FILENAME "gemma-4-e2b-it-Q4_K_M.gguf")
    set(REASONER_MODEL_URL "https://huggingface.co/bartowski/google_gemma-4-e2b-it-GGUF/resolve/main/google_gemma-4-e2b-it-Q4_K_M.gguf")
    set(REASONER_MODEL_PATH "${MODELS_DIR}/${REASONER_MODEL_FILENAME}")

    message(STATUS "")
    message(STATUS "╔══════════════════════════════════════════════════════════════╗")
    message(STATUS "║  Downloading Reasoner Model: Gemma 4 E2B Q4_K_M            ║")
    message(STATUS "║  Size: ~1.3 GB  |  Role: Q&A Generation & Reasoning         ║")
    message(STATUS "╚══════════════════════════════════════════════════════════════╝")
    file(DOWNLOAD
        "${REASONER_MODEL_URL}"
        "${REASONER_MODEL_PATH}"
        SHOW_PROGRESS
        STATUS REASONER_DL_STATUS
        TLS_VERIFY ON
    )
    list(GET REASONER_DL_STATUS 0 REASONER_DL_CODE)
    if(NOT REASONER_DL_CODE EQUAL 0)
        list(GET REASONER_DL_STATUS 1 REASONER_DL_MSG)
        message(WARNING "Reasoner model download failed: ${REASONER_DL_MSG}")
        message(WARNING "Manually download from: ${REASONER_MODEL_URL}")
        file(REMOVE "${REASONER_MODEL_PATH}")
    else()
        message(STATUS "Reasoner model downloaded: ${REASONER_MODEL_FILENAME}")
    endif()
endif()

# ─── Export model configuration ───
set(LOCALNOTEBOOK_WORKER_MODEL_PATH "${MODELS_DIR}/${WORKER_MODEL_FILENAME}" CACHE STRING "" FORCE)
set(LOCALNOTEBOOK_REASONER_MODEL_PATH "${MODELS_DIR}/${REASONER_MODEL_FILENAME}" CACHE STRING "" FORCE)
set(LOCALNOTEBOOK_WORKER_MODEL_NAME "${WORKER_MODEL_FILENAME}" CACHE STRING "" FORCE)
set(LOCALNOTEBOOK_REASONER_MODEL_NAME "${REASONER_MODEL_FILENAME}" CACHE STRING "" FORCE)
set(LOCALNOTEBOOK_MODELS_DIR "${MODELS_DIR}" CACHE STRING "" FORCE)

message(STATUS "")
message(STATUS "Model Configuration:")
message(STATUS "  Worker:   ${WORKER_MODEL_FILENAME}")
message(STATUS "  Reasoner: ${REASONER_MODEL_FILENAME}")
message(STATUS "  Dir:      ${MODELS_DIR}")
message(STATUS "")
