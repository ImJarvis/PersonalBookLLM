# ─── Automatic GGUF Model Download ───
# Downloads the tri-model pipeline:
#   Worker:    Qwen 2.5 1.5B Instruct Q4_K_M  (~0.9 GB) — indexing / structured extraction
#   Reasoner:  Phi-4-mini Instruct Q4_K_M     (~2.3 GB) — Q&A generation & reasoning
#   Embedding: Nomic Embed Text v1.5 F16      (~0.9 GB) — Semantic vector search
#
# If models already exist on disk, download is skipped.

set(MODELS_DIR "${CMAKE_SOURCE_DIR}/models")
file(MAKE_DIRECTORY "${MODELS_DIR}")

# ─── 1. Worker Model: Qwen 2.5 1.5B ───
set(WORKER_MODEL_URL "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
set(WORKER_MODEL_PATH "${MODELS_DIR}/${WORKER_MODEL_FILENAME}")

if(NOT EXISTS "${WORKER_MODEL_PATH}")
    message(STATUS "")
    message(STATUS "╔══════════════════════════════════════════════════════════════╗")
    message(STATUS "║  Downloading Worker Model: Qwen 2.5 1.5B Q4_K_M            ║")
    message(STATUS "║  Size: ~0.9 GB  |  Role: Indexing & Structured Extraction   ║")
    message(STATUS "╚══════════════════════════════════════════════════════════════╝")
    file(DOWNLOAD "${WORKER_MODEL_URL}" "${WORKER_MODEL_PATH}" SHOW_PROGRESS STATUS DL_STATUS TLS_VERIFY ON)
    list(GET DL_STATUS 0 DL_CODE)
    if(NOT DL_CODE EQUAL 0)
        message(WARNING "Failed to download Worker. Download manually from: ${WORKER_MODEL_URL}")
        file(REMOVE "${WORKER_MODEL_PATH}")
    endif()
else()
    message(STATUS "Worker model found: ${WORKER_MODEL_FILENAME}")
endif()

# ─── 2. Reasoner Model: Phi-4-mini ───
set(REASONER_MODEL_URL "https://huggingface.co/bartowski/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q4_K_M.gguf")
set(REASONER_MODEL_PATH "${MODELS_DIR}/${REASONER_MODEL_FILENAME}")

if(NOT EXISTS "${REASONER_MODEL_PATH}")
    message(STATUS "")
    message(STATUS "╔══════════════════════════════════════════════════════════════╗")
    message(STATUS "║  Downloading Reasoner Model: Phi-4-mini Q4_K_M             ║")
    message(STATUS "║  Size: ~2.3 GB  |  Role: Q&A Generation & Reasoning         ║")
    message(STATUS "╚══════════════════════════════════════════════════════════════╝")
    file(DOWNLOAD "${REASONER_MODEL_URL}" "${REASONER_MODEL_PATH}" SHOW_PROGRESS STATUS DL_STATUS TLS_VERIFY ON)
    list(GET DL_STATUS 0 DL_CODE)
    if(NOT DL_CODE EQUAL 0)
        message(WARNING "Failed to download Reasoner. Download manually from: ${REASONER_MODEL_URL}")
        file(REMOVE "${REASONER_MODEL_PATH}")
    endif()
else()
    message(STATUS "Reasoner model found: ${REASONER_MODEL_FILENAME}")
endif()

# ─── 3. Embedding Model: Nomic v1.5 ───
set(EMBEDDING_MODEL_URL "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf")
set(EMBEDDING_MODEL_PATH "${MODELS_DIR}/${EMBEDDING_MODEL_FILENAME}")

if(NOT EXISTS "${EMBEDDING_MODEL_PATH}")
    message(STATUS "")
    message(STATUS "╔══════════════════════════════════════════════════════════════╗")
    message(STATUS "║  Downloading Embedding Model: Nomic Embed v1.5 F16         ║")
    message(STATUS "║  Size: ~0.9 GB  |  Role: Semantic Vector Search             ║")
    message(STATUS "╚══════════════════════════════════════════════════════════════╝")
    file(DOWNLOAD "${EMBEDDING_MODEL_URL}" "${EMBEDDING_MODEL_PATH}" SHOW_PROGRESS STATUS DL_STATUS TLS_VERIFY ON)
    list(GET DL_STATUS 0 DL_CODE)
    if(NOT DL_CODE EQUAL 0)
        message(WARNING "Failed to download Embedding model. Download manually from: ${EMBEDDING_MODEL_URL}")
        file(REMOVE "${EMBEDDING_MODEL_PATH}")
    endif()
else()
    message(STATUS "Embedding model found: ${EMBEDDING_MODEL_FILENAME}")
endif()

# ─── Export configuration ───
message(STATUS "")
message(STATUS "Model Configuration Complete:")
message(STATUS "  Worker:    ${WORKER_MODEL_FILENAME}")
message(STATUS "  Reasoner:  ${REASONER_MODEL_FILENAME}")
message(STATUS "  Embedding: ${EMBEDDING_MODEL_FILENAME}")
message(STATUS "  Dir:       ${MODELS_DIR}")
message(STATUS "")
