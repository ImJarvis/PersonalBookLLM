#include "Infrastructure/LLM/LlamaCppProvider.h"
#include "Infrastructure/LLM/GPUDetector.h"
#include "Core/Log.h"
#include <chrono>
#include <functional>
#include <thread>

#ifdef HAS_LLAMA_CPP
#include "Infrastructure/LLM/LlamaInternalUtils.h"  // H5: shared polyfill (llama_batch_add)
#endif

namespace LocalNotebookLLM::Infrastructure {

    struct LlamaCppProvider::Impl {
#ifdef HAS_LLAMA_CPP
        llama_model*       model   = nullptr;
        llama_context*     ctx     = nullptr;
        const llama_vocab* vocab   = nullptr;
        llama_sampler*     sampler = nullptr; // Cached sampler — rebuilt only when params change
#endif
        Core::ModelParams params;
        Core::ModelStatus status;
        bool loaded = false;

        // ─── Gold Standard Two-Stage KV Cache Layout ───────────────────────────
        //
        //  The llama.cpp context window is treated as two pinned regions:
        //
        //  Seq 1 (system prefix — pinned forever):
        //    Tokens [0 .. systemPrefixLen)
        //    Contains: CRITICAL RULES + intent persona (~80 tokens)
        //    Evaluated exactly once on the first GenerateStreaming() call.
        //    Never evicted until model is unloaded.
        //
        //  Seq 0 (document context — pinned per document hash):
        //    Tokens [systemPrefixLen .. systemPrefixLen + contextLen)
        //    Contains: retrieved passages block (~1500-2500 tokens)
        //    Evicted and re-evaluated only when contextHash changes.
        //    On follow-up questions to the same document: 0 tokens re-evaluated here.
        //
        //  Variable tail (question + answer stub):
        //    Tokens [systemPrefixLen + contextLen .. N)
        //    Always freshly evaluated (~15-25 tokens).
        //
        //  Net result: 2nd+ question to same document skips ~2100/2200 tokens.
        // ────────────────────────────────────────────────────────────────────────

        // Stage 1 — system prefix
        std::vector<llama_token> systemPrefixTokens; // Tokenized CRITICAL RULES block
        int  systemPrefixLen  = 0;                   // KV positions [0, systemPrefixLen) pinned
        bool systemPrefixWarm = false;               // True after first evaluation

        // Stage 2 — document context
        std::vector<llama_token> contextTokens;      // Tokenized context block
        int    contextLen     = 0;                   // KV positions [systemPrefixLen, +contextLen)
        bool   contextWarm    = false;               // True when contextHash is valid
        size_t contextHash    = 0;                   // FNV-1a hash of formattedContext string

        // Legacy full-prompt cache (used as fallback when stages not applicable)
        std::vector<llama_token> cachedTokens;
        bool   kvCacheWarm    = false;
    };

    // ── FNV-1a 64-bit hash — fast, non-crypto, sufficient for cache keying ──
    static size_t HashString(const std::string& s) {
        size_t hash = 14695981039346656037ULL;
        for (unsigned char c : s) {
            hash ^= c;
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    // Global flag to ensure llama_backend_init is only called exactly once
    static std::once_flag g_llamaInitFlag;

    LlamaCppProvider::LlamaCppProvider()
        : m_impl(std::make_unique<Impl>()) {
#ifdef HAS_LLAMA_CPP
        std::call_once(g_llamaInitFlag, []() {
            llama_backend_init();
            LOG_INFO("LlamaCpp", "Backend initialized natively (once globally)");
            LOG_INFO("LlamaCpp", std::string("Global System Optimizations: ") + llama_print_system_info());
        });
#else
        LOG_WARN("LlamaCpp", "Compiled WITHOUT llama.cpp (HAS_LLAMA_CPP not defined)");
#endif
    }

    LlamaCppProvider::~LlamaCppProvider() {
        LOG_DEBUG("LlamaCpp", "Destructor — unloading model");
        UnloadModel();
#ifdef HAS_LLAMA_CPP
        // DO NOT call llama_backend_free() here.
        // It's global state shared by Worker, Reasoner, and Embedder.
        // Calling free here crashes the other active models.
#endif
    }

    std::expected<void, Core::LLMError>
    LlamaCppProvider::LoadModel(const std::filesystem::path& modelPath,
                                const Core::ModelParams& params) {
#ifdef HAS_LLAMA_CPP
        LOG_INFO("LlamaCpp", "LoadModel() path: " + modelPath.string());
        std::lock_guard lock(m_mutex);

        // Unload any previous model
        if (m_impl->loaded) {
            LOG_INFO("LlamaCpp", "Unloading previous model before loading new one");
            if (m_impl->ctx)   { llama_free(m_impl->ctx);          m_impl->ctx = nullptr; }
            if (m_impl->model) { llama_model_free(m_impl->model);  m_impl->model = nullptr; }
            m_impl->loaded = false;
        }

        if (!std::filesystem::exists(modelPath)) {
            LOG_ERROR("LlamaCpp", "Model file NOT FOUND: " + modelPath.string());
            return std::unexpected(Core::LLMError{
                Core::LLMError::Code::ModelNotFound,
                "Model file not found: " + modelPath.string()
            });
        }

        size_t fileSizeMB = std::filesystem::file_size(modelPath) / (1024 * 1024);
        LOG_INFO("LlamaCpp", "Model file size: " + std::to_string(fileSizeMB) + " MB");

        // ─── Model parameters ───
        auto mparams = llama_model_default_params();

        // GPU layer detection
        int gpuLayers = params.gpuLayers;
        if (gpuLayers == -1) {
            auto gpu = GPUDetector::Detect();
            if (gpu.hasDedicatedGPU) {
                gpuLayers = GPUDetector::RecommendGPULayers(fileSizeMB, gpu.freeVramMB);
                LOG_INFO("LlamaCpp", "GPU detected: " + gpu.gpuName +
                    " (VRAM: " + std::to_string(gpu.totalVramMB) + " MB, free: " +
                    std::to_string(gpu.freeVramMB) + " MB) → " +
                    std::to_string(gpuLayers) + " GPU layers");
            } else {
                gpuLayers = 0;
                LOG_INFO("LlamaCpp", "No dedicated GPU — CPU-only mode");
            }
        }
        mparams.n_gpu_layers = gpuLayers;

        // Load model
        LOG_INFO("LlamaCpp", "Loading model file (this may take a moment)...");
        auto loadStart = std::chrono::steady_clock::now();
        m_impl->model = llama_model_load_from_file(modelPath.string().c_str(), mparams);
        auto loadEnd = std::chrono::steady_clock::now();
        double loadMs = std::chrono::duration<double, std::milli>(loadEnd - loadStart).count();

        if (!m_impl->model) {
            LOG_ERROR("LlamaCpp", "llama_model_load_from_file FAILED");
            return std::unexpected(Core::LLMError{
                Core::LLMError::Code::LoadFailed,
                "Failed to load model: " + modelPath.string()
            });
        }
        LOG_INFO("LlamaCpp", "Model loaded in " + std::to_string(loadMs) + " ms");

        // Get vocab
        m_impl->vocab = llama_model_get_vocab(m_impl->model);

        // ─── Context parameters ───
        auto cparams = llama_context_default_params();
        cparams.n_ctx     = params.contextWindowSize;
        cparams.n_batch   = 512;  // llama.cpp default. Cancellation is checked between
                                   // each 512-token chunk — sufficient granularity for CPU.
        cparams.n_threads = params.threadCount > 0
                           ? params.threadCount
                           : static_cast<int>(std::thread::hardware_concurrency());
        cparams.n_threads_batch = cparams.n_threads;

        LOG_INFO("LlamaCpp", "Context params: n_ctx=" + std::to_string(cparams.n_ctx) +
            " n_threads=" + std::to_string(cparams.n_threads) +
            " flash_attn=" + std::string(params.useFlashAttention ? "ON" : "OFF"));

        // KV cache quantization
        auto toGgmlType = [](Core::ModelParams::KVCacheType t) -> ggml_type {
            switch (t) {
                case Core::ModelParams::KVCacheType::Q8_0: return GGML_TYPE_Q8_0;
                case Core::ModelParams::KVCacheType::Q4_0: return GGML_TYPE_Q4_0;
                default: return GGML_TYPE_F16;
            }
        };
        cparams.type_k     = toGgmlType(params.kvCacheTypeK);
        cparams.type_v     = toGgmlType(params.kvCacheTypeV);
        cparams.rope_freq_scale = 1.0f;
        //cparams.flash_attn = params.useFlashAttention;

        m_impl->ctx = llama_init_from_model(m_impl->model, cparams);
        if (!m_impl->ctx) {
            LOG_ERROR("LlamaCpp", "llama_init_from_model FAILED — likely OOM");
            llama_model_free(m_impl->model);
            m_impl->model = nullptr;
            return std::unexpected(Core::LLMError{
                Core::LLMError::Code::OutOfMemory,
                "Failed to create context — likely insufficient memory"
            });
        }

        // Store params & status
        m_impl->params = params;
        m_impl->loaded = true;

        m_impl->status.isLoaded      = true;
        m_impl->status.modelPath     = modelPath.string();
        m_impl->status.modelName     = modelPath.filename().string();
        m_impl->status.contextWindow = params.contextWindowSize;
        m_impl->status.gpuLayers     = gpuLayers;
        m_impl->status.device        = gpuLayers > 0 ? "GPU" : "CPU";

        // Reset all KV cache state on new model load
        m_impl->cachedTokens.clear();
        m_impl->kvCacheWarm       = false;
        m_impl->systemPrefixTokens.clear();
        m_impl->systemPrefixLen   = 0;
        m_impl->systemPrefixWarm  = false;
        m_impl->contextTokens.clear();
        m_impl->contextLen        = 0;
        m_impl->contextWarm       = false;
        m_impl->contextHash       = 0;

        // Rebuild cached sampler for the new model/params
        if (m_impl->sampler) {
            llama_sampler_free(m_impl->sampler);
            m_impl->sampler = nullptr;
        }
        m_impl->sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        // penalty_last_n=256: covers ~1 full paragraph lookback window.
        // Was 64 — too narrow for 512+ token outputs; model couldn't see its own
        // earlier text and repeated entire sentences/paragraphs.
        llama_sampler_chain_add(m_impl->sampler, llama_sampler_init_penalties(256, 1.1f, 0.0f, 0.0f));
        llama_sampler_chain_add(m_impl->sampler, llama_sampler_init_top_k(params.topK));
        llama_sampler_chain_add(m_impl->sampler, llama_sampler_init_top_p(params.topP, 1));
        llama_sampler_chain_add(m_impl->sampler, llama_sampler_init_temp(params.temperature));
        llama_sampler_chain_add(m_impl->sampler, llama_sampler_init_dist(0));
        LOG_INFO("LlamaCpp", "Sampler chain built and cached");

        LOG_INFO("LlamaCpp", "=== Model READY: " + m_impl->status.modelName +
            " | device=" + m_impl->status.device +
            " | gpuLayers=" + std::to_string(gpuLayers) +
            " | ctx=" + std::to_string(params.contextWindowSize) + " ===");

        LOG_INFO("LlamaCpp", std::string("System Info: ") + llama_print_system_info());

        return {};
#else
        LOG_ERROR("LlamaCpp", "LoadModel() REFUSED — HAS_LLAMA_CPP not defined. "
            "Rebuild with llama.cpp (FetchContent or external/llama.cpp).");
        return std::unexpected(Core::LLMError{
            Core::LLMError::Code::LoadFailed,
            "llama.cpp not compiled in (HAS_LLAMA_CPP not defined)"
        });
#endif
    }

    std::expected<Core::GenerationResult, Core::LLMError>
    LlamaCppProvider::Generate(const std::string& prompt, int maxTokens) {
        // Delegate to streaming with no callback
        return GenerateStreaming(prompt, maxTokens, nullptr);
    }

    std::expected<Core::GenerationResult, Core::LLMError>
    LlamaCppProvider::GenerateStreaming(const std::string& prompt, int maxTokens,
                                        Core::StreamCallback callback) {
#ifdef HAS_LLAMA_CPP
        std::lock_guard lock(m_mutex);

        if (!m_impl->loaded || !m_impl->ctx || !m_impl->model) {
            LOG_ERROR("LlamaCpp", "GenerateStreaming() — no model loaded");
            return std::unexpected(Core::LLMError{
                Core::LLMError::Code::GenerationFailed,
                "No model loaded"
            });
        }

        LOG_INFO("LlamaCpp", "GenerateStreaming() maxTokens=" + std::to_string(maxTokens) +
            " prompt_len=" + std::to_string(prompt.size()) + " chars");
        auto startTime = std::chrono::steady_clock::now();

        // ─── Reset cancellation token ───
        // The token may have been set by a previous user cancel.
        m_cancelToken.Reset();
        m_cancelToken.ClearEvalProgress();
        m_cancelToken.Heartbeat();  // Initial heartbeat so watchdog knows we've started

        // ─── Tokenize full prompt ───
        const int n_ctx = m_impl->params.contextWindowSize;
        std::vector<llama_token> promptTokens(n_ctx);
        int n_prompt_tokens = llama_tokenize(
            m_impl->vocab, prompt.c_str(), static_cast<int>(prompt.size()),
            promptTokens.data(), n_ctx,
            true,   // add BOS
            true    // special tokens
        );

        if (n_prompt_tokens < 0) {
            LOG_ERROR("LlamaCpp", "Tokenization failed — prompt too long");
            return std::unexpected(Core::LLMError{
                Core::LLMError::Code::ContextOverflow,
                "Prompt too long for context window"
            });
        }
        promptTokens.resize(static_cast<size_t>(n_prompt_tokens));
        LOG_INFO("LlamaCpp", "Tokenized: " + std::to_string(n_prompt_tokens) + " tokens");

        // Check context overflow
        if (n_prompt_tokens + maxTokens > n_ctx) {
            maxTokens = n_ctx - n_prompt_tokens;
            LOG_WARN("LlamaCpp", "Context overflow — clamped maxTokens to " + std::to_string(maxTokens));
            if (maxTokens <= 0) {
                return std::unexpected(Core::LLMError{
                    Core::LLMError::Code::ContextOverflow,
                    "Prompt consumes entire context window"
                });
            }
        }

        // ═══════════════════════════════════════════════════════════════
        //  Gold Standard Two-Stage KV Cache
        // ═══════════════════════════════════════════════════════════════
        //
        //  Stage 1 — System Prefix Pin
        //    On the very first call after model load, evaluate and store the
        //    system prefix in KV seq_id=1 (treated as immutable). On all
        //    subsequent calls, these positions are never re-evaluated.
        //
        //  Stage 2 — Context Block Pin
        //    Hash the document context block (the retrieved passages).
        //    If the hash matches the previous call, skip re-evaluation of
        //    the context entirely — only evaluate the question tail.
        //    If it differs, evict only [systemPrefixLen, contextEnd) and
        //    re-evaluate the new context. The system prefix is untouched.
        // ═══════════════════════════════════════════════════════════════

        int n_match = 0; // Legacy fallback — used when staged approach isn't applicable

        if (!m_impl->systemPrefixWarm) {
            // ── Stage 1: First call — evaluate and pin the system prefix ──
            // We identify the system prefix by its known stable content.
            // The prefix ends at the first occurrence of "\n\nYou are" (where
            // the intent-specific persona begins) or at a safe fallback length.
            //
            // Strategy: tokenize the full prompt, scan forward to find the
            // boundary token index where the system prefix ends, then evaluate
            // only those tokens and mark them as pinned.
            //
            // For simplicity and safety, we define the system prefix as the
            // CRITICAL RULES block (the first 4 numbered lines up to the
            // first blank line before the intent persona).

            // Identify prefix boundary: find "\n\nYou are" in the raw prompt string.
            const std::string personaMarker = "\n\nYou are";
            size_t markerPos = prompt.find(personaMarker);

            // Re-tokenize just the prefix portion to get its exact length
            int prefixCharLen = (markerPos != std::string::npos)
                                ? static_cast<int>(markerPos)
                                : std::min(static_cast<int>(prompt.size()), 400); // fallback: first 400 chars

            std::vector<llama_token> prefixToks(n_ctx);
            int nPrefixToks = llama_tokenize(
                m_impl->vocab, prompt.c_str(), prefixCharLen,
                prefixToks.data(), n_ctx, true, true);

            if (nPrefixToks > 0 && nPrefixToks < n_prompt_tokens) {
                prefixToks.resize(static_cast<size_t>(nPrefixToks));

                LOG_INFO("LlamaCpp", "[KV Stage 1] Pinning system prefix: "
                    + std::to_string(nPrefixToks) + " tokens (first call after model load)");

                // Evaluate prefix tokens and store them in the KV cache at seq_id=0
                // (we use seq_id=0 throughout — llama.cpp single-sequence mode)
                uint32_t n_batch = llama_n_batch(m_impl->ctx);
                llama_memory_clear(llama_get_memory(m_impl->ctx), true);

                for (int pi = 0; pi < nPrefixToks; pi += static_cast<int>(n_batch)) {
                    int chunk = std::min(static_cast<int>(n_batch), nPrefixToks - pi);
                    llama_batch batch = llama_batch_init(chunk, 0, 1);
                    for (int j = 0; j < chunk; j++) {
                        llama_batch_add(batch, prefixToks[pi + j], pi + j, {0}, false);
                    }
                    if (llama_decode(m_impl->ctx, batch) != 0) {
                        llama_batch_free(batch);
                        LOG_WARN("LlamaCpp", "[KV Stage 1] Prefix evaluation failed — will proceed without pin");
                        nPrefixToks = 0;
                        break;
                    }
                    llama_batch_free(batch);
                }

                if (nPrefixToks > 0) {
                    m_impl->systemPrefixTokens = std::move(prefixToks);
                    m_impl->systemPrefixLen    = nPrefixToks;
                    m_impl->systemPrefixWarm   = true;
                    // Seed cachedTokens with what we just evaluated
                    m_impl->cachedTokens = m_impl->systemPrefixTokens;
                    LOG_INFO("LlamaCpp", "[KV Stage 1] System prefix pinned at positions [0, "
                        + std::to_string(nPrefixToks) + ") — will never be re-evaluated");
                }
            } else {
                LOG_WARN("LlamaCpp", "[KV Stage 1] Could not isolate system prefix — falling back to full eval");
            }
        }

        // ── Stage 2: Context block hash check ──
        // Extract the context portion of the prompt: everything between the
        // persona block and the "=== QUESTION ===" marker.
        bool contextHit = false;
        int  contextStartPos = m_impl->systemPrefixLen; // KV position where context begins
        int  contextEndPos   = contextStartPos + m_impl->contextLen; // where it ends

        {
            // Identify the context block by its delimiters in the raw prompt string
            const std::string ctxStartMarker = "=== DOCUMENT CONTENT ===";
            const std::string ctxEndMarker   = "=== QUESTION ===";

            size_t ctxStart = prompt.find(ctxStartMarker);
            size_t ctxEnd   = prompt.find(ctxEndMarker);

            if (ctxStart != std::string::npos && ctxEnd != std::string::npos && ctxEnd > ctxStart) {
                // Hash just the context block substring
                std::string contextSubstr = prompt.substr(ctxStart, ctxEnd - ctxStart);
                size_t newContextHash = HashString(contextSubstr);

                if (m_impl->contextWarm && newContextHash == m_impl->contextHash) {
                    // ✅ Stage 2 HIT — same document context, skip re-evaluation
                    contextHit = true;
                    n_match = contextEndPos; // Reuse everything up to end of context block
                    LOG_INFO("LlamaCpp", "[KV Stage 2] Context HIT — reusing "
                        + std::to_string(n_match) + " tokens (system+context). "
                        "Only question tail ("
                        + std::to_string(n_prompt_tokens - n_match) + " tok) will be evaluated.");

                    // Evict the old question tail — keep system prefix + context
                    llama_memory_seq_rm(llama_get_memory(m_impl->ctx), 0, n_match, -1);
                } else {
                    // Stage 2 MISS — new document or first call
                    // Evict the context region (keep system prefix)
                    if (m_impl->contextWarm) {
                        LOG_INFO("LlamaCpp", "[KV Stage 2] Context MISS — document changed. "
                            "Evicting context region [" + std::to_string(contextStartPos)
                            + ", " + std::to_string(contextEndPos) + ") and re-evaluating.");
                        // Evict everything from systemPrefixLen onward (context + old tail)
                        llama_memory_seq_rm(llama_get_memory(m_impl->ctx), 0,
                                            m_impl->systemPrefixLen, -1);
                    } else if (!m_impl->systemPrefixWarm) {
                        // Neither stage warm yet — full clear
                        LOG_INFO("LlamaCpp", "[KV Stage 2] Cold start — full cache clear.");
                        llama_memory_clear(llama_get_memory(m_impl->ctx), true);
                    } else {
                        // System prefix warm, context cold — evict from systemPrefixLen onward
                        llama_memory_seq_rm(llama_get_memory(m_impl->ctx), 0,
                                            m_impl->systemPrefixLen, -1);
                        LOG_INFO("LlamaCpp", "[KV Stage 2] Context cold — evicting from pos "
                            + std::to_string(m_impl->systemPrefixLen) + " onward.");
                    }

                    // Record the new context hash and token count
                    // We'll update contextLen after evaluating the new tokens below
                    m_impl->contextHash  = newContextHash;
                    m_impl->contextWarm  = false; // Mark as pending — will be set warm after eval

                    // n_match = system prefix (already in KV cache if warm)
                    n_match = m_impl->systemPrefixWarm ? m_impl->systemPrefixLen : 0;

                    // After evaluation, measure how many context tokens were processed
                    // by comparing n_prompt_tokens against the question tail length
                    size_t tailStart = prompt.rfind(ctxEndMarker);
                    if (tailStart != std::string::npos) {
                        std::vector<llama_token> tailToks(256);
                        int nTail = llama_tokenize(
                            m_impl->vocab,
                            prompt.c_str() + tailStart,
                            static_cast<int>(prompt.size() - tailStart),
                            tailToks.data(), 256, false, true);
                        if (nTail > 0) {
                            int newContextLen = n_prompt_tokens - m_impl->systemPrefixLen - nTail;
                            if (newContextLen > 0) {
                                m_impl->contextLen  = newContextLen;
                                m_impl->contextWarm = true;
                                contextEndPos = contextStartPos + newContextLen;
                                LOG_INFO("LlamaCpp", "[KV Stage 2] New context block: "
                                    + std::to_string(newContextLen) + " tokens pinned at pos ["
                                    + std::to_string(contextStartPos) + ", "
                                    + std::to_string(contextEndPos) + ")");
                            }
                        }
                    }
                }
            } else {
                // No recognizable context delimiters — fall back to legacy prefix match
                if (m_impl->kvCacheWarm) {
                    for (size_t i = 0; i < static_cast<size_t>(n_prompt_tokens)
                                    && i < m_impl->cachedTokens.size(); i++) {
                        if (promptTokens[i] == m_impl->cachedTokens[i]) n_match++;
                        else break;
                    }
                    if (n_match > 0) {
                        llama_memory_seq_rm(llama_get_memory(m_impl->ctx), 0, n_match, -1);
                    } else {
                        llama_memory_clear(llama_get_memory(m_impl->ctx), true);
                        m_impl->cachedTokens.clear();
                    }
                    LOG_INFO("LlamaCpp", "[KV Legacy] Prefix match: " + std::to_string(n_match) + " tokens");
                } else {
                    llama_memory_clear(llama_get_memory(m_impl->ctx), true);
                    m_impl->cachedTokens.clear();
                }
            }
        }

        // Update cachedTokens to reflect current KV state
        if (contextHit) {
            m_impl->cachedTokens.resize(static_cast<size_t>(n_match));
        } else if (m_impl->systemPrefixWarm && n_match == m_impl->systemPrefixLen) {
            m_impl->cachedTokens = m_impl->systemPrefixTokens;
        }

        // Report eval progress so the UI knows where we are
        m_cancelToken.ReportEvalProgress(n_match, n_prompt_tokens);

        if (n_match < n_prompt_tokens) {
            LOG_DEBUG("LlamaCpp", "Evaluating " + std::to_string(n_prompt_tokens - n_match) + " new prompt tokens...");
            uint32_t n_batch = llama_n_batch(m_impl->ctx);
            
            for (int i = n_match; i < n_prompt_tokens; i += n_batch) {
                if (m_cancelToken.IsCancelled()) {
                    LOG_WARN("LlamaCpp", "Cancelled during prompt evaluation at token " + std::to_string(i));
                    m_impl->kvCacheWarm = false;
                    m_cancelToken.ClearEvalProgress();
                    return std::unexpected(Core::LLMError{
                        Core::LLMError::Code::Cancelled,
                        "Generation cancelled by user during prompt evaluation"
                    });
                }

                int chunk_size = std::min(static_cast<int>(n_batch), n_prompt_tokens - i);
                llama_batch batch = llama_batch_init(chunk_size, 0, 1);
                
                for (int j = 0; j < chunk_size; j++) {
                    llama_batch_add(batch, promptTokens[i + j], i + j, {0}, false);
                    m_impl->cachedTokens.push_back(promptTokens[i + j]);
                }
                
                // Only request logits for the very last token of the entire prompt
                if (i + chunk_size == n_prompt_tokens) {
                    batch.logits[batch.n_tokens - 1] = true;
                }

                if (llama_decode(m_impl->ctx, batch) != 0) {
                    llama_batch_free(batch);
                    LOG_ERROR("LlamaCpp", "Prompt evaluation FAILED at chunk starting at " + std::to_string(i));
                    m_cancelToken.ClearEvalProgress();
                    return std::unexpected(Core::LLMError{
                        Core::LLMError::Code::GenerationFailed,
                        "Failed to evaluate prompt chunk"
                    });
                }
                llama_batch_free(batch);

                m_cancelToken.Heartbeat();
                m_cancelToken.ReportEvalProgress(std::min(i + chunk_size, n_prompt_tokens), n_prompt_tokens);
            }
        }

        m_impl->kvCacheWarm = true;
        m_cancelToken.ClearEvalProgress();
        LOG_DEBUG("LlamaCpp", "Prompt ready — starting token generation");

        // ─── Generate tokens using cached sampler ───
        Core::GenerationResult result;
        result.tokensPrompt = n_prompt_tokens;

        // Reset sampler state for a fresh generation (without rebuilding the chain)
        if (m_impl->sampler) {
            llama_sampler_reset(m_impl->sampler);
        } else {
            // Fallback: build a new sampler if somehow not cached
            m_impl->sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler_chain_add(m_impl->sampler, llama_sampler_init_penalties(256, 1.1f, 0.0f, 0.0f));
            llama_sampler_chain_add(m_impl->sampler, llama_sampler_init_top_k(m_impl->params.topK));
            llama_sampler_chain_add(m_impl->sampler, llama_sampler_init_top_p(m_impl->params.topP, 1));
            llama_sampler_chain_add(m_impl->sampler, llama_sampler_init_temp(m_impl->params.temperature));
            llama_sampler_chain_add(m_impl->sampler, llama_sampler_init_dist(0));
        }

        int n_decode = 0;
        int curPos = n_prompt_tokens;

        for (int i = 0; i < maxTokens; i++) {
            // ─── CANCELLATION CHECKPOINT ─── (between generated tokens)
            if (m_cancelToken.IsCancelled()) {
                LOG_WARN("LlamaCpp", "Cancelled during token generation at token " + std::to_string(i));
                m_impl->kvCacheWarm = false;
                break;  // Return partial result — whatever we generated so far
            }

            llama_token newToken = llama_sampler_sample(m_impl->sampler, m_impl->ctx, -1);
            llama_sampler_accept(m_impl->sampler, newToken);

            // Check for end of generation
            if (llama_vocab_is_eog(m_impl->vocab, newToken)) {
                LOG_DEBUG("LlamaCpp", "EOG token reached at position " + std::to_string(i));
                break;
            }

            // Convert token to text
            char buf[256];
            int n = llama_token_to_piece(m_impl->vocab, newToken, buf, sizeof(buf), 0, true);
            if (n > 0) {
                std::string piece(buf, static_cast<size_t>(n));
                result.text += piece;
                if (callback) callback(piece);
                m_cancelToken.Heartbeat();  // Prove we're alive during generation
            }

            // Prepare batch for next token
            llama_batch nextBatch = llama_batch_init(1, 0, 1);
            llama_batch_add(nextBatch, newToken, curPos, {0}, true);
            curPos++;

            if (llama_decode(m_impl->ctx, nextBatch) != 0) {
                llama_batch_free(nextBatch);
                LOG_WARN("LlamaCpp", "Decode error at token " + std::to_string(i) + " — returning partial");
                m_impl->kvCacheWarm = false;  // Invalidate cache on error
                break;
            }
            llama_batch_free(nextBatch);
            
            // Cache the newly generated token so it can be reused in future follow-up queries
            m_impl->cachedTokens.push_back(newToken);
            n_decode++;
        }

        auto endTime = std::chrono::steady_clock::now();
        result.tokensGenerated = n_decode;
        result.generationTimeMs = std::chrono::duration<double, std::milli>(
            endTime - startTime).count();
        result.wasTruncated = (n_decode >= maxTokens);

        double tokPerSec = (result.generationTimeMs > 0)
            ? (n_decode * 1000.0 / result.generationTimeMs) : 0;
        LOG_INFO("LlamaCpp", "Generation complete: " + std::to_string(n_decode) + " tokens in " +
            std::to_string(result.generationTimeMs) + " ms (" +
            std::to_string(tokPerSec) + " tok/s)" +
            (result.wasTruncated ? " [TRUNCATED]" : ""));

        return result;
#else
        LOG_ERROR("LlamaCpp", "GenerateStreaming() — HAS_LLAMA_CPP not defined");
        return std::unexpected(Core::LLMError{
            Core::LLMError::Code::GenerationFailed,
            "llama.cpp not compiled in (HAS_LLAMA_CPP not defined)"
        });
#endif
    }

    size_t LlamaCppProvider::GetContextWindowSize() const {
        std::lock_guard lock(m_mutex);
        if (m_impl->loaded) {
            return static_cast<size_t>(m_impl->params.contextWindowSize);
        }
        return 0;
    }

    size_t LlamaCppProvider::EstimateTokenCount(const std::string& text) const {
#ifdef HAS_LLAMA_CPP
        std::lock_guard lock(m_mutex);
        if (m_impl->loaded && m_impl->vocab) {
            // Upper bound: tokens are always fewer than characters.
            // text.size()/2+4 avoids a 32K-int heap allocation for large contexts.
            int maxEstimate = static_cast<int>(text.size() / 2) + 4;
            std::vector<llama_token> tokens(maxEstimate);
            int count = llama_tokenize(m_impl->vocab, text.c_str(),
                                        static_cast<int>(text.size()),
                                        tokens.data(), maxEstimate, false, false);
            // count < 0 means the buffer was too small; fall back to char estimate
            return count > 0 ? static_cast<size_t>(count) : text.size() / 4;
        }
#endif
        return (text.size() + 3) / 4;
    }

    bool LlamaCppProvider::IsModelLoaded() const {
        std::lock_guard lock(m_mutex);
        return m_impl->loaded;
    }

    void LlamaCppProvider::UnloadModel() {
#ifdef HAS_LLAMA_CPP
        std::lock_guard lock(m_mutex);
        if (m_impl->loaded) {
            LOG_INFO("LlamaCpp", "UnloadModel: " + m_impl->status.modelName);
        }
        // Free cached sampler
        if (m_impl->sampler) {
            llama_sampler_free(m_impl->sampler);
            m_impl->sampler = nullptr;
        }
        if (m_impl->ctx) {
            llama_free(m_impl->ctx);
            m_impl->ctx = nullptr;
        }
        if (m_impl->model) {
            llama_model_free(m_impl->model);
            m_impl->model = nullptr;
        }
        m_impl->vocab = nullptr;
        m_impl->loaded = false;
        m_impl->kvCacheWarm = false;
        m_impl->status = {};
#endif
    }

    Core::ModelStatus LlamaCppProvider::GetStatus() const {
        std::lock_guard lock(m_mutex);
        return m_impl->status;
    }

} // namespace LocalNotebookLLM::Infrastructure
