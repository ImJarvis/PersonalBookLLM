#include "Infrastructure/LLM/LlamaCppProvider.h"
#include "Infrastructure/LLM/GPUDetector.h"
#include "Core/Log.h"
#include <chrono>
#include <thread>

// ─── Conditionally include llama.cpp headers ───
#ifdef HAS_LLAMA_CPP
#include <llama.h>
#include <ggml.h>
#endif

#ifdef HAS_LLAMA_CPP
#include <llama.h> // Ensure we load the API headers
#include <vector>

// Helper to polyfill the removed llama_batch_add utility function
static inline void llama_batch_add(
    struct llama_batch & batch,
    llama_token   id,
    llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
    bool   logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = static_cast<int32_t>(seq_ids.size());
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits ? 1 : 0;
    batch.n_tokens++;
}
#endif

namespace LocalNotebookLLM::Infrastructure {

    struct LlamaCppProvider::Impl {
#ifdef HAS_LLAMA_CPP
        llama_model*   model   = nullptr;
        llama_context* ctx     = nullptr;
        const llama_vocab* vocab = nullptr;
#endif
        Core::ModelParams params;
        Core::ModelStatus status;
        bool loaded = false;
    };

    LlamaCppProvider::LlamaCppProvider()
        : m_impl(std::make_unique<Impl>()) {
#ifdef HAS_LLAMA_CPP
        llama_backend_init();
        LOG_INFO("LlamaCpp", "Backend initialized (HAS_LLAMA_CPP=1)");
#else
        LOG_WARN("LlamaCpp", "Compiled WITHOUT llama.cpp (HAS_LLAMA_CPP not defined)");
#endif
    }

    LlamaCppProvider::~LlamaCppProvider() {
        LOG_DEBUG("LlamaCpp", "Destructor — unloading model");
        UnloadModel();
#ifdef HAS_LLAMA_CPP
        llama_backend_free();
        LOG_DEBUG("LlamaCpp", "Backend freed");
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
        cparams.flash_attn_type = params.useFlashAttention ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;

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

        LOG_INFO("LlamaCpp", "=== Model READY: " + m_impl->status.modelName +
            " | device=" + m_impl->status.device +
            " | gpuLayers=" + std::to_string(gpuLayers) +
            " | ctx=" + std::to_string(params.contextWindowSize) + " ===");

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

        // ─── Tokenize prompt ───
        const int n_prompt_max = m_impl->params.contextWindowSize;
        std::vector<llama_token> promptTokens(n_prompt_max);
        int n_prompt_tokens = llama_tokenize(
            m_impl->vocab, prompt.c_str(), static_cast<int>(prompt.size()),
            promptTokens.data(), n_prompt_max,
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
        if (n_prompt_tokens + maxTokens > m_impl->params.contextWindowSize) {
            maxTokens = m_impl->params.contextWindowSize - n_prompt_tokens;
            LOG_WARN("LlamaCpp", "Context overflow — clamped maxTokens to " + std::to_string(maxTokens));
            if (maxTokens <= 0) {
                return std::unexpected(Core::LLMError{
                    Core::LLMError::Code::ContextOverflow,
                    "Prompt consumes entire context window"
                });
            }
        }

        // Clear KV cache for fresh generation
        llama_memory_clear(llama_get_memory(m_impl->ctx), true);

        // ─── Evaluate prompt ───
        LOG_DEBUG("LlamaCpp", "Evaluating prompt...");
        llama_batch batch = llama_batch_init(n_prompt_tokens, 0, 1);
        for (int i = 0; i < n_prompt_tokens; i++) {
            llama_batch_add(batch, promptTokens[i], i, {0}, false);
        }
        batch.logits[batch.n_tokens - 1] = true;

        if (llama_decode(m_impl->ctx, batch) != 0) {
            llama_batch_free(batch);
            LOG_ERROR("LlamaCpp", "Prompt evaluation FAILED");
            return std::unexpected(Core::LLMError{
                Core::LLMError::Code::GenerationFailed,
                "Failed to evaluate prompt"
            });
        }
        llama_batch_free(batch);
        LOG_DEBUG("LlamaCpp", "Prompt evaluated — starting token generation");

        // ─── Generate tokens ───
        Core::GenerationResult result;
        result.tokensPrompt = n_prompt_tokens;

        auto sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(m_impl->params.topK));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(m_impl->params.topP, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(m_impl->params.temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(0));

        int n_decode = 0;
        int curPos = n_prompt_tokens;

        for (int i = 0; i < maxTokens; i++) {
            llama_token newToken = llama_sampler_sample(sampler, m_impl->ctx, -1);

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
            }

            // Prepare batch for next token
            llama_batch nextBatch = llama_batch_init(1, 0, 1);
            llama_batch_add(nextBatch, newToken, curPos, {0}, true);
            curPos++;

            if (llama_decode(m_impl->ctx, nextBatch) != 0) {
                llama_batch_free(nextBatch);
                LOG_WARN("LlamaCpp", "Decode error at token " + std::to_string(i) + " — returning partial");
                break;
            }
            llama_batch_free(nextBatch);
            n_decode++;
        }

        llama_sampler_free(sampler);

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
            int maxTokens = static_cast<int>(text.size());
            std::vector<llama_token> tokens(maxTokens);
            int count = llama_tokenize(m_impl->vocab, text.c_str(),
                                        static_cast<int>(text.size()),
                                        tokens.data(), maxTokens, false, false);
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
        m_impl->status = {};
#endif
    }

    Core::ModelStatus LlamaCppProvider::GetStatus() const {
        std::lock_guard lock(m_mutex);
        return m_impl->status;
    }

} // namespace LocalNotebookLLM::Infrastructure
