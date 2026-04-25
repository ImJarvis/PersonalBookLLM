#include "Infrastructure/LLM/LlamaCppEmbeddingProvider.h"
#include "Core/Log.h"

#ifdef HAS_LLAMA_CPP
#include <llama.h>
#endif

// Polyfill for batching if needed
#ifdef HAS_LLAMA_CPP
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

    struct LlamaCppEmbeddingProvider::Impl {
        bool loaded = false;
        size_t dimensions = 0;
        // nomic-embed-text requires task prefixes for optimal retrieval quality:
        //   passages indexed  → "search_document: "
        //   user query        → "search_query: "
        // We auto-detect this by model filename at load time.
        bool isNomicModel = false;
#ifdef HAS_LLAMA_CPP
        llama_model* model = nullptr;
        llama_context* ctx = nullptr;
        const llama_vocab* vocab = nullptr;
#endif
    };

    LlamaCppEmbeddingProvider::LlamaCppEmbeddingProvider()
        : m_impl(std::make_unique<Impl>()) {
    }

    LlamaCppEmbeddingProvider::~LlamaCppEmbeddingProvider() {
        Unload();
    }

    std::expected<void, std::string>
    LlamaCppEmbeddingProvider::LoadModel(const std::filesystem::path& modelPath) {
#ifdef HAS_LLAMA_CPP
        std::lock_guard lock(m_mutex);

        if (m_impl->loaded) {
            Unload();
        }

        if (!std::filesystem::exists(modelPath)) {
            return std::unexpected("Embedding model not found: " + modelPath.string());
        }

        auto mparams = llama_model_default_params();
        mparams.n_gpu_layers = 99; // Try fully offloading embedding model to GPU (usually small enough)

        m_impl->model = llama_model_load_from_file(modelPath.string().c_str(), mparams);
        if (!m_impl->model) {
            return std::unexpected("llama_model_load_from_file FAILED for embeddings");
        }

        m_impl->vocab = llama_model_get_vocab(m_impl->model);

        auto cparams = llama_context_default_params();
        cparams.n_ctx = 8192; // Max sum of batch tokens allowed
        cparams.n_batch = 8192; // Match batch evaluation block max to ctx
        cparams.n_threads = 4;
        cparams.embeddings = true; // Crucial: explicitly enable embeddings mode

        m_impl->ctx = llama_init_from_model(m_impl->model, cparams);
        if (!m_impl->ctx) {
            llama_model_free(m_impl->model);
            m_impl->model = nullptr;
            return std::unexpected("llama_init_from_model FAILED for embeddings (OOM?)");
        }

        // Get embedding dimensions
        const int n_embd = llama_n_embd(m_impl->model);
        if (n_embd <= 0) {
            llama_free(m_impl->ctx);
            llama_model_free(m_impl->model);
            m_impl->ctx = nullptr;
            m_impl->model = nullptr;
            return std::unexpected("Loaded model does not support embeddings (n_embd = 0)");
        }
        
        m_impl->dimensions = static_cast<size_t>(n_embd);
        m_impl->loaded = true;

        // Auto-detect nomic-embed: requires task prefixes for optimal performance
        std::string modelName = modelPath.filename().string();
        for (auto& c : modelName) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        m_impl->isNomicModel = (modelName.find("nomic") != std::string::npos);
        if (m_impl->isNomicModel) {
            LOG_INFO("LlamaCppEmbed", "nomic-embed detected — will apply task prefixes automatically");
        }

        LOG_INFO("LlamaCppEmbed", "Successfully loaded embedding model. Dimensions: " + std::to_string(n_embd));
        return {};
#else
        return std::unexpected("HAS_LLAMA_CPP not defined");
#endif
    }

    std::expected<std::vector<float>, std::string>
    LlamaCppEmbeddingProvider::Embed(const std::string& text, bool isQuery) {
#ifdef HAS_LLAMA_CPP
        std::lock_guard lock(m_mutex);

        if (!m_impl->loaded || !m_impl->ctx || !m_impl->model) {
            return std::unexpected("Embedding model not loaded");
        }
        
        // Apply nomic task prefix for retrieval passages
        // (nomic-embed-text-v1.5 requires "search_document: " prefix on indexed text)
        std::string prefixedText = text;
        if (m_impl->isNomicModel) {
            prefixedText = (isQuery ? "search_query: " : "search_document: ") + text;
        }

        // Tokenize
        std::vector<llama_token> tokens(2048);
        int n_tokens = llama_tokenize(
            m_impl->vocab, prefixedText.c_str(), static_cast<int>(prefixedText.size()),
            tokens.data(), static_cast<int>(tokens.size()),
            true, true
        );
        
        if (n_tokens < 0) {
            return std::unexpected("Context limit exceeded for embedding chunk");
        }
        
        llama_memory_clear(llama_get_memory(m_impl->ctx), true);

        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        for (int i = 0; i < n_tokens; i++) {
            llama_batch_add(batch, tokens[i], i, {0}, i == n_tokens - 1);
        }

        if (llama_decode(m_impl->ctx, batch) != 0) {
            llama_batch_free(batch);
            return std::unexpected("Failed to decode token batch for embedding");
        }
        llama_batch_free(batch);

        // Fetch embeddings (usually Sequence 0 for typical models)
        // Some models use llama_get_embeddings_seq others use llama_get_embeddings
        // Newer API prefers llama_get_embeddings_seq
        const float* embd = llama_get_embeddings_seq(m_impl->ctx, 0);
        if (embd == nullptr) {
            // Fallback to token-level embeddings (pooling req)
            embd = llama_get_embeddings_ith(m_impl->ctx, n_tokens - 1);
        }

        if (!embd) {
             return std::unexpected("Model did not return embedding vector");
        }

        std::vector<float> result(embd, embd + m_impl->dimensions);
        
        // L2 Normalize the vector for cosine similarity optimization
        float sum = 0.0f;
        for (float v : result) sum += v * v;
        if (sum > 0.0f) {
            float norm = std::sqrt(sum);
            for (float& v : result) v /= norm;
        }

        return result;
#else
        return std::unexpected("HAS_LLAMA_CPP not defined");
#endif
    }

    std::expected<std::vector<std::vector<float>>, std::string>
    LlamaCppEmbeddingProvider::EmbedBatch(const std::vector<std::string>& texts) {
#ifdef HAS_LLAMA_CPP
        std::lock_guard lock(m_mutex);
        if (!m_impl->loaded || !m_impl->ctx || !m_impl->model) {
            return std::unexpected("Embedding model not loaded");
        }

        std::vector<std::vector<float>> results(texts.size());
        if (texts.empty()) return results;

        llama_memory_clear(llama_get_memory(m_impl->ctx), true);

        // Pre-tokenize all strings (with nomic prefix if needed)
        std::vector<std::vector<llama_token>> all_tokens;
        int total_tokens = 0;
        for (const auto& txt : texts) {
            std::string prefixed = m_impl->isNomicModel
                ? "search_document: " + txt
                : txt;
            std::vector<llama_token> tokens(2048);
            int n = llama_tokenize(m_impl->vocab, prefixed.c_str(), static_cast<int>(prefixed.size()), tokens.data(), static_cast<int>(tokens.size()), true, true);
            if (n < 0) return std::unexpected("A chunk exceeds context limit");
            tokens.resize(n);
            total_tokens += n;
            all_tokens.push_back(std::move(tokens));
        }

        if (total_tokens > 8192) {
            return std::unexpected("Total batch tokens exceed context window");
        }

        llama_batch batch = llama_batch_init(total_tokens, 0, static_cast<int32_t>(texts.size()));
        for (size_t seq = 0; seq < all_tokens.size(); ++seq) {
            const auto& toks = all_tokens[seq];
            for (size_t i = 0; i < toks.size(); i++) {
                llama_batch_add(batch, toks[i], static_cast<llama_pos>(i), {static_cast<llama_seq_id>(seq)}, i == toks.size() - 1);
            }
        }

        if (llama_decode(m_impl->ctx, batch) != 0) {
            llama_batch_free(batch);
            return std::unexpected("Failed to decode token batch for embeddings");
        }
        llama_batch_free(batch);

        for (size_t seq = 0; seq < texts.size(); ++seq) {
            const float* embd = llama_get_embeddings_seq(m_impl->ctx, static_cast<llama_seq_id>(seq));
            if (!embd) {
                // Fallback attempt (unlikely if sequential embeddings enabled natively)
                return std::unexpected("Model did not return embedding vector for seq " + std::to_string(seq));
            }
            
            std::vector<float> res(embd, embd + m_impl->dimensions);
            
            float sum = 0.0f;
            for (float v : res) sum += v * v;
            if (sum > 0.0f) {
                float norm = std::sqrt(sum);
                for (float& v : res) v /= norm;
            }
            results[seq] = std::move(res);
        }

        return results;
#else
        return std::unexpected("HAS_LLAMA_CPP not defined");
#endif
    }

    bool LlamaCppEmbeddingProvider::IsLoaded() const {
        std::lock_guard lock(m_mutex);
        return m_impl->loaded;
    }

    size_t LlamaCppEmbeddingProvider::GetDimensions() const {
        std::lock_guard lock(m_mutex);
        return m_impl->dimensions;
    }

    void LlamaCppEmbeddingProvider::Unload() {
#ifdef HAS_LLAMA_CPP
        std::lock_guard lock(m_mutex);
        if (m_impl->loaded) {
            LOG_INFO("LlamaCppEmbed", "Unloading embedding model");
            if (m_impl->ctx) { llama_free(m_impl->ctx); m_impl->ctx = nullptr; }
            if (m_impl->model) { llama_model_free(m_impl->model); m_impl->model = nullptr; }
            m_impl->vocab = nullptr;
            m_impl->loaded = false;
        }
#endif
    }

} // namespace LocalNotebookLLM::Infrastructure
