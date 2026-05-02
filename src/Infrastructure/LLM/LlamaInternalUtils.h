#pragma once
// ─── Shared llama.cpp internal helpers ───────────────────────────────────────
// This header is for INTERNAL use by Infrastructure/LLM TUs only.
// It contains polyfills for API functions removed from llama.cpp (e.g. llama_batch_add)
// that both LlamaCppProvider and LlamaCppEmbeddingProvider need identically.
// Having them in one place prevents silent divergence if the llama_batch struct changes.
// ─────────────────────────────────────────────────────────────────────────────

#ifdef HAS_LLAMA_CPP
#include <llama.h>
#include <vector>

/// Polyfill for the removed llama_batch_add() utility function.
/// llama.cpp removed this from its public API; we replicate it locally
/// to avoid being blocked on API churn across releases.
static inline void llama_batch_add(
    struct llama_batch& batch,
    llama_token         id,
    llama_pos           pos,
    const std::vector<llama_seq_id>& seq_ids,
    bool                logits)
{
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = static_cast<int32_t>(seq_ids.size());
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits[batch.n_tokens] = logits ? 1 : 0;
    batch.n_tokens++;
}
#endif // HAS_LLAMA_CPP
