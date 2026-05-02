#include "Application/InferenceService.h"
#include "Application/QueryAnalyzer.h"
#include "Core/Log.h"
#include <sstream>
#include <regex>
#include <set>
#include <chrono>

namespace LocalNotebookLLM::Application {

    struct InferenceService::Impl {
        std::shared_ptr<Core::ILLMProvider>        llm;
        std::shared_ptr<Core::IHybridSearch>       search;
        std::unique_ptr<Core::IContextAssembler>   assembler;
        std::shared_ptr<Core::IMemoryOrchestrator> memoryOrch;
    };

    InferenceService::InferenceService(
        std::shared_ptr<Core::ILLMProvider> llmProvider,
        std::shared_ptr<Core::IHybridSearch> search,
        std::unique_ptr<Core::IContextAssembler> assembler,
        std::shared_ptr<Core::IMemoryOrchestrator> memoryOrch)
        : m_impl(std::make_unique<Impl>())
    {
        m_impl->llm        = std::move(llmProvider);
        m_impl->search     = std::move(search);
        m_impl->assembler  = std::move(assembler);
        m_impl->memoryOrch = std::move(memoryOrch);
    }

    InferenceService::~InferenceService() = default;
    InferenceService::InferenceService(InferenceService&&) noexcept = default;
    InferenceService& InferenceService::operator=(InferenceService&&) noexcept = default;

    void InferenceService::UpdateLLMProvider(std::shared_ptr<Core::ILLMProvider> newProvider) {
        if (newProvider) {
            m_impl->llm = std::move(newProvider);
            LOG_INFO("Inference", "LLM Provider updated successfully");
        }
    }

    std::expected<Core::CitedAnswer, std::string>
    InferenceService::AskQuestion(const std::string& query, AnswerProgressCallback progress) {
        LOG_INFO("Inference", "AskQuestion() query: \"" + query + "\"");
        if (!m_impl->llm->IsModelLoaded()) {
            LOG_ERROR("Inference", "AskQuestion() — no model loaded");
            return std::unexpected("No model loaded. Call LoadModel() first.");
        }

        // ─── Step 1: Search ───
        if (progress) progress({AnswerProgress::Stage::Searching, ""});

        auto results = m_impl->search->Search(query);
        LOG_INFO("Inference", "Search returned " + std::to_string(results.size()) + " results");
        if (results.empty()) {
            LOG_WARN("Inference", "No relevant content found");
            return std::unexpected("No relevant content found in indexed documents.");
        }

        // Convert HybridSearchResult → SearchResult for assembler
        std::vector<Core::SearchResult> baseResults(results.begin(), results.end());

        // ─── Step 2: Assemble context ───
        if (progress) progress({AnswerProgress::Stage::Assembling, ""});

        size_t contextWindow = m_impl->llm->GetContextWindowSize();
        auto ctx = m_impl->assembler->Assemble(baseResults, query, contextWindow);

        // ─── Step 3: Generate ───
        if (progress) progress({AnswerProgress::Stage::Generating, ""});

        std::string prompt = BuildPrompt(ctx, query);
        LOG_INFO("Inference", "Prompt built: " + std::to_string(prompt.size()) + " chars, " +
            std::to_string(ctx.sectionsIncluded) + " sections included, " +
            std::to_string(ctx.sectionsOmitted) + " omitted");
        auto genResult = m_impl->llm->Generate(prompt);

        if (!genResult) {
            LOG_ERROR("Inference", "Generation failed: " + genResult.error().message);
            return std::unexpected("Generation failed: " + genResult.error().message);
        }
        LOG_INFO("Inference", "Generated " + std::to_string(genResult->tokensGenerated) + " tokens");

        // ─── Step 4: Build cited answer ───
        Core::CitedAnswer answer;
        answer.answerText       = genResult->text;
        answer.tokensGenerated  = genResult->tokensGenerated;
        answer.tokensPrompt     = genResult->tokensPrompt;
        answer.generationTimeMs = genResult->generationTimeMs;
        answer.sectionsSearched = ctx.sectionsIncluded;
        answer.sectionsOmitted  = ctx.sectionsOmitted;
        answer.citations        = ctx.sourceMap;

        // ─── Step 5: Validate citations ───
        if (progress) progress({AnswerProgress::Stage::Validating, answer.answerText});
        ValidateCitations(answer, ctx);

        if (progress) progress({AnswerProgress::Stage::Complete, answer.answerText});
        return answer;
    }

    std::expected<Core::CitedAnswer, std::string>
    InferenceService::AskQuestionStreaming(
        const std::string& query,
        Core::StreamCallback tokenCallback,
        AnswerProgressCallback progress)
    {
        LOG_INFO("Inference", "AskQuestionStreaming() query: \"" + query + "\"");
        if (!m_impl->llm->IsModelLoaded()) {
            LOG_ERROR("Inference", "AskQuestionStreaming() — no model loaded");
            return std::unexpected("No model loaded. Call LoadModel() first.");
        }

        if (progress) progress({AnswerProgress::Stage::Searching, ""});
        auto results = m_impl->search->Search(query);
        if (results.empty()) {
            return std::unexpected("No relevant content found in indexed documents.");
        }

        std::vector<Core::SearchResult> baseResults(results.begin(), results.end());

        if (progress) progress({AnswerProgress::Stage::Assembling, ""});
        size_t contextWindow = m_impl->llm->GetContextWindowSize();
        auto ctx = m_impl->assembler->Assemble(baseResults, query, contextWindow);

        if (progress) progress({AnswerProgress::Stage::Generating, ""});
        std::string prompt = BuildPrompt(ctx, query);

        // Compute a safe generation budget to prevent context overflow.
        // n_ctx - prompt_tokens - 64 (headroom) = tokens actually available for generation.
        // Without this, a 3000-token Summation prompt + 2048 budget overflows 4096 ctx,
        // the overflow guard clamps to ~900 tokens, then the model loops endlessly.
        auto   intent       = Application::QueryAnalyzer::DetectIntent(query);
        size_t ctxWindow    = contextWindow;  // Already fetched above for Assemble()
        size_t promptEst    = m_impl->llm->EstimateTokenCount(prompt);
        int    safeGenBudget = static_cast<int>(ctxWindow)
                             - static_cast<int>(promptEst) - 64;  // 64-token headroom
        safeGenBudget = std::max(128, safeGenBudget);

        int maxGenTokens = std::min(safeGenBudget,
            (intent == Application::QueryIntent::Summation ||
             intent == Application::QueryIntent::ChapterSummary) ? 1024 : 512);

        LOG_INFO("Inference", "Generation budget: ctx=" + std::to_string(ctxWindow)
            + " promptEst=" + std::to_string(promptEst)
            + " safe=" + std::to_string(safeGenBudget)
            + " maxGenTokens=" + std::to_string(maxGenTokens));

        auto genResult = m_impl->llm->GenerateStreaming(prompt, maxGenTokens, tokenCallback);

        if (!genResult) {
            return std::unexpected("Generation failed: " + genResult.error().message);
        }

        Core::CitedAnswer answer;
        answer.answerText       = genResult->text;
        answer.tokensGenerated  = genResult->tokensGenerated;
        answer.tokensPrompt     = genResult->tokensPrompt;
        answer.generationTimeMs = genResult->generationTimeMs;
        answer.sectionsSearched = ctx.sectionsIncluded;
        answer.sectionsOmitted  = ctx.sectionsOmitted;
        answer.citations        = ctx.sourceMap;

        ValidateCitations(answer, ctx);
        if (progress) progress({AnswerProgress::Stage::Complete, answer.answerText});
        return answer;
    }

    std::expected<void, std::string>
    InferenceService::LoadModel(const std::filesystem::path& modelPath, const Core::ModelParams& params) {
        LOG_INFO("Inference", "LoadModel() path: " + modelPath.string());
        if (m_impl->memoryOrch) {
            LOG_INFO("Inference", "Delegating to MemoryOrchestrator for Reasoner");
            auto req = m_impl->memoryOrch->RequestModel(Core::ModelRole::Reasoner);
            if (!req) {
                LOG_ERROR("Inference", "MemoryOrchestrator failed: " + req.error());
                return std::unexpected("Memory orchestrator: " + req.error());
            }
            LOG_INFO("Inference", "Model loaded via MemoryOrchestrator");
            return {};
        }

        LOG_INFO("Inference", "Direct provider load (no orchestrator)");
        auto result = m_impl->llm->LoadModel(modelPath, params);
        if (!result) {
            LOG_ERROR("Inference", "Direct load failed: " + result.error().message);
            return std::unexpected(result.error().message);
        }
        LOG_INFO("Inference", "Model loaded directly");
        return {};
    }

    Core::ModelStatus InferenceService::GetModelStatus() const {
        Core::ModelStatus status;
        status.isLoaded      = m_impl->llm->IsModelLoaded();
        status.contextWindow = m_impl->llm->GetContextWindowSize();
        return status;
    }

    bool InferenceService::IsReady() const {
        return m_impl->llm->IsModelLoaded();
    }

    void InferenceService::ValidateCitations(Core::CitedAnswer& answer,
                                              const Core::AssembledContext& ctx) {
        // Extract [Source: ..., Page N] references from the answer text
        std::regex pageRegex(R"(\[Source:\s*[^,]+,\s*Page\s+(\d+)\])");
        std::smatch match;
        std::string text = answer.answerText;
        std::set<int> referencedPages;

        auto it = std::sregex_iterator(text.begin(), text.end(), pageRegex);
        auto end = std::sregex_iterator();
        for (; it != end; ++it) {
            referencedPages.insert(std::stoi((*it)[1].str()));
        }

        // Also match the simpler [Page N] format for backward compatibility
        std::regex simplePageRegex(R"(\[Page\s+(\d+)\])");
        auto it2 = std::sregex_iterator(text.begin(), text.end(), simplePageRegex);
        for (; it2 != end; ++it2) {
            referencedPages.insert(std::stoi((*it2)[1].str()));
        }

        // Mark citations as verified if they appear in the provided context
        std::set<int> contextPages;
        for (const auto& src : ctx.sourceMap) {
            contextPages.insert(src.pageNumber);
        }

        for (auto& citation : answer.citations) {
            citation.isVerified = contextPages.count(citation.pageNumber) > 0;
        }
    }

    std::string InferenceService::BuildPrompt(const Core::AssembledContext& ctx,
                                               const std::string& query) {
        auto intent = Application::QueryAnalyzer::DetectIntent(query);

        std::ostringstream oss;

        // ─── Strict Grounding Preamble (applies to ALL prompt variants) ───
        // Constrains the model to only use provided document passages.
        // Without this, 1.5B–4B models freely interpolate parametric knowledge.
        oss << "CRITICAL RULES — READ BEFORE ANSWERING:\n"
            << "1. You MUST ONLY answer using information from the passages below.\n"
            << "2. If information is not in the passages, respond: "
            << "'This information is not available in the provided documents.'\n"
            << "3. For every factual claim, cite the source concisely as [Page N].\n"
            << "4. NEVER fabricate information not present in the passages.\n\n";

        if (intent == Application::QueryIntent::Summation) {
            // Synthesis prompt: demands reasoning and explanation, forbids copy-paste.
            // Key design decisions:
            //   1. Explicitly forbid verbatim copying — small models default to extraction.
            //   2. Ask for step-by-step synthesis before giving the answer.
            //   3. No completion primer — let the model decide how to open its response.
            oss << "You are an expert document analyst. "
                << "Your task is to read the document passages below and produce an intelligent, "
                << "well-structured explanation of the document's content.\n\n"
                << "RULES:\n"
                << "- Do NOT reproduce passages verbatim.\n"
                << "- Provide a comprehensive, professional summary of the document.\n"
                << "- Use clear paragraphs and bullet points where appropriate for readability.\n"
                << "- Write in a highly professional, informative tone.\n"
                << "- Cite sources concisely as [Page N] for key claims.\n\n"
                << ctx.formattedContext
                << "=== QUESTION ===\n"
                << query << "\n\n"
                << "=== SUMMARY ===\n";
        }
        else if (intent == Application::QueryIntent::ChapterSummary) {
            // Chapter synthesis: focused explanation of a specific section.
            oss << "You are an expert document analyst. "
                << "Read the chapter passages provided below and produce an intelligent summary "
                << "of what this chapter is about.\n\n"
                << "RULES:\n"
                << "- Do NOT reproduce passages verbatim.\n"
                << "- Explain the chapter's central argument, supporting ideas, and conclusions.\n"
                << "- Write as if explaining the chapter to someone who has not read it.\n"
                << "- Keep the response structured: briefly state what the chapter is about, "
                << "then explain its key points, then state what the reader will take away.\n"
                << "- Cite sources as [Source: filename, Page N] for key claims.\n\n"
                << ctx.formattedContext
                << "=== QUESTION ===\n"
                << query << "\n\n"
                << "=== YOUR EXPLANATION ===\n";
        }
        else {
            // Specific fact/concept query: precision mode with synthesis over simple retrieval.
            oss << "You are a precise and thoughtful document assistant. "
                << "Use the document passages below to answer the question directly and clearly.\n\n"
                << "RULES:\n"
                << "- Answer in your own words — do not quote the source text verbatim.\n"
                << "- Be specific and concrete. Avoid vague hedging.\n"
                << "- If the answer requires connecting ideas from multiple passages, do so explicitly.\n"
                << "- If the information is not present in the provided passages, say so clearly.\n"
                << "- Cite sources as [Source: filename, Page N] for each factual claim.\n\n"
                << ctx.formattedContext
                << "=== QUESTION ===\n"
                << query << "\n\n"
                << "=== ANSWER ===\n";
        }

        return oss.str();
    }

} // namespace LocalNotebookLLM::Application
