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
        auto genResult = m_impl->llm->GenerateStreaming(prompt, 1024, tokenCallback);

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
        // Extract [Page N] references from the answer text
        std::regex pageRegex(R"(\[Page\s+(\d+)\])");
        std::smatch match;
        std::string text = answer.answerText;
        std::set<int> referencedPages;

        auto it = std::sregex_iterator(text.begin(), text.end(), pageRegex);
        auto end = std::sregex_iterator();
        for (; it != end; ++it) {
            referencedPages.insert(std::stoi((*it)[1].str()));
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
        std::ostringstream oss;
        oss << "You are a precise document analysis assistant. "
            << "Answer questions using ONLY the provided document context. "
            << "Always cite your sources using [Page X] notation. "
            << "If the context does not contain enough information, say so.\n\n"
            << ctx.formattedContext
            << "=== QUESTION ===\n"
            << query << "\n\n"
            << "=== ANSWER ===\n";
        return oss.str();
    }

} // namespace LocalNotebookLLM::Application
