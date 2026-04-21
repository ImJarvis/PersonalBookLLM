#pragma once
#include "Core/Interfaces/ILLMProvider.h"
#include "Core/Interfaces/IHybridSearch.h"
#include "Core/Interfaces/IContextAssembler.h"
#include "Core/Interfaces/IMemoryOrchestrator.h"
#include "Core/Models/CitedAnswer.h"
#include "Core/Models/ModelStatus.h"
#include <memory>
#include <string>
#include <expected>
#include <functional>

namespace LocalNotebookLLM::Application {

    struct AnswerProgress {
        enum class Stage { Searching, Assembling, Generating, Validating, Complete };
        Stage       stage;
        std::string partialAnswer;
    };

    using AnswerProgressCallback = std::function<void(const AnswerProgress&)>;

    /// @brief Orchestrates the query → search → assemble → generate pipeline.
    class InferenceService {
    public:
        InferenceService(
            std::shared_ptr<Core::ILLMProvider> llmProvider,
            std::shared_ptr<Core::IHybridSearch> search,
            std::unique_ptr<Core::IContextAssembler> assembler,
            std::shared_ptr<Core::IMemoryOrchestrator> memoryOrch = nullptr
        );
        ~InferenceService();

        InferenceService(const InferenceService&) = delete;
        InferenceService& operator=(const InferenceService&) = delete;
        InferenceService(InferenceService&&) noexcept;
        InferenceService& operator=(InferenceService&&) noexcept;

        [[nodiscard]]
        std::expected<Core::CitedAnswer, std::string>
        AskQuestion(const std::string& query, AnswerProgressCallback progress = nullptr);

        [[nodiscard]]
        std::expected<Core::CitedAnswer, std::string>
        AskQuestionStreaming(const std::string& query,
                            Core::StreamCallback tokenCallback,
                            AnswerProgressCallback progress = nullptr);

        [[nodiscard]]
        std::expected<void, std::string>
        LoadModel(const std::filesystem::path& modelPath, const Core::ModelParams& params = {});

        [[nodiscard]] Core::ModelStatus GetModelStatus() const;
        [[nodiscard]] bool IsReady() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;

        /// Validate that all citations in the answer reference provided context.
        static void ValidateCitations(Core::CitedAnswer& answer,
                                       const Core::AssembledContext& ctx);

        /// Build the full prompt from assembled context + user query.
        static std::string BuildPrompt(const Core::AssembledContext& ctx,
                                        const std::string& query);
    };

} // namespace LocalNotebookLLM::Application
