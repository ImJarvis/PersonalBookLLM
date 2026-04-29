#pragma once
#include "Application/DocumentService.h"
#include "Application/InferenceService.h"
#include "Core/Models/DocumentMetadata.h"
#include "Core/Models/CitedAnswer.h"
#include <memory>
#include <string>
#include <vector>
#include <filesystem>
#include <mutex>
#include <future>
#include <deque>
#include <chrono>

namespace LocalNotebookLLM::Core {
    class IEmbeddingProvider;
}

namespace LocalNotebookLLM::Infrastructure {
    class DatabaseManager;
    class LlamaCppProvider;
    class MemoryOrchestrator;
}

namespace LocalNotebookLLM::UI {

    /// A single message in the chat history.
    struct ChatMessage {
        enum class Role { User, Assistant, System };
        Role        role;
        std::string text;
        std::vector<Core::Citation> citations;
        double      generationTimeMs = 0.0;
        int         sectionsSearched = 0;
        bool        isHidden = false;
    };

    /// Application state for the UI.
    struct AppState {
        // ─── Documents ───
        std::vector<Core::DocumentMetadata> documents;
        bool ingesting = false;
        std::string ingestionStatus;
        float ingestionProgress = 0.0f;
        // Persists after ingestion completes so the user can see the error even
        // when the ingestion status bar has been hidden. Cleared on next successful ingest.
        std::string lastIngestionError;

        // ─── Chat ───
        std::deque<ChatMessage> chatHistory;
        char queryBuffer[1024] = {};
        bool generating = false;
        bool generationCancelled = false;
        std::string generationStatus;
        std::string streamingAnswer;  // Partial answer during streaming
        std::chrono::steady_clock::time_point lastActivityTime; // Watchdog timer
        std::chrono::steady_clock::time_point generationStartTime; // When generation started
        double elapsedGenerationSec = 0.0; // Updated every frame for UI display

        // ─── Reasoner Model (Q&A) ───
        bool reasonerLoaded = false;
        std::string reasonerName;
        std::string reasonerPath;
        std::string reasonerStatus;

        // ─── Worker Model (Indexing) ───
        bool workerLoaded = false;
        std::string workerName;
        std::string workerPath;
        std::string workerStatus;

        // ─── Embed Model (Semantic Search) ───
        bool embedLoaded = false;
        std::string embedName;
        std::string embedPath;
        std::string embedStatus;

        // ─── Combined convenience ───
        bool IsAnyModelLoaded() const { return reasonerLoaded || workerLoaded || embedLoaded; }
        bool IsReasonerReady()  const { return reasonerLoaded; }
        bool IsWorkerReady()    const { return workerLoaded; }

        // ─── UI ───
        enum class ActivePage { Chat, Library, Settings };
        ActivePage activePage = ActivePage::Library;
        bool showAbout = false;
    };

    /// @brief Main application orchestrator.
    /// Owns all services, manages async operations, provides state for UI rendering.
    class App {
    public:
        App();
        ~App();

        /// Initialize all services (call once at startup).
        bool Initialize(const std::filesystem::path& dataDir);

        /// Get mutable state for UI rendering.
        AppState& GetState() { return m_state; }

        // ─── Actions (called from UI) ───

        /// Drop a document file for ingestion (async).
        void IngestDocument(const std::filesystem::path& filePath);

        /// Remove a document by ID.
        void RemoveDocument(int64_t docId);

        /// Ask a question (async, populates chat history).
        void AskQuestion(const std::string& query);

        /// Cancel the current generation (cooperative — signals the LLM to stop
        /// at the next safe checkpoint; does NOT destroy the model).
        void CancelGeneration();

        /// Load the Reasoner model (async). Used for Q&A.
        void LoadReasonerModel(const std::filesystem::path& modelPath);

        /// Load the Worker model (async). Used for indexing enrichment.
        void LoadWorkerModel(const std::filesystem::path& modelPath);

        /// Legacy: Load model (defaults to reasoner).
        void LoadModel(const std::filesystem::path& modelPath) { LoadReasonerModel(modelPath); }

        /// Poll for async operation completion (call each frame).
        void Update();

        /// Scan the models/ directory for available GGUF files.
        std::vector<std::string> ScanAvailableModels() const;

        /// Get the models directory path.
        const std::filesystem::path& GetModelsDir() const { return m_modelsDir; }

    private:
        AppState m_state;
        std::filesystem::path m_dataDir;
        std::filesystem::path m_modelsDir;

        // Services (owned)
        std::unique_ptr<Application::DocumentService>  m_docService;
        std::unique_ptr<Application::InferenceService>  m_inferService;

        // Dual-model providers (shared with MemoryOrchestrator)
        std::shared_ptr<Infrastructure::LlamaCppProvider> m_workerLLM;
        std::shared_ptr<Infrastructure::LlamaCppProvider> m_reasonerLLM;
        std::shared_ptr<Core::IEmbeddingProvider>         m_embedder;
        std::shared_ptr<Infrastructure::MemoryOrchestrator> m_memoryOrch;

        // Shared infrastructure (kept alive for service lifetime)
        std::shared_ptr<Infrastructure::DatabaseManager> m_db;

        // Async futures
        std::future<void> m_ingestionFuture;
        std::future<void> m_generationFuture;
        std::vector<std::future<void>> m_abandonedFutures; // Keep hung threads here so we don't block the UI thread on destructor assignment
        std::future<void> m_reasonerLoadFuture;
        std::future<void> m_workerLoadFuture;

        std::mutex m_stateMutex;
        std::string m_pendingAutoQuery;

        void RefreshDocumentList();
        void AutoDetectAndLoadModels();
    };

} // namespace LocalNotebookLLM::UI
