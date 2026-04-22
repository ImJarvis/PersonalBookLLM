#include "UI/App.h"
#include "Infrastructure/Indexing/DatabaseManager.h"
#include "Infrastructure/Indexing/SqliteFts5Indexer.h"
#include "Infrastructure/Storage/SqliteDocumentRepository.h"
#include "Infrastructure/Parsing/CompositeParser.h"
#include "Infrastructure/Parsing/DocxParser.h"
#include "Infrastructure/Parsing/PdfParser.h"
#include "Infrastructure/Search/HybridSearchEngine.h"
#include "Infrastructure/Search/StructuralNavigator.h"
#include "Infrastructure/LLM/LlamaCppProvider.h"
#include "Infrastructure/LLM/LlamaCppEmbeddingProvider.h"
#include "Infrastructure/LLM/MemoryOrchestrator.h"
#include "Application/ContextAssembler.h"
#include "Core/Log.h"
#include <algorithm>

// Compile-time defaults from CMake (overridable)
#ifndef LOCALNOTEBOOK_MODELS_DIR
#define LOCALNOTEBOOK_MODELS_DIR "models"
#endif
#ifndef LOCALNOTEBOOK_WORKER_MODEL
#define LOCALNOTEBOOK_WORKER_MODEL "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
#endif
#ifndef LOCALNOTEBOOK_REASONER_MODEL
#define LOCALNOTEBOOK_REASONER_MODEL "gemma-3-4b-it-Q4_K_M.gguf"
#endif
#ifndef LOCALNOTEBOOK_EMBEDDING_MODEL
#define LOCALNOTEBOOK_EMBEDDING_MODEL "nomic-embed-text-v1.5.Q8_0.gguf"
#endif

namespace LocalNotebookLLM::UI {

    App::App() {
        LOG_DEBUG("App", "App constructor called");
    }

    App::~App() {
        LOG_DEBUG("App", "App destructor called");
    }

    bool App::Initialize(const std::filesystem::path& dataDir) {
        LOG_INFO("App", "=== Initialize() begin ===");
        m_dataDir   = dataDir;
        m_modelsDir = std::filesystem::path(LOCALNOTEBOOK_MODELS_DIR);

        // If relative path, resolve relative to executable's parent
        if (m_modelsDir.is_relative()) {
            m_modelsDir = std::filesystem::current_path() / m_modelsDir;
        }

        LOG_INFO("App", "Data dir:   " + m_dataDir.string());
        LOG_INFO("App", "Models dir: " + m_modelsDir.string());

        // Ensure directories exist
        std::filesystem::create_directories(m_dataDir);
        std::filesystem::create_directories(m_modelsDir);

        // ─── Wire up infrastructure (Dependency Injection) ───
        std::string dbPath = (m_dataDir / "notebook.db").string();
        LOG_INFO("App", "Opening database: " + dbPath);
        m_db = std::make_shared<Infrastructure::DatabaseManager>(dbPath);
        LOG_INFO("App", "Database initialized with WAL + FTS5");

        // ─── Embedder ───
        m_embedder = std::make_shared<Infrastructure::LlamaCppEmbeddingProvider>();
        
        // Indexer + Repository
        auto indexer = std::make_shared<Infrastructure::SqliteFts5Indexer>(m_db, m_embedder);
        auto repo    = std::make_shared<Infrastructure::SqliteDocumentRepository>(m_db);
        LOG_INFO("App", "Indexer + Repository created");

        // Parser (CompositeParser with DOCX + PDF support)
        auto composite = std::make_unique<Infrastructure::CompositeParser>();
        composite->RegisterParser(std::make_unique<Infrastructure::DocxParser>());
        composite->RegisterParser(std::make_unique<Infrastructure::PdfParser>());
        LOG_INFO("App", "CompositeParser created (DOCX + PDF support)");

        // ─── Dual LLM Providers ───
        m_workerLLM   = std::make_shared<Infrastructure::LlamaCppProvider>();
        m_reasonerLLM = std::make_shared<Infrastructure::LlamaCppProvider>();
        LOG_INFO("App", "Dual LlamaCppProvider instances created (Worker + Reasoner)");

        // ─── Memory Orchestrator (manages Worker <-> Reasoner lifecycle) ───
        m_memoryOrch = std::make_shared<Infrastructure::MemoryOrchestrator>(
            m_workerLLM, m_reasonerLLM);
        LOG_INFO("App", "MemoryOrchestrator created");

        // Configure model paths for the orchestrator
        auto workerPath   = m_modelsDir / LOCALNOTEBOOK_WORKER_MODEL;
        auto reasonerPath = m_modelsDir / LOCALNOTEBOOK_REASONER_MODEL;

        LOG_INFO("App", "Worker model path:   " + workerPath.string() +
            " (exists: " + std::string(std::filesystem::exists(workerPath) ? "YES" : "NO") + ")");
        LOG_INFO("App", "Reasoner model path: " + reasonerPath.string() +
            " (exists: " + std::string(std::filesystem::exists(reasonerPath) ? "YES" : "NO") + ")");

        if (std::filesystem::exists(workerPath)) {
            Infrastructure::MemoryOrchestrator::ModelConfig workerCfg;
            workerCfg.modelPath      = workerPath;
            workerCfg.estimatedSizeMB = 1024;  // ~1 GB for Qwen 1.5B Q4
            workerCfg.params.contextWindowSize = 2048;
            workerCfg.params.temperature = 0.1f;
            m_memoryOrch->SetWorkerConfig(workerCfg);
            LOG_INFO("App", "Worker config set: 2048 ctx, 0.1 temp, ~1024 MB");
        }

        if (std::filesystem::exists(reasonerPath)) {
            Infrastructure::MemoryOrchestrator::ModelConfig reasonerCfg;
            reasonerCfg.modelPath       = reasonerPath;
            reasonerCfg.estimatedSizeMB = 2560;  // ~2.5 GB for Gemma 4B Q4
            reasonerCfg.params.contextWindowSize = 4096;
            reasonerCfg.params.temperature = 0.0f; // Force deterministic, strict grounding
            m_memoryOrch->SetReasonerConfig(reasonerCfg);
            LOG_INFO("App", "Reasoner config set: 4096 ctx, 0.3 temp, ~2560 MB");
        }

        // ─── DocumentService (uses Worker via MemoryOrchestrator for enrichment) ───
        m_docService = std::make_unique<Application::DocumentService>(
            std::move(composite), indexer, repo, m_memoryOrch);
        LOG_INFO("App", "DocumentService created");

        // ─── Search engine (BM25 + Structural navigation) ───
        auto navigator = std::make_shared<Infrastructure::StructuralNavigator>(m_db);
        auto search    = std::make_shared<Infrastructure::HybridSearchEngine>(indexer, navigator, m_embedder, m_db);
        LOG_INFO("App", "HybridSearchEngine created (BM25 + Structural + Semantic)");

        // ─── Context Assembler ───
        auto assembler = std::make_unique<Application::ContextAssembler>();

        // ─── InferenceService (uses Reasoner for Q&A) ───
        m_inferService = std::make_unique<Application::InferenceService>(
            m_reasonerLLM, search, std::move(assembler), m_memoryOrch);
        LOG_INFO("App", "InferenceService created");

        // Load initial state
        RefreshDocumentList();
        LOG_INFO("App", "Document list loaded: " + std::to_string(m_state.documents.size()) + " docs");

        // ─── Auto-detect and load BOTH models ───
        LOG_INFO("App", "Starting model auto-detection...");
        AutoDetectAndLoadModels();

        LOG_INFO("App", "=== Initialize() complete ===");
        return true;
    }

    void App::AutoDetectAndLoadModels() {
        LOG_INFO("App", "AutoDetectAndLoadModels() begin");
        m_state.workerStatus   = "Scanning for models...";
        m_state.reasonerStatus = "Scanning for models...";
        m_state.embedStatus    = "Scanning for models...";

        // ─── Worker Model (Qwen) ───
        const std::vector<std::string> workerCandidates = {
            LOCALNOTEBOOK_WORKER_MODEL,
        };

        std::filesystem::path workerPath;
        for (const auto& name : workerCandidates) {
            auto path = m_modelsDir / name;
            LOG_DEBUG("App", "  Checking worker candidate: " + path.string());
            if (std::filesystem::exists(path)) {
                workerPath = path;
                LOG_INFO("App", "  Worker model found: " + name);
                break;
            }
        }

        // Fallback: scan for any small model
        if (workerPath.empty()) {
            LOG_INFO("App", "  No default worker — scanning all .gguf files...");
            auto models = ScanAvailableModels();
            for (const auto& m : models) {
                if (m.find("qwen") != std::string::npos ||
                    m.find("1.5b") != std::string::npos ||
                    m.find("1b") != std::string::npos) {
                    workerPath = m_modelsDir / m;
                    LOG_INFO("App", "  Worker fallback found: " + m);
                    break;
                }
            }
        }

        if (!workerPath.empty()) {
            m_state.workerPath   = workerPath.string();
            m_state.workerStatus = "Auto-loading: " + workerPath.filename().string();
            LOG_INFO("App", "Auto-loading Worker: " + workerPath.filename().string());
            LoadWorkerModel(workerPath);
        } else {
            m_state.workerStatus = "No worker model found";
            LOG_WARN("App", "No worker model found in: " + m_modelsDir.string());
        }

        // ─── Reasoner Model (Gemma) ───
        const std::vector<std::string> reasonerCandidates = {
            "gemma-4-e2b-it-Q4_K_M.gguf",
            LOCALNOTEBOOK_REASONER_MODEL,
        };

        std::filesystem::path reasonerPath;
        for (const auto& name : reasonerCandidates) {
            auto path = m_modelsDir / name;
            LOG_DEBUG("App", "  Checking reasoner candidate: " + path.string());
            if (std::filesystem::exists(path)) {
                reasonerPath = path;
                LOG_INFO("App", "  Reasoner model found: " + name);
                break;
            }
        }

        // Fallback: scan all .gguf files
        if (reasonerPath.empty()) {
            LOG_INFO("App", "  No default reasoner — scanning all .gguf files...");
            auto models = ScanAvailableModels();
            for (const auto& m : models) {
                if (m.find("gemma") != std::string::npos ||
                    m.find("e2b") != std::string::npos ||
                    m.find("e4b") != std::string::npos) {
                    reasonerPath = m_modelsDir / m;
                    LOG_INFO("App", "  Reasoner fallback found: " + m);
                    break;
                }
            }
        }

        if (!reasonerPath.empty()) {
            m_state.reasonerPath   = reasonerPath.string();
            m_state.reasonerStatus = "Auto-loading: " + reasonerPath.filename().string();
            LOG_INFO("App", "Auto-loading Reasoner: " + reasonerPath.filename().string());
            LoadReasonerModel(reasonerPath);
        } else {
            m_state.reasonerStatus = "No reasoner model found";
            LOG_WARN("App", "No reasoner model found in: " + m_modelsDir.string());
        }

        // ─── Embed Model ───
        const std::vector<std::string> embedCandidates = {
            LOCALNOTEBOOK_EMBEDDING_MODEL,
            "bge-micro-v2.gguf",
            "nomic-embed-text-v1.5.Q8_0.gguf"
        };
        
        std::filesystem::path embedPath;
        for (const auto& name : embedCandidates) {
            auto path = m_modelsDir / name;
            if (std::filesystem::exists(path)) {
                embedPath = path;
                LOG_INFO("App", "  Embed model found: " + name);
                break;
            }
        }
        if (std::filesystem::exists(embedPath)) {
            m_state.embedName   = embedPath.filename().string();
            m_state.embedPath   = embedPath.string();
            m_state.embedStatus = "Ready";
            m_state.embedLoaded = true;
            m_embedder->LoadModel(embedPath);
            LOG_INFO("App", "Loaded embedding model: " + m_state.embedName);
        } else {
            LOG_WARN("App", "No embedding models found");
            m_state.embedStatus = "Missing embedding models (semantic search disabled)";
            m_state.embedLoaded = false;
        }

        LOG_INFO("App", "AutoDetectAndLoadModels() complete");
    }

    void App::IngestDocument(const std::filesystem::path& filePath) {
        if (m_state.ingesting) {
            LOG_WARN("App", "IngestDocument skipped — already ingesting");
            return;
        }

        LOG_INFO("App", "IngestDocument() begin: " + filePath.string());
        m_state.ingesting         = true;
        m_state.ingestionProgress = 0.0f;
        m_state.ingestionStatus   = "Starting ingestion...";
        m_state.lastIngestionError = "";  // Clear previous error so banner doesn't show during new ingestion


        m_ingestionFuture = std::async(std::launch::async, [this, filePath]() {
            try {
                LOG_INFO("App", "[async] Ingestion thread started for: " + filePath.filename().string());
                auto result = m_docService->IngestDocument(filePath,
                    [this](const Application::IngestionProgress& p) {
                        std::lock_guard lock(m_stateMutex);
                        m_state.ingestionProgress = static_cast<float>(p.progressPct);
                        m_state.ingestionStatus   = p.message;
                    });

                std::lock_guard lock(m_stateMutex);
                if (result) {
                    m_state.ingestionStatus   = "Done: " + result->filename +
                        " (" + std::to_string(result->pageCount) + " pages)";
                    m_state.lastIngestionError = "";  // Clear any previous error on success
                } else {
                    // Store error persistently so the UI shows it even after ingesting=false
                    m_state.ingestionStatus    = "Failed";
                    m_state.lastIngestionError = result.error();
                    LOG_ERROR("App", "[async] Ingestion FAILED: " + result.error());
                }
            } catch (const std::exception& e) {
                std::lock_guard lock(m_stateMutex);
                m_state.ingestionStatus    = "Exception";
                m_state.lastIngestionError = std::string(e.what());
                LOG_ERROR("App", "[async] Exception in Ingestion: " + std::string(e.what()));
            } catch (...) {
                std::lock_guard lock(m_stateMutex);
                m_state.ingestionStatus    = "Unknown error";
                m_state.lastIngestionError = "An unknown fatal error occurred during ingestion.";
            }
            
            std::lock_guard finalLock(m_stateMutex);
            m_state.ingesting = false;
            RefreshDocumentList();
        });

    }

    void App::RemoveDocument(int64_t docId) {
        LOG_INFO("App", "RemoveDocument() id=" + std::to_string(docId));
        m_docService->RemoveDocument(docId);
        RefreshDocumentList();
    }

    void App::AskQuestion(const std::string& query) {
        if (m_state.generating || query.empty()) {
            LOG_WARN("App", "AskQuestion skipped — generating=" +
                std::string(m_state.generating ? "true" : "false") +
                " empty=" + std::string(query.empty() ? "true" : "false"));
            return;
        }

        LOG_INFO("App", "AskQuestion() query: \"" + query + "\"");
        m_state.chatHistory.push_back({ChatMessage::Role::User, query, {}, 0, 0});
        m_state.generating = true;
        m_state.generationStatus = "Searching documents...";
        m_state.streamingAnswer.clear();

        m_generationFuture = std::async(std::launch::async, [this, query]() {
            try {
                LOG_INFO("App", "[async] Q&A thread started");
                auto result = m_inferService->AskQuestionStreaming(query,
                    [this](const std::string& token) {
                        std::lock_guard lock(m_stateMutex);
                        m_state.streamingAnswer += token;
                    },
                    [this](const Application::AnswerProgress& p) {
                        std::lock_guard lock(m_stateMutex);
                        switch (p.stage) {
                            case Application::AnswerProgress::Stage::Searching: m_state.generationStatus = "Searching documents..."; break;
                            case Application::AnswerProgress::Stage::Assembling: m_state.generationStatus = "Assembling context..."; break;
                            case Application::AnswerProgress::Stage::Generating: m_state.generationStatus = "Generating answer..."; break;
                            case Application::AnswerProgress::Stage::Validating: m_state.generationStatus = "Validating citations..."; break;
                            case Application::AnswerProgress::Stage::Complete: m_state.generationStatus = "Complete"; break;
                        }
                    });

                std::lock_guard lock(m_stateMutex);
                if (result) {
                    m_state.chatHistory.push_back({ ChatMessage::Role::Assistant, result->answerText, result->citations, result->generationTimeMs, result->sectionsSearched });
                } else {
                    m_state.chatHistory.push_back({ ChatMessage::Role::System, "Error: " + result.error(), {}, 0, 0 });
                }
            } catch (const std::exception& e) {
                std::lock_guard lock(m_stateMutex);
                m_state.chatHistory.push_back({ ChatMessage::Role::System, "Exception: " + std::string(e.what()), {}, 0, 0 });
                LOG_ERROR("App", "[async] Exception in Q&A: " + std::string(e.what()));
            }

            std::lock_guard finalLock(m_stateMutex);
            m_state.generating = false;
            m_state.streamingAnswer.clear();
        });
    }

    void App::LoadReasonerModel(const std::filesystem::path& modelPath) {
        LOG_INFO("App", "LoadReasonerModel() path: " + modelPath.string());
        m_state.reasonerStatus = "Loading " + modelPath.filename().string() + "...";

        m_reasonerLoadFuture = std::async(std::launch::async, [this, modelPath]() {
            LOG_INFO("App", "[async] Reasoner load thread started: " + modelPath.filename().string());
            Core::ModelParams params;
            params.contextWindowSize = 4096;
            params.temperature = 0.3f;
            auto result = m_reasonerLLM->LoadModel(modelPath, params);

            std::lock_guard lock(m_stateMutex);
            if (result) {
                m_state.reasonerLoaded = true;
                m_state.reasonerName   = modelPath.filename().string();
                m_state.reasonerPath   = modelPath.string();
                m_state.reasonerStatus = "Loaded: " + m_state.reasonerName;
                LOG_INFO("App", "[async] Reasoner LOADED: " + m_state.reasonerName);
            } else {
                m_state.reasonerStatus = "Failed: " + result.error().message;
                LOG_ERROR("App", "[async] Reasoner FAILED: " + result.error().message);
            }
        });
    }

    void App::LoadWorkerModel(const std::filesystem::path& modelPath) {
        LOG_INFO("App", "LoadWorkerModel() path: " + modelPath.string());
        m_state.workerStatus = "Loading " + modelPath.filename().string() + "...";

        m_workerLoadFuture = std::async(std::launch::async, [this, modelPath]() {
            LOG_INFO("App", "[async] Worker load thread started: " + modelPath.filename().string());
            Core::ModelParams params;
            params.contextWindowSize = 2048;
            params.temperature = 0.1f;
            auto result = m_workerLLM->LoadModel(modelPath, params);

            std::lock_guard lock(m_stateMutex);
            if (result) {
                m_state.workerLoaded = true;
                m_state.workerName   = modelPath.filename().string();
                m_state.workerPath   = modelPath.string();
                m_state.workerStatus = "Loaded: " + m_state.workerName;
                LOG_INFO("App", "[async] Worker LOADED: " + m_state.workerName);
            } else {
                m_state.workerStatus = "Failed: " + result.error().message;
                LOG_ERROR("App", "[async] Worker FAILED: " + result.error().message);
            }
        });
    }

    void App::Update() {
        auto check = [](std::future<void>& f) {
            if (f.valid() &&
                f.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                f.get();
            }
        };
        check(m_ingestionFuture);
        check(m_generationFuture);
        check(m_reasonerLoadFuture);
        check(m_workerLoadFuture);
    }

    std::vector<std::string> App::ScanAvailableModels() const {
        std::vector<std::string> models;
        if (!std::filesystem::exists(m_modelsDir)) {
            LOG_WARN("App", "ScanAvailableModels: dir not found: " + m_modelsDir.string());
            return models;
        }

        for (const auto& entry : std::filesystem::directory_iterator(m_modelsDir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                models.push_back(entry.path().filename().string());
            }
        }
        LOG_INFO("App", "ScanAvailableModels: found " + std::to_string(models.size()) + " .gguf files");

        // Sort: put reasoner first, worker second, then alphabetical
        std::sort(models.begin(), models.end(), [](const std::string& a, const std::string& b) {
            auto isReasoner = [](const std::string& s) {
                return s.find("gemma") != std::string::npos ||
                       s.find("e2b") != std::string::npos ||
                       s.find("e4b") != std::string::npos;
            };
            if (isReasoner(a) != isReasoner(b)) return isReasoner(a);
            return a < b;
        });

        return models;
    }

    void App::RefreshDocumentList() {
        m_state.documents = m_docService->ListDocuments();
        LOG_DEBUG("App", "RefreshDocumentList: " + std::to_string(m_state.documents.size()) + " docs");
    }

} // namespace LocalNotebookLLM::UI
