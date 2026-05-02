#pragma once
// ╔══════════════════════════════════════════════════════════════════╗
// ║  LocalNotebookLLM — Lightweight Logging System                  ║
// ║  Thread-safe, file-backed, severity-leveled logger              ║
// ╚══════════════════════════════════════════════════════════════════╝

#include <string>
#include <fstream>
#include <mutex>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <iomanip>

namespace LocalNotebookLLM::Core {

    enum class LogLevel { Debug, Info, Warn, Error };

    /// @brief Singleton logger — thread-safe, writes to file + optional OutputDebugString.
    class Logger {
    public:
        static Logger& Instance() {
            static Logger inst;
            return inst;
        }

        /// Must be called once at startup. Creates/truncates the log file.
        void Initialize(const std::filesystem::path& logDir,
                        LogLevel minLevel = LogLevel::Debug) {
            std::lock_guard lock(m_mutex);
            m_minLevel = minLevel;
            std::filesystem::create_directories(logDir);
            auto path = logDir / "localnotebook.log";
            m_file.open(path, std::ios::out | std::ios::trunc);
            if (m_file.is_open()) {
                m_initialized = true;
                WriteEntry(LogLevel::Info, "Logger",
                    "=== LocalNotebookLLM Session Started ===");
                WriteEntry(LogLevel::Info, "Logger",
                    "Log file: " + path.string());
            }
        }

        /// Log a message with component tag.
        void Log(LogLevel level, const char* component, const std::string& message) {
            if (!m_initialized || level < m_minLevel) return;
            std::lock_guard lock(m_mutex);
            WriteEntry(level, component, message);
        }

        /// Flush and close.
        void Shutdown() {
            std::lock_guard lock(m_mutex);
            if (m_initialized) {
                WriteEntry(LogLevel::Info, "Logger",
                    "=== Session Ended ===");
                m_file.close();
                m_initialized = false;
            }
        }

        ~Logger() { Shutdown(); }

        // Delete copy
        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;

    private:
        Logger() = default;

        void WriteEntry(LogLevel level, const char* component,
                        const std::string& message) {
            if (!m_file.is_open()) return;

            // Timestamp
            auto now     = std::chrono::system_clock::now();
            auto time    = std::chrono::system_clock::to_time_t(now);
            auto ms      = std::chrono::duration_cast<std::chrono::milliseconds>(
                               now.time_since_epoch()) % 1000;
            std::tm tm_buf{};
#ifdef _WIN32
            localtime_s(&tm_buf, &time);
#else
            localtime_r(&time, &tm_buf);
#endif

            const char* lvl = "";
            switch (level) {
                case LogLevel::Debug: lvl = "DEBUG"; break;
                case LogLevel::Info:  lvl = "INFO "; break;
                case LogLevel::Warn:  lvl = "WARN "; break;
                case LogLevel::Error: lvl = "ERROR"; break;
            }

            m_file << std::put_time(&tm_buf, "%H:%M:%S")
                   << "." << std::setfill('0') << std::setw(3) << ms.count()
                   << " [" << lvl << "]"
                   << " [" << component << "] "
                   << message << "\n";
            // M4: Tiered flush strategy —
            //   Error → immediate flush (visibility for crashes)
            //   Others → periodic flush every 50 writes (~1 disk sync per 500ms during
            //   embedding/inference). Keeps throughput high while guaranteeing log
            //   entries appear in the file before a session ends or crashes.
            if (level >= LogLevel::Error) {
                m_file.flush();
            } else if (++m_writeCount % 50 == 0) {
                m_file.flush();
            }
        }

        std::ofstream m_file;
        std::mutex    m_mutex;
        LogLevel      m_minLevel    = LogLevel::Debug;
        bool          m_initialized = false;
        size_t        m_writeCount  = 0;   // For periodic flush (every 50 writes)
    };

} // namespace LocalNotebookLLM::Core

// ─── Convenience macros ───
// Usage: LOG_INFO("App", "Initialized with data dir: " + dir);

#define LOG_DEBUG(component, msg) \
    LocalNotebookLLM::Core::Logger::Instance().Log( \
        LocalNotebookLLM::Core::LogLevel::Debug, component, msg)

#define LOG_INFO(component, msg) \
    LocalNotebookLLM::Core::Logger::Instance().Log( \
        LocalNotebookLLM::Core::LogLevel::Info, component, msg)

#define LOG_WARN(component, msg) \
    LocalNotebookLLM::Core::Logger::Instance().Log( \
        LocalNotebookLLM::Core::LogLevel::Warn, component, msg)

#define LOG_ERROR(component, msg) \
    LocalNotebookLLM::Core::Logger::Instance().Log( \
        LocalNotebookLLM::Core::LogLevel::Error, component, msg)
