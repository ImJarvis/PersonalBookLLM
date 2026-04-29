# PersonalBookLLM

[![Windows Native](https://img.shields.all_python.org/badge/Platform-Windows-0078D4?style=flat&logo=windows)](https://github.com/ImJarvis/PersonalBookLLM)
[![Fully Offline](https://img.shields.all_python.org/badge/Privacy-100%25_Offline-00AD7C?style=flat)](https://github.com/ImJarvis/PersonalBookLLM)
[![C++ Native](https://img.shields.all_python.org/badge/Language-C%2B%2B-00599C?style=flat&logo=c%2B%2B)](https://github.com/ImJarvis/PersonalBookLLM)

**PersonalBookLLM** is a high-performance, privacy-first alternative to cloud-based document intelligence. It allows you to chat with your local documents (PDFs, DOCX, Text) with absolute privacy, zero latency, and page-accurate citations.

---

## 🌟 Key Features

- **🔒 100% Offline & Private**: No data ever leaves your machine. Inference, indexing, and retrieval are performed entirely locally.
- **⚡ Blazing Fast**: Engineered in native C++ with a custom Dear ImGui + DirectX 11 interface. Cold startup in <100ms.
- **🧠 Dual-Model Intelligence**: Orchestrates a 1B "Worker" for fast reasoning and a 4B "Reasoner" for complex multi-page synthesis.
- **📍 Interactive Citations**: Every answer is grounded in your documents with precise, clickable page numbers to verify information instantly.
- **🎨 Smart Markdown UI**: Beautiful, readable text rendering with automatic bolding, proper paragraph spacing, and copy-to-clipboard functionality.
- **🧹 Ephemeral Sessions**: Zero left-over state. The local database and document memory are completely wiped when you close the application, ensuring a true clean slate every time.
- **🤖 Auto-Summarization**: Automatically generates a bulleted summary the moment your document is ingested so you can grasp the context immediately.
- **🔍 Structural-Hybrid Retrieval**: Uses advanced SQLite FTS5 BM25 combined with structural document indexing for accuracy that outperforms standard vector databases.

## 🚀 Why PersonalBookLLM?

Most "Chat with your PDF" tools are either slow, cloud-dependent, or resource-heavy. PersonalBookLLM is built to be a permanent utility on your Windows machine:
- **Zero VRAM swapping**: Optimized memory orchestration for low-end to high-end GPUs.
- **Mica Aesthetics & Polish**: Seamlessly integrates with Windows 11 design language with professional prose formatting.
- **Native Parsing**: Fast, high-accuracy extraction from DOCX and PDF without bloated third-party dependencies.

## 🛠️ Tech Stack

- **UI**: Dear ImGui / DirectX 11 / DWM Mica
- **Inference**: llama.cpp (GGUF)
- **Indexing**: SQLite FTS5 (Full Text Search)
- **Parsing**: pugixml & minizip (Custom DOCX), MuPDF (PDF)
- **Build**: CMake / MSVC

---

## 📦 Getting Started

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/ImJarvis/PersonalBookLLM.git
   cd PersonalBookLLM
   ```
2. **Setup Submodules**:
   ```bash
   git submodule update --init --recursive
   ```
3. **Build**:
   Open the folder in Visual Studio 2022 or use CMake CLI with the Release configuration.

---

## 🤝 Contributing

Contributions are welcome! Whether it's adding support for new file formats or optimizing the inference pipeline, please feel free to open a Pull Request.

## 📄 License

Distributed under the MIT License. See `LICENSE.txt` for more information.