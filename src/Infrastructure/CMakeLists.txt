# Infrastructure: Concrete implementations
add_library(Infrastructure STATIC
    Parsing/StructureDetector.cpp
    Parsing/DocxParser.cpp
    Parsing/PdfParser.cpp
    Parsing/CompositeParser.cpp
    Parsing/TextQualityValidator.cpp
    Indexing/DatabaseManager.cpp
    Indexing/SqliteFts5Indexer.cpp
    Search/StructuralNavigator.cpp
    Search/HybridSearchEngine.cpp
    LLM/LlamaCppProvider.cpp
    LLM/LlamaCppEmbeddingProvider.cpp
    LLM/MemoryOrchestrator.cpp
    LLM/GPUDetector.cpp
    Storage/SqliteDocumentRepository.cpp
)

target_include_directories(Infrastructure PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(Infrastructure
    PUBLIC
        Core
    PRIVATE
        pugixml::pugixml
        nlohmann_json::nlohmann_json
)

if (TARGET unofficial::sqlite3::sqlite3)
    target_link_libraries(Infrastructure PUBLIC unofficial::sqlite3::sqlite3)
else()
    target_link_libraries(Infrastructure PUBLIC SQLite::SQLite3)
endif()

# Conditionally link llama.cpp if built
if(TARGET llama)
    target_link_libraries(Infrastructure PRIVATE llama)
    target_compile_definitions(Infrastructure PRIVATE HAS_LLAMA_CPP=1)
endif()

# Link minizip if available
if(TARGET unofficial::minizip::minizip)
    target_link_libraries(Infrastructure PRIVATE unofficial::minizip::minizip)
elseif(TARGET minizip::minizip)
    target_link_libraries(Infrastructure PRIVATE minizip::minizip)
elseif(TARGET MINIZIP::minizip)
    target_link_libraries(Infrastructure PRIVATE MINIZIP::minizip)
endif()

# Link MuPDF if available (enables PDF parsing)
find_package(unofficial-libmupdf CONFIG QUIET)
if(unofficial-libmupdf_FOUND)
    target_link_libraries(Infrastructure PRIVATE unofficial::libmupdf::libmupdf)
    target_compile_definitions(Infrastructure PRIVATE HAS_MUPDF=1)
    message(STATUS "MuPDF found — PDF parsing enabled")
else()
    message(STATUS "MuPDF NOT found — PDF parsing disabled (add 'libmupdf' to vcpkg.json)")
endif()

# Windows system libraries for GPU detection + DWM
if(WIN32)
    target_link_libraries(Infrastructure PRIVATE dxgi.lib dwmapi.lib)
endif()
