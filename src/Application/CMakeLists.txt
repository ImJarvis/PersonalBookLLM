# Application: Service layer — depends only on Core interfaces
add_library(Application STATIC
    DocumentService.cpp
    InferenceService.cpp
    QueryAnalyzer.cpp
    ContextAssembler.cpp
)

target_include_directories(Application PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(Application PUBLIC Core)
