# Core: Header-only domain layer (zero dependencies)
add_library(Core INTERFACE)
target_include_directories(Core INTERFACE
    ${CMAKE_SOURCE_DIR}/src
)
