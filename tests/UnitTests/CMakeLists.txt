# UnitTests — fast, isolated, no I/O or model loading
add_executable(UnitTests
    QueryAnalyzerTests.cpp
    ContextAssemblerTests.cpp
    StructureDetectorTests.cpp
    TextQualityValidatorTests.cpp
    DatabaseManagerTests.cpp
)

target_include_directories(UnitTests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(UnitTests PRIVATE
    Application
    Infrastructure
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(UnitTests)
