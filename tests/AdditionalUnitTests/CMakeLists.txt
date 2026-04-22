add_executable(AdditionalUnitTests
    QueryAnalyzerAdditionalTests.cpp
    ContextAssemblerAdditionalTests.cpp
    TextQualityValidatorAdditionalTests.cpp
)

target_include_directories(AdditionalUnitTests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(AdditionalUnitTests PRIVATE
    Application
    Infrastructure
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(AdditionalUnitTests)
