# FunctionalTests — end-to-end, may require model files and test PDFs
add_executable(FunctionalTests
    Fts5SearchTests.cpp
)

target_include_directories(FunctionalTests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(FunctionalTests PRIVATE
    Application
    Infrastructure
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(FunctionalTests)
