cmake_minimum_required(VERSION 3.20)
project(mayajiva VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Core library (header-only for now) ────────────────────────────
add_library(mayajiva_core INTERFACE)
target_include_directories(mayajiva_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ── Dependencies via FetchContent ─────────────────────────────────
include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2
)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)

FetchContent_MakeAvailable(Catch2 nlohmann_json)

# ── Tests ─────────────────────────────────────────────────────────
enable_testing()

set(REFERENCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/reference")

function(add_mayajiva_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE
        mayajiva_core
        Catch2::Catch2WithMain
        nlohmann_json::nlohmann_json
    )
    target_compile_definitions(${name} PRIVATE
        REFERENCE_DIR="${REFERENCE_DIR}"
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_mayajiva_test(test_compass)
add_mayajiva_test(test_ring_attractor)
add_mayajiva_test(test_cpu4)
add_mayajiva_test(test_landscape)
add_mayajiva_test(test_bug)

# Include Catch2 CMake integration for CTest
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

catch_discover_tests(test_compass)
catch_discover_tests(test_ring_attractor)
catch_discover_tests(test_cpu4)
catch_discover_tests(test_landscape)
catch_discover_tests(test_bug)
