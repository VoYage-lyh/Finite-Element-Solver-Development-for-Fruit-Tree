cmake_minimum_required(VERSION 3.21)

project(
    OrchardVibrationSolver
    VERSION 0.1.0
    DESCRIPTION "Dedicated fast solver for orchard tree vibration analysis"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ORCHARD_BUILD_TESTS "Build orchard solver tests" ON)

add_library(
    orchard_solver
    src/OrchardModel.cpp
    src/geometry_topology/TreeTopology.cpp
    src/cross_section/CrossSection.cpp
    src/materials/Materials.cpp
    src/branches/BranchModel.cpp
    src/joints_and_bifurcations/Joints.cpp
    src/discretization/BeamElement.cpp
    src/discretization/Assembler.cpp
    src/solver_core/LinearAlgebra.cpp
    src/solver_core/DynamicSystem.cpp
    src/io/SimpleJson.cpp
    src/io/ModelIO.cpp
)

target_include_directories(
    orchard_solver
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(orchard_solver PUBLIC cxx_std_20)

target_compile_options(
    orchard_solver
    PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

add_executable(orchard_cli apps/orchard_cli.cpp)
target_link_libraries(orchard_cli PRIVATE orchard_solver)

if(ORCHARD_BUILD_TESTS)
    enable_testing()

    add_executable(orchard_tests tests/orchard_tests.cpp)
    target_link_libraries(orchard_tests PRIVATE orchard_solver)
    target_compile_definitions(
        orchard_tests
        PRIVATE
            ORCHARD_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_test(NAME orchard_unit_tests COMMAND orchard_tests)

    add_test(
        NAME orchard_demo_cli
        COMMAND
            orchard_cli
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/demo_orchard.json
            ${CMAKE_CURRENT_BINARY_DIR}/demo_frequency_response.csv
    )

    function(add_orchard_verification_test target_name source_file)
        add_executable(${target_name} ${source_file})
        target_link_libraries(${target_name} PRIVATE orchard_solver)
        add_test(NAME ${target_name} COMMAND ${target_name})
        set_tests_properties(${target_name} PROPERTIES LABELS "verification")
    endfunction()

    add_orchard_verification_test(verification_cantilever_first_mode tests/verification/case_cantilever_first_mode.cpp)
    add_orchard_verification_test(verification_cantilever_modes tests/verification/case_cantilever_modes.cpp)
    add_orchard_verification_test(verification_simple_beam_static tests/verification/case_simple_beam_static.cpp)
    add_orchard_verification_test(verification_duffing_frequency_response tests/verification/case_duffing_frequency_response.cpp)
    add_orchard_verification_test(verification_hinged_two_bar tests/verification/case_hinged_two_bar.cpp)
endif()
