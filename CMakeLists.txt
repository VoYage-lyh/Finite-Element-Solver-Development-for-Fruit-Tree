cmake_minimum_required(VERSION 3.21)

project(
    OrchardVibrationSolver
    VERSION 0.1.0
    DESCRIPTION "Dedicated fast solver for orchard tree vibration analysis"
    LANGUAGES NONE
)

option(ORCHARD_ENABLE_LEGACY_CPP "Build archived C++ solver targets" OFF)

message(STATUS "The main branch no longer ships active C++ solver sources.")
message(STATUS "Use the Orchard FEM package for active workflows:")
message(STATUS "  python -m orchard_fem --help")

if(ORCHARD_ENABLE_LEGACY_CPP)
    message(FATAL_ERROR
        "Archived C++ headers and sources have been removed from the main branch. "
        "Use an older git revision if you need the historical C++ build."
    )
endif()
