cmake_minimum_required(VERSION 4.0)
set(CMAKE_C_COMPILER /opt/homebrew/bin/gcc-15)
project(quattordici C)
set(CMAKE_C_STANDARD 17)

add_executable(quattordici quattordici.c)

find_package(OpenMP)
if (OpenMP_C_FOUND)
    message(STATUS "Found OpenMP")
    #target_link_libraries(quattordici PUBLIC OpenMP::OpenMP_C)
else ()
    message(WARNING "OpenMP not found, trying to link manually")
    # Include libomp headers from Homebrew
    target_include_directories(quattordici PRIVATE /opt/homebrew/opt/libomp/include)
    # Link library
    target_link_libraries(quattordici PRIVATE /opt/homebrew/opt/libomp/lib/libomp.dylib)
    # Compile options for Apple Clang
    target_compile_options(quattordici PRIVATE -Xpreprocessor -fopenmp -O3 -Wall -lm)
    message(STATUS "OpenMP manually linked")
endif ()