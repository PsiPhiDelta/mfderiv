cmake_minimum_required(VERSION 3.16)
project(mfderiv VERSION 0.1.0
        DESCRIPTION "Pressure derivatives of stationary effective potentials with automatic differentiation"
        LANGUAGES CXX)

include(GNUInstallDirs)

find_package(Eigen3 REQUIRED NO_MODULE)

# autodiff (https://autodiff.github.io) is header-only and only its scalar types are used. Take an
# installed copy if there is one, otherwise download the headers.
find_path(AUTODIFF_INCLUDE_DIR autodiff/forward/dual.hpp)
option(MFDERIV_FETCH_AUTODIFF "Download the autodiff headers if they are not found" ON)
if(NOT AUTODIFF_INCLUDE_DIR AND MFDERIV_FETCH_AUTODIFF)
    include(FetchContent)
    FetchContent_Declare(autodiff
        GIT_REPOSITORY https://github.com/autodiff/autodiff.git
        GIT_TAG v1.1.2
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR headers-only)  # no CMakeLists.txt there, so autodiff itself is not configured
    FetchContent_MakeAvailable(autodiff)
    set(AUTODIFF_INCLUDE_DIR ${autodiff_SOURCE_DIR} CACHE PATH "autodiff headers" FORCE)
endif()
if(NOT AUTODIFF_INCLUDE_DIR)
    message(FATAL_ERROR "autodiff headers not found. Set AUTODIFF_INCLUDE_DIR or MFDERIV_FETCH_AUTODIFF=ON.")
endif()

add_library(mfderiv INTERFACE)
add_library(mfderiv::mfderiv ALIAS mfderiv)
target_include_directories(mfderiv INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(mfderiv SYSTEM INTERFACE $<BUILD_INTERFACE:${AUTODIFF_INCLUDE_DIR}>)
target_link_libraries(mfderiv INTERFACE Eigen3::Eigen)
target_compile_features(mfderiv INTERFACE cxx_std_17)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(MFDERIV_TOP_LEVEL ON)
else()
    set(MFDERIV_TOP_LEVEL OFF)
endif()
option(MFDERIV_BUILD_TESTS "Build the tests" ${MFDERIV_TOP_LEVEL})
option(MFDERIV_BUILD_EXAMPLES "Build the examples" ${MFDERIV_TOP_LEVEL})

if(MFDERIV_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if(MFDERIV_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

install(DIRECTORY include/mfderiv DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
