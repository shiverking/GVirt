cmake_minimum_required(VERSION 3.16)
project(xlite_m200_cube_probe LANGUAGES C CXX)

set(RUN_MODE "npu" CACHE STRING "probe run mode")
set(SOC_VERSION "Ascend310P3" CACHE STRING "target SoC")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "Ascend CANN package path")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT SOC_VERSION STREQUAL "Ascend310P3")
    message(FATAL_ERROR "The Xlite M200 Cube probe only supports SOC_VERSION=Ascend310P3")
endif()
if(NOT RUN_MODE STREQUAL "npu")
    message(FATAL_ERROR "The Xlite M200 Cube probe must run with RUN_MODE=npu")
endif()

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake was not found below ASCEND_CANN_PACKAGE_PATH")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(xlite_m200_cube_probe_kernel SHARED m200_cube_probe.cpp)
ascendc_compile_definitions(xlite_m200_cube_probe_kernel PRIVATE
    CUSTOM_ASCEND310P
    HAVE_WORKSPACE
    HAVE_TILING)

add_executable(xlite_m200_cube_probe_runner
    main.cpp
    m200_cube_probe_tiling.cpp)
target_compile_features(xlite_m200_cube_probe_runner PRIVATE cxx_std_17)
target_compile_options(xlite_m200_cube_probe_runner PRIVATE
    -O2 -Wall -Werror -D_GLIBCXX_USE_CXX11_ABI=0)
target_compile_definitions(xlite_m200_cube_probe_runner PRIVATE
    CUSTOM_ASCEND310P
    SOC_VERSION="${SOC_VERSION}")
target_link_libraries(xlite_m200_cube_probe_runner PRIVATE
    host_intf_pub
    xlite_m200_cube_probe_kernel
    tiling_api
    register
    platform
    ascendalog
    dl)
