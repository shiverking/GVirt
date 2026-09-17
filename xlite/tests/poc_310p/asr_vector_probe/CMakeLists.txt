cmake_minimum_required(VERSION 3.16)
project(xlite_asr_vector_probe LANGUAGES C CXX)

set(RUN_MODE "npu" CACHE STRING "probe run mode")
set(SOC_VERSION "Ascend310P3" CACHE STRING "target SoC")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "Ascend CANN package path")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(NOT SOC_VERSION STREQUAL "Ascend310P3")
    message(FATAL_ERROR "The ASR vector probe only supports Ascend310P3")
endif()
if(NOT RUN_MODE STREQUAL "npu")
    message(FATAL_ERROR "The ASR vector probe must run on NPU")
endif()

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake was not found")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# One vector-only library avoids mixing these AIV objects with AIC projection
# objects while the kernels are still isolated micro-probes.
ascendc_library(xlite_asr_vector_probe_kernel SHARED
    ../../../csrc/kernels/310p/asr_add_rmsnorm_fp16.cpp
    ../../../csrc/kernels/310p/asr_qk_norm_mrope_cache_fp16.cpp
    ../../../csrc/kernels/310p/asr_rmsnorm_fp16.cpp
    ../../../csrc/kernels/310p/asr_silu_mul_fp16.cpp)
ascendc_compile_definitions(xlite_asr_vector_probe_kernel PRIVATE CUSTOM_ASCEND310P)

add_executable(xlite_asr_vector_probe_runner main.cpp)
target_compile_features(xlite_asr_vector_probe_runner PRIVATE cxx_std_17)
target_compile_options(xlite_asr_vector_probe_runner PRIVATE
    -O2 -Wall -Werror -D_GLIBCXX_USE_CXX11_ABI=0)
target_compile_definitions(xlite_asr_vector_probe_runner PRIVATE
    CUSTOM_ASCEND310P SOC_VERSION="${SOC_VERSION}")
target_link_libraries(xlite_asr_vector_probe_runner PRIVATE
    host_intf_pub xlite_asr_vector_probe_kernel ascendalog dl)
