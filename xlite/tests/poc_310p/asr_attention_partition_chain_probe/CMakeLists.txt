cmake_minimum_required(VERSION 3.16)
project(xlite_asr_attention_partition_chain_probe LANGUAGES C CXX)
set(RUN_MODE "npu" CACHE STRING "run mode")
set(SOC_VERSION "Ascend310P3" CACHE STRING "target SoC")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE STRING "CANN path")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR "The partition-chain probe must be built in Release mode")
endif()
if(NOT SOC_VERSION STREQUAL "Ascend310P3" OR NOT RUN_MODE STREQUAL "npu")
    message(FATAL_ERROR "The partition-chain probe requires Ascend310P3 NPU mode")
endif()
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake was not found")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(xlite_asr_attention_partition_state_kernel SHARED
    ../../../csrc/kernels/310p/asr_paged_decode_attention_fp16.cpp)
ascendc_compile_definitions(xlite_asr_attention_partition_state_kernel
    PRIVATE CUSTOM_ASCEND310P XLITE_ASR_ATTENTION_PARTITION_PROBE=1)

ascendc_library(xlite_asr_attention_partition_merge_chain_kernel SHARED
    ../asr_attention_partition_merge_probe/asr_attention_partition_merge_probe.cpp)
ascendc_compile_definitions(xlite_asr_attention_partition_merge_chain_kernel
    PRIVATE CUSTOM_ASCEND310P)

add_executable(xlite_asr_attention_partition_chain_runner main.cpp)
target_compile_features(xlite_asr_attention_partition_chain_runner PRIVATE cxx_std_17)
target_compile_options(xlite_asr_attention_partition_chain_runner PRIVATE
    -O2 -Wall -Werror -D_GLIBCXX_USE_CXX11_ABI=0)
target_link_libraries(xlite_asr_attention_partition_chain_runner PRIVATE
    host_intf_pub
    xlite_asr_attention_partition_state_kernel
    xlite_asr_attention_partition_merge_chain_kernel
    ascendalog dl)
