cmake_minimum_required(VERSION 3.16)
project(xlite)

set(RUN_MODE "npu" CACHE STRING "sim/npu")
set(SOC_VERSION "Ascend910B3" CACHE STRING "system on chip type")
set(XLITE_KERNEL_SET "full" CACHE STRING "kernel set: full or llm_fp16")
set_property(CACHE XLITE_KERNEL_SET PROPERTY STRINGS full llm_fp16)
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory"
)

# Environment-facing projects such as vLLM-Ascend commonly spell the SoC name
# in lowercase, while AscendC's CMake/packaging layer uses the canonical
# product spelling (and embeds it in case-sensitive simulator/library paths).
# Accept either spelling at the boundary, but never pass the lowercase value
# into ascendc_library().
set(XLITE_SOC_VERSION_INPUT "${SOC_VERSION}")
string(TOLOWER "${SOC_VERSION}" XLITE_SOC_VERSION_LOWER)

if(NOT XLITE_KERNEL_SET STREQUAL "full" AND NOT XLITE_KERNEL_SET STREQUAL "llm_fp16")
    message(FATAL_ERROR "XLITE_KERNEL_SET must be one of: full, llm_fp16")
endif()

if(XLITE_SOC_VERSION_LOWER MATCHES "^ascend310p" AND
   NOT XLITE_KERNEL_SET STREQUAL "llm_fp16")
    message(FATAL_ERROR "Ascend310P POC only supports XLITE_KERNEL_SET=llm_fp16")
endif()

if(XLITE_KERNEL_SET STREQUAL "llm_fp16")
    if(NOT XLITE_SOC_VERSION_LOWER STREQUAL "ascend310p3")
        message(FATAL_ERROR
            "llm_fp16 requires SOC_VERSION=Ascend310P3 (case-insensitive input)")
    endif()
    set(SOC_VERSION "Ascend310P3" CACHE STRING "system on chip type" FORCE)
    set(XLITE_BUILD_ARCH_310P TRUE)
    message(STATUS
        "Normalized SOC_VERSION '${XLITE_SOC_VERSION_INPUT}' -> '${SOC_VERSION}' for AscendC")
    message(STATUS "Ascend310P architecture gate: device compilation must report __NPU_ARCH__=2002")
    message(STATUS "Ascend310P conservative memory profile: UB=192KiB; no 910B swizzle/CoreAssigner")
    foreach(_aclnn_header
            aclnn_matmul.h
            aclnn_prompt_flash_attention.h
            aclnn_incre_flash_attention.h)
        if(NOT EXISTS "${ASCEND_CANN_PACKAGE_PATH}/include/aclnnop/${_aclnn_header}")
            message(FATAL_ERROR
                "Ascend310P POC stop gate: missing ACLNN header "
                "${ASCEND_CANN_PACKAGE_PATH}/include/aclnnop/${_aclnn_header}")
        endif()
    endforeach()
    find_library(XLITE_NNOPBASE_LIBRARY NAMES nnopbase
        PATHS "${ASCEND_CANN_PACKAGE_PATH}/lib64" NO_DEFAULT_PATH)
    find_library(XLITE_OPAPI_LIBRARY NAMES opapi
        PATHS "${ASCEND_CANN_PACKAGE_PATH}/lib64" NO_DEFAULT_PATH)
    if(NOT XLITE_NNOPBASE_LIBRARY OR NOT XLITE_OPAPI_LIBRARY)
        message(FATAL_ERROR
            "Ascend310P POC stop gate: libnnopbase.so and libopapi.so are required in "
            "${ASCEND_CANN_PACKAGE_PATH}/lib64")
    endif()
endif()

message(STATUS "RUN_MODE: ${RUN_MODE}")
message(STATUS "SOC_VERSION: ${SOC_VERSION}")
message(STATUS "XLITE_KERNEL_SET: ${XLITE_KERNEL_SET}")
message(STATUS "ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")

find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug (default Release)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PATH ${CMAKE_CURRENT_LIST_DIR})
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/out" CACHE STRING "path for install()" FORCE)
else()
    set(CMAKE_INSTALL_PATH ${CMAKE_INSTALL_PREFIX})
endif()
message(STATUS "CMAKE_INSTALL_PATH: ${CMAKE_INSTALL_PATH}")

# xlite_kernels_common_${RUN_MODE}
file(GLOB KERNEL_FILES_COMMON
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/add_bias_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/group_matmul_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/matmul_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/silu_and_mul_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/softmax_topk_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/sigmoid_topk_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/topk_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/cast_bfloat16_t_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/permutation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/unpermutation_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/all_reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/reduce_scatter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/all_gather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/matmul_int8_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/matmul_int4b_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/unpack_activation_int8_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/group_matmul_int8_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/group_matmul_int4b_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/experts_counts_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/reorder_moe.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/concat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/split.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/conv1d_and_silu_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/beta_decay_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/recurrent_gated_delta_rule_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/msd_merge_dequant_int8_t.cpp
)

# xlite_kernels_bf16_${RUN_MODE}
file(GLOB KERNEL_FILES_BF16
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/add_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/add_bias_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/cast_float_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/embed_kernel_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/group_matmul_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/matmul_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/rope_and_cache_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/silu_and_mul_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/softmax_topk_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/sigmoid_topk_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/topk_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/unpermutation_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/attention_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/rope_complex_and_cache_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/flash_attention_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/quant_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/quant_dyn_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/norm_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/qk_rms_norm_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/mla_prepare_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/indexer_scores_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/muls_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/mla_v2_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/flash_mla_v2_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/gather_sparse_kv_cache_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/mla_v3_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/conv1d_and_silu_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/beta_decay_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/sigmoid_gate_mul_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/recurrent_gated_delta_rule_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/transpose_1_2_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/indexer_topk_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/einsum_mht_hdt_mhd_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/einsum_mht_htd_mhd_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/indexer_prepare_bfloat16_t.cpp
)

# xlite_kernels_f16_${RUN_MODE}
file(GLOB KERNEL_FILES_F16
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/add_bias_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/add_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/embed_kernel_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/group_matmul_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/matmul_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/rope_and_cache_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/silu_and_mul_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/attention_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/rope_complex_and_cache_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/flash_attention_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/dequant_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/norm_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/qk_rms_norm_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/mla_prepare_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/indexer_scores_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/muls_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/conv1d_and_silu_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/beta_decay_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/sigmoid_gate_mul_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/recurrent_gated_delta_rule_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/transpose_1_2_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/indexer_topk_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/einsum_mht_hdt_mhd_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/einsum_mht_htd_mhd_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/indexer_prepare_float16_t.cpp
)

if(XLITE_KERNEL_SET STREQUAL "llm_fp16")
    # Keep the original entries out of the compatibility target to avoid
    # duplicate launch symbols.  Dedicated 310P translation units validate the
    # real __NPU_ARCH__ and select the non-empty implementation before including
    # the shared kernel headers.
    set(KERNEL_FILES_F16_310P_COMPAT_ENTRIES
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/add_float16_t.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/embed_kernel_float16_t.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/norm_float16_t.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/qk_rms_norm_float16_t.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/rope_and_cache_float16_t.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/silu_and_mul_float16_t.cpp
    )
    set(KERNEL_FILES_F16_310P
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/310p/probe_310p.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/310p/add_float16_t.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/310p/embed_kernel_float16_t.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/310p/norm_float16_t.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/310p/qk_rms_norm_float16_t.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/310p/rope_and_cache_float16_t.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/310p/silu_and_mul_float16_t.cpp
    )
    # Keep the official Add baseline in a one-kernel library.  The passing
    # AddKernelInvocationNeo sample produces a default AIC device object;
    # mixing that kernel with AIV_ONLY entries can change the legacy AscendC
    # precompile/merge route on Ascend310P.
    set(KERNEL_FILES_310P_BASELINE
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/310p/official_add_probe_310p.cpp
    )
    list(REMOVE_ITEM KERNEL_FILES_F16
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/matmul_float16_t.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/attention_float16_t.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/flash_attention_float16_t.cpp
        ${KERNEL_FILES_F16_310P_COMPAT_ENTRIES}
    )
endif()

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist ,please check whether the cann package is installed")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# ascendc_library use to add kernel file to generate ascendc library
ascendc_library(xlite_kernels_common_${RUN_MODE} SHARED ${KERNEL_FILES_COMMON})
ascendc_library(xlite_kernels_bf16_${RUN_MODE} SHARED ${KERNEL_FILES_BF16})
ascendc_library(xlite_kernels_f16_${RUN_MODE} SHARED ${KERNEL_FILES_F16})

if(XLITE_BUILD_ARCH_310P)
    ascendc_library(xlite_kernels_f16_310p_${RUN_MODE} SHARED ${KERNEL_FILES_F16_310P})
    ascendc_compile_definitions(xlite_kernels_f16_310p_${RUN_MODE} PRIVATE XLITE_ARCH_310P=1)
    ascendc_library(xlite_kernels_baseline_310p_${RUN_MODE} SHARED ${KERNEL_FILES_310P_BASELINE})
    ascendc_compile_definitions(xlite_kernels_baseline_310p_${RUN_MODE}
        PRIVATE XLITE_ARCH_310P=1)
endif()

# Bundle all generated aclrtlaunch_*.h headers into one merged header so
# csrc/op.cpp only needs a single #include. The individual headers are only
# produced while the ascendc kernel libraries build (not at configure time),
# so the merge is regenerated by the build after the kernel libs are done.
set(_MERGED_KERN_H "${CMAKE_CURRENT_BINARY_DIR}/aclrtlaunch_all.h")
set(_XLITE_KERNEL_TARGETS
    xlite_kernels_common_${RUN_MODE}
    xlite_kernels_bf16_${RUN_MODE}
    xlite_kernels_f16_${RUN_MODE}
)
if(XLITE_BUILD_ARCH_310P)
    list(APPEND _XLITE_KERNEL_TARGETS
        xlite_kernels_f16_310p_${RUN_MODE}
        xlite_kernels_baseline_310p_${RUN_MODE}
    )
endif()
add_custom_command(
    OUTPUT "${_MERGED_KERN_H}"
    COMMAND bash -c "{ \
        echo '#pragma once'; \
        find '${CMAKE_CURRENT_BINARY_DIR}/include' -name 'aclrtlaunch_*.h' \
            -printf '#include \"%f\"\\n' | sort; } > \
            '${_MERGED_KERN_H}'"
    DEPENDS ${_XLITE_KERNEL_TARGETS}
    COMMENT "Generating ${_MERGED_KERN_H}"
    VERBATIM
)
add_custom_target(xlite_aclrtlaunch_all DEPENDS "${_MERGED_KERN_H}")

# If XLITE_DEBUG_ON env var is NOT set (empty / falsy), disable ascendc printf
# by adding ASCENDC_DUMP=0 to the ascendc kernel compile definitions. When
# XLITE_DEBUG_ON is set, ASCENDC_DUMP is left as the default (1) so printf
# remains enabled for debugging.
set(_xlite_dbg_enabled FALSE)
if(DEFINED ENV{XLITE_DEBUG_ON})
    string(TOLOWER "$ENV{XLITE_DEBUG_ON}" _xlite_dbg_raw)
    string(STRIP "${_xlite_dbg_raw}" _xlite_dbg_raw)
    if(NOT (_xlite_dbg_raw STREQUAL "" OR _xlite_dbg_raw MATCHES "^(0|false|no|off)$"))
        set(_xlite_dbg_enabled TRUE)
    endif()
    unset(_xlite_dbg_raw)
endif()

if(NOT _xlite_dbg_enabled)
    message(STATUS "XLITE_DEBUG_ON not set -> disabling ascendc printf via ASCENDC_DUMP=0 (default ASCENDC_DUMP=1)")
    ascendc_compile_definitions(xlite_kernels_common_${RUN_MODE} PRIVATE ASCENDC_DUMP=0)
    ascendc_compile_definitions(xlite_kernels_bf16_${RUN_MODE} PRIVATE ASCENDC_DUMP=0)
    ascendc_compile_definitions(xlite_kernels_f16_${RUN_MODE} PRIVATE ASCENDC_DUMP=0)
    if(XLITE_BUILD_ARCH_310P)
        ascendc_compile_definitions(xlite_kernels_f16_310p_${RUN_MODE} PRIVATE ASCENDC_DUMP=0)
        ascendc_compile_definitions(xlite_kernels_baseline_310p_${RUN_MODE}
            PRIVATE ASCENDC_DUMP=0)
    endif()
endif()

# xlite
add_library(xlite SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/debug.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/sock.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/swizzle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/runtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/core_assigner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/ccl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/auto_tuner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/trace/csv_sink.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/trace/matmul_tracer.cpp
)

if(XLITE_KERNEL_SET STREQUAL "llm_fp16")
    target_sources(xlite PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/csrc/aclnn_310p.cpp)
    target_compile_definitions(xlite PRIVATE
        XLITE_ARCH_310P=1
        XLITE_310P_LLM_FP16_POC=1
        XLITE_DISABLE_SWIZZLE_TABLE=1
    )
endif()

target_include_directories(xlite PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

add_dependencies(xlite xlite_aclrtlaunch_all)

# ---------------------------------------------------------------------------
# Debug control.
#
# Driven solely by the XLITE_DEBUG_ON env var (a comma-separated token list).
# CMake only PARSES it into macros; the debug module (csrc/debug.h) owns the
# logic of which macro gates which block.
#
#   XLITE_DEBUG_ON not set, empty, or a falsy token (0/false/no/off)
#       -> no debug macros defined (distribution build).
#   XLITE_DEBUG_ON=<tok>[,<tok>...]  (any non-falsy value)
#       -> defines XLITE_DEBUG_ON (base debug) PLUS, for each token <tok>,
#          a macro XLITE_DEBUG_ON_<TOK> (token uppercased). e.g.
#          XLITE_DEBUG_ON=forward,tuner  ->  XLITE_DEBUG_ON
#                                            XLITE_DEBUG_ON_FORWARD
#                                            XLITE_DEBUG_ON_TUNER
#
# Token recognition (uppercasing) happens here; unknown tokens are still passed
# through as XLITE_DEBUG_ON_<TOK> so the debug module can warn/ignore them.
# ---------------------------------------------------------------------------
if(DEFINED ENV{XLITE_DEBUG_ON})
    string(TOLOWER "$ENV{XLITE_DEBUG_ON}" _xlite_dbg_raw)
    string(STRIP "${_xlite_dbg_raw}" _xlite_dbg_raw)
    if(NOT (_xlite_dbg_raw STREQUAL "" OR _xlite_dbg_raw MATCHES "^(0|false|no|off)$"))
        target_compile_definitions(xlite PUBLIC XLITE_DEBUG_ON)
        string(REPLACE "," ";" _xlite_dbg_list "${_xlite_dbg_raw}")
        foreach(_tok IN LISTS _xlite_dbg_list)
            string(STRIP "${_tok}" _tok)
            if(_tok STREQUAL "")
                continue()
            endif()
            string(TOUPPER "${_tok}" _tok)
            target_compile_definitions(xlite PUBLIC "XLITE_DEBUG_ON_${_tok}")
        endforeach()
        message(STATUS "XLITE_DEBUG_ON='${_xlite_dbg_raw}' -> XLITE_DEBUG_ON + per-token XLITE_DEBUG_ON_<TOK>")
    else()
        message(STATUS "XLITE_DEBUG_ON='${_xlite_dbg_raw}' -> no debug macros")
    endif()
    unset(_xlite_dbg_raw)
    unset(_xlite_dbg_list)
    unset(_tok)
endif()

# ---------------------------------------------------------------------------
# Matmul shape trace. Driven solely by the XLITE_TRACE_MATMUL env var. When set
# (truthy), defines XLITE_TRACE_MATMUL so xlite tracer records a histogram of
# matmul shapes and flushes it crash-safely to a file.
# ---------------------------------------------------------------------------
if(DEFINED ENV{XLITE_TRACE_MATMUL})
    string(TOLOWER "$ENV{XLITE_TRACE_MATMUL}" _xlite_trace_raw)
    string(STRIP "${_xlite_trace_raw}" _xlite_trace_raw)
    if(NOT (_xlite_trace_raw STREQUAL "" OR _xlite_trace_raw MATCHES "^(0|false|no|off)$"))
        target_compile_definitions(xlite PUBLIC XLITE_TRACE_MATMUL)
        message(STATUS "XLITE_TRACE_MATMUL='${_xlite_trace_raw}' -> XLITE_TRACE_MATMUL")
    endif()
    unset(_xlite_trace_raw)
endif()

target_compile_options(xlite PRIVATE
    -std=c++17 -O3 -Wall -Werror
)
if(XLITE_KERNEL_SET STREQUAL "llm_fp16")
    # CANN 9.1 emits unconditional #warning directives from the still-supported
    # v1 Prompt/Incre FlashAttention headers.  Keep -Werror for project code,
    # but do not turn third-party preprocessor notices into build failures.
    target_compile_options(xlite PRIVATE -Wno-error=cpp)
endif()

target_link_libraries(xlite PRIVATE
    xlite_kernels_common_${RUN_MODE}
    xlite_kernels_bf16_${RUN_MODE}
    xlite_kernels_f16_${RUN_MODE}
    hccl
    torch
)
if(XLITE_KERNEL_SET STREQUAL "llm_fp16")
    target_link_libraries(xlite PRIVATE
        xlite_kernels_f16_310p_${RUN_MODE}
        xlite_kernels_baseline_310p_${RUN_MODE}
        ${XLITE_NNOPBASE_LIBRARY}
        ${XLITE_OPAPI_LIBRARY}
    )
endif()

# _C
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
    OUTPUT_VARIABLE PYTHON_SYS_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(PYTHON_SYS_DIR)
  # prepend so Python sys dir is searched first; use APPEND if you want it last
  list(PREPEND CMAKE_PREFIX_PATH "${PYTHON_SYS_DIR}")
endif()

find_package(pybind11 REQUIRED)
find_package(Torch COMPONENTS Development REQUIRED)

message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")
message(STATUS "pybind11 include: ${pybind11_INCLUDE_DIRS}")
message(STATUS "torch cpp_extension include: ${TORCH_INCLUDE_DIRS}")
message(STATUS "torch cpp_extension lib: ${TORCH_INSTALL_PREFIX}/lib")

pybind11_add_module(
    _C
    csrc/_C.cpp
)

target_compile_options(_C PRIVATE
    -std=c++17 -O3 -Wall -Werror -fPIC
)

target_include_directories(_C PRIVATE
    ${pybind11_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

target_link_directories(_C PRIVATE
    ${TORCH_INSTALL_PREFIX}/lib
)

target_link_libraries(_C PRIVATE
    xlite
    c10
    torch
    torch_python
    torch_cpu
)

set(XLITE_EDITABLE_BUILD "OFF" CACHE STRING "Enable editable build and include torch from source path")
if (XLITE_EDITABLE_BUILD STREQUAL "ON")
    set(TORCH_RPATH "\$ORIGIN/../torch/lib:${TORCH_INSTALL_PREFIX}/lib")
else()
    set(TORCH_RPATH "\$ORIGIN/../torch/lib")
endif()
message(STATUS "TORCH_RPATH: ${TORCH_RPATH}")

set_target_properties(xlite PROPERTIES
    LINK_FLAGS "-Wl,-rpath,\$ORIGIN:\$ORIGIN/lib:${TORCH_RPATH}"
)
set_target_properties(_C PROPERTIES
    LINK_FLAGS "-Wl,-rpath,\$ORIGIN:\$ORIGIN/lib:${TORCH_RPATH}"
)

install(TARGETS xlite_kernels_common_${RUN_MODE} xlite_kernels_bf16_${RUN_MODE} xlite_kernels_f16_${RUN_MODE} xlite
    LIBRARY DESTINATION ${CMAKE_INSTALL_PATH}/xlite/lib
)
if(XLITE_BUILD_ARCH_310P)
    install(TARGETS
        xlite_kernels_f16_310p_${RUN_MODE}
        xlite_kernels_baseline_310p_${RUN_MODE}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PATH}/xlite/lib
    )
endif()

install(TARGETS _C
    LIBRARY DESTINATION ${CMAKE_INSTALL_PATH}/xlite
)

# Install public C++ headers.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/csrc/xlite.h
              ${CMAKE_CURRENT_SOURCE_DIR}/csrc/base.h
              ${CMAKE_CURRENT_SOURCE_DIR}/csrc/runtime.h
              ${CMAKE_CURRENT_SOURCE_DIR}/csrc/model.h
        DESTINATION ${CMAKE_INSTALL_PATH}/xlite/include/xlite)

# Install cmake package config for find_package(xlite).
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/csrc/xliteConfig.cmake
        DESTINATION ${CMAKE_INSTALL_PATH}/xlite/lib/cmake/xlite)
