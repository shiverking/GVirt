/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"
#include "kernel_param.h"

#if defined(__DAV_C220_VEC__) || defined(XLITE_ARCH_310P)

#if defined(XLITE_ARCH_310P)
__aicore__ inline void xlite_310p_set_mask(uint32_t len)
{
    uint64_t tail = len % 64;
    uint64_t mask = (static_cast<uint64_t>(1) << tail) - 1;
    if (len == 128) {
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    } else if (len >= 64) {
        set_vector_mask(mask, static_cast<uint64_t>(-1));
    } else {
        set_vector_mask(0, mask);
    }
}

// CANN 9.1's M200 AscendC does not expose the C220 vcadd reduction intrinsic.
// Qwen3's norm dimensions are powers of two, so reduce in place with the vadd
// primitive that is available on M200.  Larger stages use repeated 64-lane
// FP32 vectors; the final stages progressively narrow the vector mask.
__aicore__ inline void xlite_310p_reduce_sum(__ubuf__ float *dst, __ubuf__ float *src,
                                             uint32_t dim)
{
    assert(dim != 0 && (dim & (dim - 1)) == 0);
    constexpr uint32_t lanes = VECTOR_MAX_BYTESIZE / sizeof(float);
    uint32_t remain = dim;
    set_mask_norm();
    while (remain > 1) {
        uint32_t half = remain / 2;
        uint32_t repeat = 1;
        if (half >= lanes) {
            set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            repeat = half / lanes;
        } else {
            xlite_310p_set_mask(half);
        }
        vadd(dst, src, src + half, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        src = dst;
        remain = half;
    }
    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}
#endif

__aicore__ inline void reduce_sum(__ubuf__ float *buf, uint32_t cnt_per_token, uint32_t norm_dim)
{
    if (norm_dim == 128) {
        for (uint32_t norm_idx = 0; norm_idx < cnt_per_token; norm_idx++) {
            auto buf_norm = buf + norm_idx * norm_dim;
#if defined(XLITE_ARCH_310P)
            xlite_310p_reduce_sum(buf_norm, buf_norm, norm_dim);
#else
            vadd(buf_norm, buf_norm, buf_norm + 64, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vcadd(buf_norm, buf_norm, 1, 1, 1, 8, 0);
#endif
        }
    } else {
        for (uint32_t norm_idx = 0; norm_idx < cnt_per_token; norm_idx++) {
            auto buf_norm = buf + norm_idx * norm_dim;
#if defined(XLITE_ARCH_310P)
            xlite_310p_reduce_sum(buf_norm, buf_norm, norm_dim);
#else
            ReduceSum(buf_norm, buf_norm, norm_dim);
#endif
        }
    }
}

__aicore__ inline void duplicate_item(__ubuf__ float *buf, uint32_t cnt_per_token, uint64_t repeat,
                                      uint32_t norm_dim)
{
    for (uint32_t norm_idx = 0; norm_idx < cnt_per_token; norm_idx++) {
        auto calc_norm = buf + norm_idx * norm_dim;
        float dupnum = *calc_norm;
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        vector_dup(calc_norm, dupnum, repeat, 1, 1, 8, 1);
    }
}

// Tiled no-affine RMSNorm for norm_dim > NORM_TILED_THRESHOLD.
// Two-pass scan in tileDim chunks (re-reads GM in pass 2; row ~32KB so cheap):
//   pass 1: sum(x^2) -> scalar;  pass 2: y = x * rsqrt(sum/norm_dim + eps)
template <typename Dtype>
__aicore__ inline void rmsnorm_noaffine_tiled(__gm__ Dtype *input, __gm__ Dtype *output,
                                              uint32_t token_num, uint32_t norm_dim, float norm_eps,
                                              uint32_t in_step, uint32_t out_step,
                                              bool outFp32 = false)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    constexpr int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);
    // Two-pass scan over tileDim chunks (re-reads GM in pass 2; one row ~32KB).
    // IO ping-pong (in/out dual slot) + single compute buffer: x and x^2/√ never
    // coexist across the two passes, so one calc suffices (unlike single-pass
    // norm<>(), which needs calc0=x and calc1=√ alive together for vdiv).
    // ~96KB UB. flag protocol (IN->CALC->OUT) modeled on hc_post.h; pipe_barrier
    // does not wait GM<->UB DMA, hence the MTE2/MTE3 flags.
    const uint32_t tileDim = 8192;
    const uint32_t nTile = DIV_ROUND_UP(norm_dim, tileDim);

    const uint64_t dtypeLen = ROUND_UP(tileDim * sizeof(Dtype), UB_BUF_ALIGN_SIZE);
    const uint64_t fp32Len = ROUND_UP(tileDim * sizeof(float), UB_BUF_ALIGN_SIZE);

    uint64_t off = 0;
    __ubuf__ Dtype *in0 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += dtypeLen;
    __ubuf__ Dtype *in1 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += dtypeLen;
    __ubuf__ Dtype *inArr[2] = {in0, in1};
    __ubuf__ float *calc = reinterpret_cast<__ubuf__ float *>(off);
    off += fp32Len;
    // out buffer: fp32-sized when outFp32 (result staged as fp32 before GM
    // store), else Dtype-sized (cast path). Always ping-pong (ID2/ID3) so the
    // MTE3 GM store is decoupled from V's next-tile load+convert on calc.
    const uint64_t outLen = outFp32 ? fp32Len : dtypeLen;
    __ubuf__ Dtype *out0 = nullptr;
    __ubuf__ Dtype *out1 = nullptr;
    __ubuf__ Dtype *outArr[2] = {nullptr, nullptr};
    __ubuf__ float *outFp0 = nullptr;
    __ubuf__ float *outFp1 = nullptr;
    __ubuf__ float *outFpArr[2] = {nullptr, nullptr};
    if (outFp32) {
        outFp0 = reinterpret_cast<__ubuf__ float *>(off);
        off += outLen;
        outFp1 = reinterpret_cast<__ubuf__ float *>(off);
        off += outLen;
        outFpArr[0] = outFp0;
        outFpArr[1] = outFp1;
    } else {
        out0 = reinterpret_cast<__ubuf__ Dtype *>(off);
        off += outLen;
        out1 = reinterpret_cast<__ubuf__ Dtype *>(off);
        off += outLen;
        outArr[0] = out0;
        outArr[1] = out1;
    }
    __ubuf__ float *scalar =
        reinterpret_cast<__ubuf__ float *>(off);  // vector_dup/vsqrt landing (32B)
    off += ROUND_UP(sizeof(float), UB_BUF_ALIGN_SIZE);
    assert(off <= UB_SIZE);

    const float inv_n = (float)1.0 / norm_dim;

    // Prime: ID0/ID1 -> in slots 0/1; ID2/ID3 -> out slots 0/1.
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);

    for (uint32_t tok = block_idx; tok < token_num; tok += (uint32_t)block_num) {
        __gm__ Dtype *gm_in = input + tok * in_step;
        // out_step is an element count passed by the host (op.cpp sends
        // out.shape[1]). Pointer arithmetic on __gm__ Dtype* scales it by
        // sizeof(Dtype) -- correct for the cast path (output is Dtype), but
        // outFp32 writes fp32 (4B) to an fp32 GM buffer, so the per-token byte
        // stride must be out_step * sizeof(float). Reinterpret to float* here
        // so tok*out_step lands on the right fp32 row.
        __gm__ Dtype *gm_out = !outFp32
                                   ? (output + tok * out_step)
                                   : reinterpret_cast<__gm__ Dtype *>(
                                         reinterpret_cast<__gm__ float *>(output) + tok * out_step);

        // Pass 1: accumulate sum(x^2) over tiles.
        float sum_sq = 0.f;
        int curr = 0;
        for (uint32_t t = 0; t < nTile; t++) {
            const uint32_t tileOffset = t * tileDim;
            const uint32_t dLeft = norm_dim - tileOffset;
            const uint32_t dCur = dLeft < tileDim ? dLeft : tileDim;
            const int repeat = DIV_ROUND_UP(dCur, calcPad);
            const uint32_t dBytes = dCur * sizeof(Dtype);

            // IN: MTE2 load -> V cast to fp32.
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
            CopyGmToUbufAligned(inArr[curr], gm_in + tileOffset, dBytes);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
            convert_input(calc, inArr[curr], repeat);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);

            // CALC: x^2 -> in-place reduce to calc[0] -> S-pipe gather.
            vmul(calc, calc, calc, repeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            ReduceSum(calc, calc, dCur);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            sum_sq += *calc;
            curr = 1 - curr;
        }

        // rsqrt(sum_sq/n + eps). sqrtf is unavailable in AIV; broadcast via
        // vector_dup, vsqrt one element, read back through S pipe. vrec is
        // low-precision, so invert as scalar 1.0f/x. SetMask(1) keeps the single
        // fp32 write within the 32B scalar slot.
        SetMask(1);
        vector_dup(scalar, sum_sq * inv_n + norm_eps, 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        vsqrt(scalar, scalar, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float sqrt_val = *scalar;
        float rsqrt = 1.0f / sqrt_val;
        pipe_barrier(PIPE_ALL);

        // Pass 2: y = x * rsqrt. Pass-1 tail left both in slots reusable.
        curr = 0;
        for (uint32_t t = 0; t < nTile; t++) {
            const uint32_t tileOffset = t * tileDim;
            const uint32_t dLeft = norm_dim - tileOffset;
            const uint32_t dCur = dLeft < tileDim ? dLeft : tileDim;
            const int repeat = DIV_ROUND_UP(dCur, calcPad);
            const uint32_t dBytes = dCur * sizeof(Dtype);

            // IN: re-load x.
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
            CopyGmToUbufAligned(inArr[curr], gm_in + tileOffset, dBytes);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
            convert_input(calc, inArr[curr], repeat);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);

            // CALC + OUT: x*rsqrt -> cast -> store.
            vmuls(calc, calc, rsqrt, repeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);

            if (outFp32) {
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
                copy_ubuf_to_ubuf(outFpArr[curr], calc, 0, 1,
                                  DIV_ROUND_UP(dCur * sizeof(float), BLOCK_SIZE), 0, 0);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
                CopyUbufToGmAligned(reinterpret_cast<__gm__ float *>(gm_out) + tileOffset,
                                    outFpArr[curr], dCur * sizeof(float));
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
            } else {
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
                convert_output(outArr[curr], calc, repeat);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
                CopyUbufToGmAligned(gm_out + tileOffset, outArr[curr], dBytes);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
            }
            curr = 1 - curr;
        }
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
}

template <typename Dtype>
__aicore__ inline void norm(GM_ADDR input, GM_ADDR addInOut, GM_ADDR weight, GM_ADDR bias,
                            GM_ADDR output, uint32_t token_num, uint32_t norm_dim, float norm_eps,
                            int kind, uint32_t cnt_per_token, uint32_t in_step, uint32_t out_step,
                            uint32_t in_start_offset, uint32_t out_start_offset, bool useNorm,
                            GM_ADDR variance, uint32_t tpSize, GM_ADDR kcache = nullptr,
                            GM_ADDR slot_mapping = nullptr, uint32_t block_size = 0,
                            int coreOffset = 0, int *nextCoreOffset = nullptr, bool outFp32 = false)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    auto normKind = static_cast<NormKind>(kind);

    if (useNorm && !weight && norm_dim > 6144 && cnt_per_token == 1 && normKind == NormKind::Rms) {
        rmsnorm_noaffine_tiled<Dtype>(reinterpret_cast<__gm__ Dtype *>(input),
                                      reinterpret_cast<__gm__ Dtype *>(output), token_num, norm_dim,
                                      norm_eps, in_step, out_step, outFp32);
        if (nextCoreOffset) {
            *nextCoreOffset = (coreOffset + token_num) % block_num;
        }
        return;
    }

    float inv_n = (float)1.0 / norm_dim;
    float divTpSize = (float)1.0 / tpSize;
    uint32_t total_dim = norm_dim * cnt_per_token;
    uint64_t len_burst = DIV_ROUND_UP(total_dim * sizeof(Dtype), BLOCK_SIZE);
    uint64_t len_burst_per_norm = DIV_ROUND_UP(norm_dim * sizeof(Dtype), BLOCK_SIZE);
    uint32_t inout_blocksize = ROUND_UP(total_dim * sizeof(Dtype), BLOCK_SIZE);
    uint32_t inout_blocksize_float = ROUND_UP(total_dim * sizeof(float), BLOCK_SIZE);
    uint32_t calc_blocksize = ROUND_UP(total_dim * sizeof(float), BLOCK_SIZE);
    int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);
    uint64_t repeat = DIV_ROUND_UP(total_dim, calcPad);
    uint64_t repeat_per_norm = DIV_ROUND_UP(norm_dim, calcPad);
    uint64_t len_burst_float_per_norm = DIV_ROUND_UP(norm_dim * sizeof(float), BLOCK_SIZE);
    uint64_t len_burst_float = DIV_ROUND_UP(total_dim * sizeof(float), BLOCK_SIZE);
    uint64_t repeat_stride = DIV_ROUND_UP(norm_dim * sizeof(float), BLOCK_SIZE);

    uint64_t off = 0;
    __ubuf__ Dtype *in0 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += inout_blocksize;
    __ubuf__ Dtype *in1 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += inout_blocksize;
    __ubuf__ Dtype *in[2] = {in0, in1};

    __ubuf__ Dtype *out[2];
    __ubuf__ float *out_float[2];
    __ubuf__ float *in_variance_float[2];

    // When useNorm is set to false, only variance is calculated, so no need for bias or weight
    assert(useNorm || (!weight && !bias));
    // addInOut is only passed by AddAndRmsNorm, which launches with outFp32=false (EachXDtype
    // forces in==out dtype there). The addInOut residue path below uses the Dtype out buffer,
    // so outFp32 (which leaves out unallocated) must never combine with a non-null addInOut.
    assert(!(addInOut && outFp32));

    if (useNorm && !outFp32) {
        out[0] = reinterpret_cast<__ubuf__ Dtype *>(off);
        off += inout_blocksize;
        out[1] = reinterpret_cast<__ubuf__ Dtype *>(off);
        off += inout_blocksize;
    } else {
        out_float[0] = reinterpret_cast<__ubuf__ float *>(off);
        off += inout_blocksize_float;
        out_float[1] = reinterpret_cast<__ubuf__ float *>(off);
        off += inout_blocksize_float;
    }
    if (variance) {
        in_variance_float[0] = reinterpret_cast<__ubuf__ float *>(off);
        off += inout_blocksize_float;
        in_variance_float[1] = reinterpret_cast<__ubuf__ float *>(off);
        off += inout_blocksize_float;
    }

    __ubuf__ float *calc0 = reinterpret_cast<__ubuf__ float *>(off);
    off += calc_blocksize;
    __ubuf__ float *calc1 = reinterpret_cast<__ubuf__ float *>(off);
    off += calc_blocksize;
    __ubuf__ float *weight_calc = reinterpret_cast<__ubuf__ float *>(off);
    off += calc_blocksize;
    __ubuf__ float *bias_calc = reinterpret_cast<__ubuf__ float *>(off);
    off += calc_blocksize;
    assert(off <= UB_SIZE);

    int inCurr = 0;
    int outCurr = 0;

    if (token_num >= block_num || (block_idx >= coreOffset && block_idx < coreOffset + token_num) ||
        (block_idx < coreOffset && block_idx < ((coreOffset + token_num) % block_num))) {
        if (weight) {
            copy_gm_to_ubuf(in[inCurr], (__gm__ Dtype *)weight, 0, 1, len_burst_per_norm, 0, 0);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + inCurr);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + inCurr);
            convert_input(weight_calc, in[inCurr], repeat_per_norm);
            inCurr = 1 - inCurr;
        }
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        if (bias) {
            copy_gm_to_ubuf(in[inCurr], (__gm__ Dtype *)bias, 0, 1, len_burst_per_norm, 0, 0);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + inCurr);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + inCurr);
            convert_input(bias_calc, in[inCurr], repeat_per_norm);
            inCurr = 1 - inCurr;
        }
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

        pipe_barrier(PIPE_V);

        if (weight || bias) {
            for (uint32_t norm_idx = 1; norm_idx < cnt_per_token; norm_idx++) {
                if (weight) {
                    auto weight_norm = weight_calc + norm_idx * norm_dim;
                    copy_ubuf_to_ubuf(weight_norm, weight_calc, 0, 1, len_burst_float_per_norm, 0,
                                      0);
                }
                if (bias) {
                    auto bias_norm = bias_calc + norm_idx * norm_dim;
                    copy_ubuf_to_ubuf(bias_norm, bias_calc, 0, 1, len_burst_float_per_norm, 0, 0);
                }
                if (norm_idx == cnt_per_token - 1) {
                    pipe_barrier(PIPE_V);
                }
            }
        }
    } else {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    }

    bool need_cache = (block_size != 0) && (kcache != nullptr) && (slot_mapping != nullptr);
    if (need_cache) {
        assert(useNorm);
    }

    if (nextCoreOffset) {
        *nextCoreOffset = (coreOffset + token_num) % block_num;
    }
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    int first = (block_idx + block_num - coreOffset) % block_num;
    for (uint32_t loop = first; loop < token_num; loop += block_num) {
        uint32_t block = 0, block_offset = 0;
        if (need_cache) {
            uint32_t slot_idx = (uint32_t)(*((__gm__ uint32_t *)slot_mapping + loop));
            block = slot_idx / block_size;
            block_offset = slot_idx % block_size;
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        }

        uint32_t in_offset = in_start_offset + loop * in_step;
        uint32_t out_offset = out_start_offset + loop * out_step;
        auto gm_in = (__gm__ Dtype *)input + in_offset;
        auto gm_out_dtype = (__gm__ Dtype *)output + out_offset;
        auto gm_out_float = (__gm__ float *)output + out_offset;

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + inCurr);
        copy_gm_to_ubuf(in[inCurr], gm_in, 0, 1, len_burst, 0, 0);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + inCurr);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + inCurr);

        convert_input(calc0, in[inCurr], repeat);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + inCurr);
        inCurr = 1 - inCurr;

        if (addInOut) {
            auto gm_addInOut = (__gm__ Dtype *)addInOut + in_offset;
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + inCurr);
            copy_gm_to_ubuf(in[inCurr], gm_addInOut, 0, 1, len_burst, 0, 0);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + inCurr);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + inCurr);

            convert_input(calc1, in[inCurr], repeat);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + inCurr);
            inCurr = 1 - inCurr;
            vadd(calc0, calc1, calc0, repeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + outCurr);
            convert_output(out[outCurr], calc0, repeat);

            pipe_barrier(PIPE_V);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + outCurr);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + outCurr);

            copy_ubuf_to_gm(gm_addInOut, out[outCurr], 0, 1, len_burst, 0, 0);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + outCurr);
            outCurr = 1 - outCurr;
        }

        // Compute (x - mean) if LayerNorm
        if (normKind == NormKind::Layer) {
            // x / n
            vmuls(calc1, calc0, inv_n, repeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);

            // mean = sum(x / n)
            reduce_sum(calc1, cnt_per_token, norm_dim);
            pipe_barrier(PIPE_V);

            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

            // duplicate item
            duplicate_item(calc1, cnt_per_token, repeat_per_norm, norm_dim);
            pipe_barrier(PIPE_V);

            // x - mean
            vsub(calc0, calc0, calc1, repeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
        }

        if (variance) {
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + inCurr);
            auto gm_variance_float = (__gm__ float *)variance + loop;
#if defined(XLITE_ARCH_310P)
            float variance_value = *gm_variance_float;
            set_flag(PIPE_S, PIPE_V, EVENT_ID0 + inCurr);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0 + inCurr);
            vector_dup(in_variance_float[inCurr], variance_value, 1, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);
#else
            copy_gm_to_ubuf_align_b16(in_variance_float[inCurr], gm_variance_float, 0, 1,
                                      sizeof(float), 0, 0, 0, 0);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + inCurr);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + inCurr);
#endif
            vmuls(calc1, in_variance_float[inCurr], divTpSize, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + inCurr);
        } else {
            // x ^ 2
            vmul(calc1, calc0, calc0, repeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);

            if (normKind != NormKind::L2) {
                // x ^ 2 / n
                vmuls(calc1, calc1, inv_n, repeat, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            }

            // sum(x ^ 2)
            reduce_sum(calc1, cnt_per_token, norm_dim);
            pipe_barrier(PIPE_V);
        }

        if (useNorm) {
#if defined(XLITE_ARCH_310P)
            xlite_310p_set_mask(1);
#else
            SetMask(1);
#endif
            // sum(x ^ 2) + eps
            vadds(calc1, calc1, norm_eps, cnt_per_token, 1, 1, repeat_stride, repeat_stride);
            pipe_barrier(PIPE_V);

            // sqrt(sum(x ^ 2) + eps)
            vsqrt(calc1, calc1, cnt_per_token, 1, 1, repeat_stride, repeat_stride);
            pipe_barrier(PIPE_V);
            set_vector_mask((uint64_t)-1, (uint64_t)-1);

            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

            // duplicate item
            duplicate_item(calc1, cnt_per_token, repeat_per_norm, norm_dim);
            pipe_barrier(PIPE_V);

            // x / sqrt(sum(x ^ 2) + eps)
            vdiv(calc1, calc0, calc1, repeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);

            // x / sqrt(sum(x ^ 2) + eps) * weight
            if (weight) {
                vmul(calc1, weight_calc, calc1, repeat, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
            }

            // x / sqrt(sum(x ^ 2) + eps) * weight + bias
            if (bias) {
                vadd(calc1, bias_calc, calc1, repeat, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
            }

            // cast data type
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + outCurr);
            if (outFp32) {
                copy_ubuf_to_ubuf(out_float[outCurr], calc1, 0, 1, len_burst_float_per_norm, 0, 0);
            } else {
                convert_output(out[outCurr], calc1, repeat);
            }
            pipe_barrier(PIPE_V);
        } else {
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + outCurr);
            copy_ubuf_to_ubuf(out_float[outCurr], calc1, 0, 1, len_burst_float_per_norm, 0, 0);
            pipe_barrier(PIPE_V);
        }

#if defined(XLITE_ARCH_310P)
        if (!useNorm) {
            set_flag(PIPE_V, PIPE_S, EVENT_ID0 + outCurr);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0 + outCurr);
            if (output) {
                *gm_out_float = *out_float[outCurr];
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + outCurr);
            outCurr = 1 - outCurr;
            continue;
        }
#endif
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + outCurr);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + outCurr);

        if (useNorm && !outFp32) {
            if (output) {
                copy_ubuf_to_gm(gm_out_dtype, out[outCurr], 0, 1, len_burst, 0, 0);
            }
            if (need_cache) {
                auto *kcache_ptr = ((__gm__ Dtype *)(kcache)) + block * block_size * total_dim +
                                   block_offset * total_dim;
                copy_ubuf_to_gm(kcache_ptr, out[outCurr], 0, 1, len_burst, 0, 0);
            }
        } else if (useNorm && outFp32) {
            if (output) {
                copy_ubuf_to_gm_align_b16(gm_out_float, out_float[outCurr], 0, 1,
                                          total_dim * sizeof(float), 0, 0, 0, 0);
            }
        } else {
            if (output) {
                copy_ubuf_to_gm_align_b16(gm_out_float, out_float[outCurr], 0, 1, sizeof(float), 0,
                                          0, 0, 0);
            }
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + outCurr);
        outCurr = 1 - outCurr;
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
}

#define NORM_FUNC_DEFINE(dtype)                                                                   \
    extern "C" __global__ __aicore__ void norm_##dtype(                                           \
        GM_ADDR input, GM_ADDR addInOut, GM_ADDR weight, GM_ADDR bias, GM_ADDR out,               \
        uint32_t token_num, uint32_t norm_dim, float norm_eps, int kind, uint32_t cnt_per_token,  \
        uint32_t in_step, uint32_t out_step, uint32_t in_start_offset, uint32_t out_start_offset, \
        bool useNorm, GM_ADDR variance, uint32_t tpSize, bool outFp32)                            \
    {                                                                                             \
        norm<dtype>(input, addInOut, weight, bias, out, token_num, norm_dim, norm_eps, kind,      \
                    cnt_per_token, in_step, out_step, in_start_offset, out_start_offset, useNorm, \
                    variance, tpSize, nullptr, nullptr, 0, 0, nullptr, outFp32);                  \
    }
#else
#define NORM_FUNC_DEFINE(dtype)                                                                   \
    extern "C" __global__ __aicore__ void norm_##dtype(                                           \
        GM_ADDR input, GM_ADDR addInOut, GM_ADDR weight, GM_ADDR bias, GM_ADDR out,               \
        uint32_t token_num, uint32_t norm_dim, float norm_eps, int kind, uint32_t cnt_per_token,  \
        uint32_t in_step, uint32_t out_step, uint32_t in_start_offset, uint32_t out_start_offset, \
        bool useNorm, GM_ADDR variance, uint32_t tpSize, bool outFp32)                            \
    {                                                                                             \
    }
#endif
