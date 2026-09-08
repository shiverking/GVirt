#include "paged_decode_common.h"

extern "C" __global__ __aicore__ void xlite_paged_decode_310p(
    GM_ADDR qkv, GM_ADDR key, GM_ADDR value, GM_ADDR metadata, GM_ADDR scratch,
    GM_ADDR result, uint32_t count, uint32_t partitions, uint32_t partitionLength)
{
    using namespace Paged310P;
    using namespace XlitePaged310P;
    set_atomic_none();
    set_mask_norm();
    auto *kvHalf = (__ubuf__ float16_t *)0;                 // 8192 bytes
    auto *qHalf = (__ubuf__ float16_t *)8192;              // 256 bytes
    auto *outHalf = (__ubuf__ float16_t *)8448;            // 256 bytes
    auto *kv = (__ubuf__ float *)8704;                    // 16384 bytes
    auto *q = (__ubuf__ float *)25088;                    // 512 bytes
    auto *state = (__ubuf__ float *)25600;                // 544 bytes
    auto *acc = state + 8;
    auto *temp = (__ubuf__ float *)26144;                 // 512 bytes
    auto *scores = (__ubuf__ float *)26656;               // 256 bytes
    static_assert(26912 <= UbBytes, "decode UB layout");
    auto *meta = (__gm__ uint32_t *)metadata;
    for (uint32_t task = block_idx; task < count * Heads * partitions; task += block_num) {
        const uint32_t part = task % partitions;
        const uint32_t head = (task / partitions) % Heads;
        const uint32_t requestIndex = task / (partitions * Heads);
        const uint32_t request = meta[requestIndex * RecordWords];
        const uint32_t row = meta[requestIndex * RecordWords + 1];
        const uint32_t length = meta[requestIndex * RecordWords + 2];
        uint32_t start = part * partitionLength;
        const uint32_t end = length < start + partitionLength ? length : start + partitionLength;
        mask();
        vector_dup(acc, 0.0f, 2, 1, 1, 8, 0);
        copy_gm_to_ubuf(qHalf, (__gm__ float16_t *)qkv + row * 4096 + head * 128,
                        0, 1, 8, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        vconv_f162f32(q, qHalf, 2, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        float maximum = -3.402823466e38f, sum = 0.0f;
        while (start < end) {
            uint32_t tokens = end - start;
            if (tokens > Tile) tokens = Tile;
            if (tokens > 128 - start % 128) tokens = 128 - start % 128;
            const uint32_t physical = meta[TableOffset + request * MaxBlocks + start / 128];
            const uint64_t offset = ((uint64_t(physical) * 128 + start % 128) * 8 + head / 2) * 128;
            load(kvHalf, (__gm__ float16_t *)key + offset, tokens);
            vconv_f162f32(kv, kvHalf, tokens * 2, 1, 1, 8, 4);
            pipe_barrier(PIPE_V);
            for (uint32_t t = 0; t < tokens; ++t) {
                vmul(kv + t * 128, kv + t * 128, q, 2, 1, 1, 1, 8, 8, 8);
            }
            pipe_barrier(PIPE_V);
            for (uint32_t half = 64; half > 0; half /= 2) {
                mask(half);
                vadd(kv, kv, kv + half, tokens, 1, 1, 1, 16, 16, 16);
                pipe_barrier(PIPE_V);
            }
            vs();
            float nextMaximum = maximum;
            for (uint32_t t = 0; t < tokens; ++t) {
                if (kv[t * 128] > nextMaximum) nextMaximum = kv[t * 128];
            }
            for (uint32_t t = 0; t < 64; ++t) scores[t] = -3.402823466e38f;
            for (uint32_t t = 0; t < tokens; ++t) scores[t] = kv[t * 128] - nextMaximum;
            scores[32] = sum > 0 ? maximum - nextMaximum : -3.402823466e38f;
            sv();
            mask();
            vexp(scores, scores, 1, 1, 1, 8, 8);
            vs();
            float alpha = scores[32];
            sum *= alpha;
            for (uint32_t t = 0; t < tokens; ++t) sum += scores[t];
            maximum = nextMaximum;
            sv();
            vmuls(acc, acc, alpha, 2, 1, 1, 8, 8);
            load(kvHalf, (__gm__ float16_t *)value + offset, tokens);
            vconv_f162f32(kv, kvHalf, tokens * 2, 1, 1, 8, 4);
            pipe_barrier(PIPE_V);
            for (uint32_t t = 0; t < tokens; ++t) {
                // M200 vmuls requires an unqualified scalar float, not a UB
                // element expression. The preceding V->S event makes scores visible.
                float weight = scores[t];
                vmuls(temp, kv + t * 128, weight, 2, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vadd(acc, acc, temp, 2, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
            }
            // Do not overwrite UB while Vector is still consuming the tile.
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            start += tokens;
        }
        if (partitions == 1) {
            output((__gm__ float16_t *)result + row * 2048 + head * 128, outHalf, acc, sum);
        } else {
            vs();
            state[0] = maximum;
            state[1] = sum;
            for (uint32_t p = 2; p < 8; ++p) state[p] = 0.0f;
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            copy_ubuf_to_gm((__gm__ float *)scratch + task * PartialFloats, state,
                            0, 1, PartialFloats / 8, 0, 0);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
        pipe_barrier(PIPE_ALL);
    }
}
