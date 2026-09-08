#include "paged_decode_common.h"

extern "C" __global__ __aicore__ void xlite_paged_decode_merge_310p(
    GM_ADDR metadata, GM_ADDR scratch, GM_ADDR result, uint32_t count)
{
    using namespace Paged310P;
    using namespace XlitePaged310P;
    set_atomic_none();
    set_mask_norm();
    auto *states = (__ubuf__ float *)0;     // 2176 bytes
    auto *scores = (__ubuf__ float *)2176;  // 256 bytes
    auto *acc = (__ubuf__ float *)2432;     // 512 bytes
    auto *temp = (__ubuf__ float *)2944;    // 512 bytes
    auto *half = (__ubuf__ float16_t *)3456;
    auto *meta = (__gm__ uint32_t *)metadata;
    for (uint32_t task = block_idx; task < count * Heads; task += block_num) {
        copy_gm_to_ubuf(states, (__gm__ float *)scratch + task * 4 * PartialFloats,
                        0, 1, 4 * PartialFloats / 8, 0, 0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        float maximum = -3.402823466e38f;
        for (uint32_t p = 0; p < 4; ++p) {
            if (states[p * PartialFloats + 1] > 0 && states[p * PartialFloats] > maximum)
                maximum = states[p * PartialFloats];
        }
        for (uint32_t p = 0; p < 64; ++p) scores[p] = -3.402823466e38f;
        for (uint32_t p = 0; p < 4; ++p) {
            if (states[p * PartialFloats + 1] > 0) scores[p] = states[p * PartialFloats] - maximum;
        }
        sv();
        mask();
        vexp(scores, scores, 1, 1, 1, 8, 8);
        vector_dup(acc, 0.0f, 2, 1, 1, 8, 0);
        vs();
        float sum = 0.0f;
        for (uint32_t p = 0; p < 4; ++p) {
            if (states[p * PartialFloats + 1] == 0) continue;
            const float factor = scores[p];
            sum += factor * states[p * PartialFloats + 1];
            vmuls(temp, states + p * PartialFloats + 8, factor, 2, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vadd(acc, acc, temp, 2, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
        }
        const uint32_t row = meta[(task / Heads) * RecordWords + 1];
        output((__gm__ float16_t *)result + row * 2048 + (task % Heads) * 128, half, acc, sum);
        pipe_barrier(PIPE_ALL);
    }
}
