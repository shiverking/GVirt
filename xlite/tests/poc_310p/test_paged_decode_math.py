"""CPU oracle for the P2 partition/online-softmax equations, not a device test."""
import unittest

import numpy as np


def partitioned(q, k, v, batch_max_length):
    partitions = 4 if batch_max_length > 512 else 1
    width = ((batch_max_length + partitions * 128 - 1) // (partitions * 128)) * 128
    states = []
    for part in range(partitions):
        maximum, total = np.float32(-np.finfo(np.float32).max), np.float32(0)
        acc = np.zeros(128, np.float32)
        for start in range(part * width, min(len(k), (part + 1) * width), 32):
            end = min(start + 32, len(k), (part + 1) * width)
            scores = k[start:end] @ q
            next_max = max(maximum, scores.max())
            alpha = np.exp(maximum - next_max) if total else np.float32(0)
            weights = np.exp(scores - next_max)
            acc = acc * alpha + weights @ v[start:end]
            total = total * alpha + weights.sum()
            maximum = next_max
        if total:
            states.append((maximum, total, acc))
    maximum = max(state[0] for state in states)
    denominator = sum(total * np.exp(m - maximum) for m, total, _ in states)
    return sum(acc * np.exp(m - maximum) for m, _, acc in states) / denominator


class PagedMathTest(unittest.TestCase):
    def test_partition_boundaries_and_empty_partitions(self):
        rng = np.random.default_rng(310)
        for length in (1, 16, 127, 128, 129, 512, 513, 1025, 2048):
            for batch_max in (length, 2048):
                with self.subTest(length=length, batch_max=batch_max):
                    q = rng.normal(size=128).astype(np.float32) / np.sqrt(128)
                    k = rng.normal(size=(length, 128)).astype(np.float32)
                    v = rng.normal(size=(length, 128)).astype(np.float32)
                    scores = k @ q
                    weights = np.exp(scores - scores.max())
                    expected = (weights @ v) / weights.sum()
                    np.testing.assert_allclose(partitioned(q, k, v, batch_max), expected,
                                               rtol=2e-5, atol=2e-6)

    def test_paged_gqa_addressing_and_reuse(self):
        rng = np.random.default_rng(42)
        cache = np.full((40, 128, 8, 128), np.nan, np.float16)
        for round_id in range(2):
            # Reassign the same physical blocks to different logical requests.
            table = rng.permutation(40).reshape(20, 2)
            for request in range(20):
                for token in range(129):
                    cache[table[request, token // 128], token % 128] = request + round_id
            for request in reversed(range(20)):
                for token in (0, 127, 128):
                    for q_head in range(16):
                        physical = table[request, token // 128]
                        offset = ((physical * 128 + token % 128) * 8 + q_head // 2) * 128
                        np.testing.assert_array_equal(cache.reshape(-1)[offset:offset + 128],
                                                      np.full(128, request + round_id))


if __name__ == "__main__":
    unittest.main()
