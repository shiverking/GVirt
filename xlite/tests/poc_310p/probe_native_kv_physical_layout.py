#!/usr/bin/env python3
"""Capture the raw Native 5D/NZ KV-cache write map on Ascend310P3.

The probe deliberately does not guess a layout formula.  It writes unique,
exactly representable FP16 landmarks through Native reshape-and-cache, copies
the storage bytes without a format conversion, and reports each raw index.
The resulting JSON is the input to the later AscendC NZ cache writer design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BLOCK_SIZE = 128
NUM_KV_HEADS = 8
HEAD_DIM = 128
NZ_FORMAT = 29

# Cover both physical blocks, block edges, all head extremes, and dimensions
# on either side of the 16-element fractal boundaries.
LANDMARKS = (
    (0, 0, 0),
    (0, 7, 127),
    (1, 1, 1),
    (15, 2, 15),
    (16, 3, 16),
    (63, 4, 31),
    (127, 5, 63),
    (128, 6, 64),
    (129, 7, 126),
    (255, 0, 127),
)


def _capture(name: str, marker_base: int, cache, slots, runtime,
             raw_device_copy_310p, torch) -> dict[str, object]:
    expected: list[dict[str, int]] = []
    for ordinal, (slot, head, dim) in enumerate(LANDMARKS):
        marker = marker_base + ordinal
        expected.append({
            "marker": marker,
            "slot": slot,
            "block": slot // BLOCK_SIZE,
            "block_offset": slot % BLOCK_SIZE,
            "head": head,
            "dim": dim,
        })

    raw = torch.empty((cache.numel(),), dtype=torch.float16, device="npu")
    raw_device_copy_310p(runtime, cache, raw)
    raw_cpu = raw.cpu()
    observations = []
    for item in expected:
        indices = (raw_cpu == float(item["marker"])).nonzero().flatten().tolist()
        if len(indices) != 1:
            raise AssertionError(
                f"{name} marker {item['marker']} occurs {len(indices)} times: {indices[:8]}"
            )
        observations.append({**item, "raw_index": int(indices[0])})
    nonzero = int(torch.count_nonzero(raw_cpu))
    if nonzero != len(expected):
        raise AssertionError(
            f"{name} raw cache contains {nonzero} nonzero values, expected {len(expected)}"
        )
    return {"name": name, "observations": observations, "nonzero_values": nonzero}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    import torch
    import torch_npu
    from xlite._C import Runtime, raw_device_copy_310p

    slots = sorted({slot for slot, _, _ in LANDMARKS})
    slot_mapping = torch.tensor(slots, dtype=torch.int32, device="npu")
    source_shape = (len(slots), NUM_KV_HEADS, HEAD_DIM)
    key_cpu = torch.zeros(source_shape, dtype=torch.float16)
    value_cpu = torch.zeros_like(key_cpu)
    cache_shape = (2, NUM_KV_HEADS * HEAD_DIM // 16, BLOCK_SIZE, 16)
    key_cache = torch_npu.empty_with_format(
        size=cache_shape, dtype=torch.float16, device="npu", acl_format=NZ_FORMAT
    )
    value_cache = torch_npu.empty_with_format(
        size=cache_shape, dtype=torch.float16, device="npu", acl_format=NZ_FORMAT
    )
    key_cache.zero_()
    value_cache.zero_()

    # Values 257..778 are exactly representable in FP16 and distinct from the
    # zero sentinel.  No device selection is performed here: the caller owns
    # the current logical-device context.
    for ordinal, (slot, head, dim) in enumerate(LANDMARKS):
        row = slots.index(slot)
        key_cpu[row, head, dim] = 257 + ordinal
        value_cpu[row, head, dim] = 769 + ordinal
    key = key_cpu.to("npu")
    value = value_cpu.to("npu")
    torch_npu._npu_reshape_and_cache(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_indices=slot_mapping,
    )
    torch.npu.synchronize()

    runtime = Runtime(0, 32)
    # _capture only reads raw storage; the Native operator above is the sole
    # cache writer under test.
    key_map = _capture(
        "key", 257, key_cache, slots, runtime, raw_device_copy_310p, torch
    )
    value_map = _capture(
        "value", 769, value_cache, slots, runtime, raw_device_copy_310p, torch
    )
    payload = {
        "soc_contract": "Ascend310P3/CANN-9.1-beta1",
        "writer": "torch_npu._npu_reshape_and_cache",
        "logical_source_shape": list(source_shape),
        "cache_shape": list(cache_shape),
        "cache_format": int(torch_npu.get_npu_format(key_cache)),
        "block_size": BLOCK_SIZE,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "key": key_map,
        "value": value_map,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
