#!/usr/bin/env python3
"""Identify CANN 9.1 FRACTAL_NZ physical order from raw device bytes."""

from __future__ import annotations

import json


def main() -> int:
    import torch
    import torch_npu
    from xlite._C import Runtime, raw_device_copy_310p

    rows = 32
    cols = 32
    # Integers through 1023 are exactly representable by FP16, giving every
    # logical coordinate an unambiguous raw-storage signature.
    logical_cpu = torch.arange(rows * cols, dtype=torch.float16).view(rows, cols)
    logical_nd = logical_cpu.to("npu")
    physical_nz = torch_npu.npu_format_cast(logical_nd, 29)
    raw_nd = torch.empty((rows * cols,), dtype=torch.float16, device="npu")
    runtime = Runtime(0, 32)
    raw_device_copy_310p(runtime, physical_nz, raw_nd)
    raw_cpu = raw_nd.cpu()

    blocked = logical_cpu.view(rows // 16, 16, cols // 16, 16)
    candidates = {
        "k1_n1_n0_k0": blocked.permute(2, 0, 1, 3).contiguous().view(-1),
        "n1_k1_n0_k0": blocked.permute(0, 2, 1, 3).contiguous().view(-1),
        "k1_n1_k0_n0": blocked.permute(2, 0, 3, 1).contiguous().view(-1),
        "n1_k1_k0_n0": blocked.permute(0, 2, 3, 1).contiguous().view(-1),
    }
    matches = [name for name, expected in candidates.items()
               if torch.equal(raw_cpu, expected)]
    payload = {
        "logical_shape": [rows, cols],
        "npu_format": int(torch_npu.get_npu_format(physical_nz)),
        "matches": matches,
        "required_by_kernel": "k1_n1_n0_k0",
        "raw_prefix": raw_cpu[:32].tolist(),
    }
    print(json.dumps(payload))
    if matches != ["k1_n1_n0_k0"]:
        raise AssertionError(
            "CANN FRACTAL_NZ physical layout does not match the compiled "
            f"AscendC contract: {matches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
