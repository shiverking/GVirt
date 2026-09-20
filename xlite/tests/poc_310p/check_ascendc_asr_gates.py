#!/usr/bin/env python3
"""Static stop gates for the Ascend310P3 Qwen3-ASR AscendC backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


XLITE_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = XLITE_ROOT / "doc" / "310p_asr_ascendc"
KERNEL_ROOT = XLITE_ROOT / "csrc" / "kernels" / "310p"
MANIFEST = DOC_ROOT / "shared_assets.json"
RESOURCES = DOC_ROOT / "kernel_resources.json"

FORBIDDEN = {
    "high-level Matmul header": re.compile(r"lib/matmul_intf\.h"),
    "high-level Matmul type": re.compile(
        r"\b(?:MatmulImpl|MatmulType|MatmulCallBackFunc|Matmul)\s*<"
    ),
    "high-level Matmul policy": re.compile(r"\b(?:CFG_NORM|CFG_MDL)\b"),
    "unsupported DataCopyPad": re.compile(r"\bDataCopyPad\s*\("),
    "unsupported LoadDataWithTranspose": re.compile(
        r"\bLoadDataWithTranspose\s*\("
    ),
    "910B architecture macro": re.compile(r"__CCE_AICORE__\s*==\s*220"),
}


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _production_sources() -> list[Path]:
    return sorted(KERNEL_ROOT.glob("asr_*.cpp"))


def _check_sources(errors: list[str]) -> None:
    for path in _production_sources():
        text = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN.items():
            if pattern.search(text):
                errors.append(f"{path.relative_to(XLITE_ROOT)}: {label}")
        # The already-shipped projection kernel predates the dynamic-event rule.
        # Do not allow that debt to spread to any subsequent ASR kernel.
        if path.stem != "asr_m200_projection_fp16" and re.search(r"\bEVENT_ID\d+\b", text):
            errors.append(f"{path.relative_to(XLITE_ROOT)}: fixed event identifier")
        if "__NPU_ARCH__" not in text or "2002" not in text:
            errors.append(
                f"{path.relative_to(XLITE_ROOT)}: missing guarded __NPU_ARCH__=2002 gate"
            )


def _check_resources(require_resources: bool, errors: list[str]) -> None:
    data = _load(RESOURCES)
    budget = int(data["ub_design_budget_bytes"])
    expected = set(data["kernels"])
    present = {path.stem for path in _production_sources()}
    unknown = present - expected
    if unknown:
        errors.append(f"production kernels absent from resource manifest: {sorted(unknown)}")
    for name, item in data["kernels"].items():
        if name not in present:
            continue
        ub_bytes = item.get("ub_bytes")
        effective_ub_bytes = item.get(
            "ub_bytes_including_nd2nz_reserve", ub_bytes)
        event_pairs = item.get("event_pairs")
        if require_resources and (not isinstance(ub_bytes, int) or not isinstance(event_pairs, int)):
            errors.append(f"{name}: exact ub_bytes and event_pairs are required")
        if (isinstance(effective_ub_bytes, int) and
                effective_ub_bytes > budget):
            errors.append(
                f"{name}: UB use {effective_ub_bytes} exceeds design "
                f"budget {budget}")


def _check_shared(root: Path, errors: list[str]) -> None:
    manifest = _load(MANIFEST)
    for item in manifest["assets"]:
        path = root / item["path"]
        if not path.is_file():
            errors.append(f"shared asset missing: {path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            errors.append(f"shared asset hash mismatch: {item['path']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-root", type=Path)
    parser.add_argument("--require-resources", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    manifest = _load(MANIFEST)
    target = manifest["target"]
    if target["soc"] != "Ascend310P3" or target["npu_arch"] != 2002:
        errors.append("asset manifest target must remain Ascend310P3/__NPU_ARCH__=2002")
    if int(target["ub_design_budget_bytes"]) > 192 * 1024:
        errors.append("UB design budget must not exceed the conservative 192 KiB limit")
    _check_sources(errors)
    _check_resources(args.require_resources, errors)
    if args.shared_root is not None:
        _check_shared(args.shared_root, errors)

    if errors:
        print("Ascend310P3 ASR source gates FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        "Ascend310P3 ASR source gates PASS: "
        f"{len(_production_sources())} production kernel source(s), "
        f"{len(manifest['assets'])} pinned design asset(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
