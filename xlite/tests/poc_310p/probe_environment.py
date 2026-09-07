#!/usr/bin/env python3
"""Record the target 310P/CANN environment before running hardware gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


ACLNN_HEADERS = (
    "aclnn_matmul.h",
    "aclnn_prompt_flash_attention.h",
    "aclnn_incre_flash_attention.h",
)


def _command(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": str(error)}
    return {
        "available": True,
        "returncode": result.returncode,
        "stdout": result.stdout[-12000:],
        "stderr": result.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cann-path",
        type=Path,
        default=Path(os.environ.get(
            "ASCEND_CANN_PACKAGE_PATH", "/usr/local/Ascend/ascend-toolkit/latest")),
    )
    parser.add_argument("--report", type=Path, default=Path("poc_310p_environment.json"))
    args = parser.parse_args()

    headers: dict[str, object] = {}
    for name in ACLNN_HEADERS:
        path = args.cann_path / "include" / "aclnnop" / name
        headers[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        }
    version_files = {}
    for relative in ("version.info", "ascend_toolkit_install.info"):
        matches = list(args.cann_path.glob(f"**/{relative}"))
        version_files[relative] = {
            str(path): path.read_text(encoding="utf-8", errors="replace") for path in matches[:8]
        }

    report = {
        "expected_soc_version": "ascend310p3",
        "expected_npu_arch": 2002,
        "npu_arch_verification": (
            "enforced by csrc/kernels/kernel_macro.h during AscendC device compilation"
        ),
        "cann_path": str(args.cann_path),
        "versions": version_files,
        "aclnn_headers": headers,
        "aclnn_libraries": {
            name: [str(path) for path in (args.cann_path / "lib64").glob(f"lib{name}.so*")]
            for name in ("nnopbase", "opapi")
        },
        "npu_smi": _command(["npu-smi", "info"]),
        "build_profile": {
            "ub_bytes": 192 * 1024,
            "swizzle": False,
            "core_assigner": False,
            "dtype": "float16",
            "batch": 1,
            "tp": 1,
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    headers_ok = all(item["exists"] for item in headers.values())
    libs_ok = all(report["aclnn_libraries"].values())
    return 0 if headers_ok and libs_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
