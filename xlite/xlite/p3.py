"""Version-bound, offline-selected P3 MatMul policies; no hot-path tuning."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import warnings

SHAPES = ((4096, 2048), (2048, 2048), (12288, 2048), (2048, 6144), (151936, 2048))
CHUNKS = (12288, 24576, 49152, 151936)
SYNC_KEYS = ("force_sync_aclnn", "force_sync_matmul", "force_sync_attention", "sync_forward_boundary")


def fingerprint(runtime):
    from xlite._C import get_build_info
    roots = [os.getenv("ASCEND_CANN_PACKAGE_PATH"), os.getenv("ASCEND_HOME_PATH"),
             "/usr/local/Ascend/ascend-toolkit/latest", "/usr/local/Ascend/cann/latest"]
    version = None
    for root in filter(None, roots):
        for suffix in ("version.info", "aarch64-linux/ascend_toolkit_install.info",
                       "ascend_toolkit_install.info"):
            file = Path(root) / suffix
            if file.is_file():
                lines = [line.strip() for line in file.read_text(errors="replace").splitlines()
                         if "version" in line.lower()]
                if lines:
                    version = lines
                    break
        if version:
            break
    if version is None:
        raise RuntimeError("Cannot identify CANN package version; set ASCEND_CANN_PACKAGE_PATH to the active installation")
    build = dict(get_build_info())
    if not build.get("p3_aclnn"):
        raise RuntimeError("installed Xlite extension has no P3 support")
    stats = runtime.get_stats()
    data = {"build": build, "cann_version": version,
            "torch_npu": importlib.metadata.version("torch-npu"),
            "sync_policy": {key: stats[key] for key in SYNC_KEYS},
            "shapes": [list(shape) for shape in SHAPES]}
    data["id"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return data


def validate_policy(policy, current):
    if policy.get("schema") != 1 or policy.get("fingerprint") != current:
        return False
    seen = set()
    for entry in policy.get("entries", []):
        m, n, k = entry["m"], entry["n"], entry["k"]
        if not 1 <= m <= 20 or (n, k) not in SHAPES or entry["chunk"] not in CHUNKS:
            raise ValueError("invalid P3 policy shape/chunk")
        if type(entry["direct"]) is not bool or type(entry["enabled"]) is not bool:
            raise ValueError("invalid P3 policy boolean")
        if (m, n, k) in seen:
            raise ValueError("duplicate P3 policy shape")
        seen.add((m, n, k))
    if not seen:
        raise ValueError("empty P3 policy")
    return True


def configure(runtime, mode="legacy", policy_path=None):
    if mode not in ("legacy", "p3_aclnn"):
        raise ValueError("invalid matmul optimization")
    if policy_path and mode != "p3_aclnn":
        raise ValueError("MatMul policy requires p3_aclnn")
    runtime.set_matmul_optimization(mode)
    if not policy_path:
        return {"mode": mode, "policy": "default12288", "validated": False}
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    current = fingerprint(runtime)
    if not validate_policy(policy, current):
        warnings.warn("P3 policy/environment mismatch: using 12288 default, no online tuning", stacklevel=2)
        return {"mode": mode, "policy": "default12288", "validated": False}
    # Unmeasured shapes keep the original implementation, not an inferred winner.
    for m in range(1, 21):
        for n, k in SHAPES:
            runtime.set_matmul_plan(m, n, k, 12288, False, False)
    for entry in policy["entries"]:
        runtime.set_matmul_plan(*(entry[key] for key in ("m", "n", "k", "chunk", "direct", "enabled")))
    return {"mode": mode, "policy": str(policy_path), "validated": True, "fingerprint": current}
