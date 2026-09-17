# ASR paged-staging/MMAD same-stream interaction probe

This probe launches the already validated AIV paged-KV staging kernel followed
by the QK and PV AIC MMAD probes on one ACL stream with no synchronization
between launches. Fixed device scratch connects the kernels. One final stream
synchronization protects host validation only.

It checks the interaction contracts that isolated probes cannot establish:

- staged `K[16,128]` is consumed directly by QK MMAD;
- staged `V^T[128,16]` is consumed directly by PV MMAD;
- AIV-to-AIC stream order is sufficient without host waits;
- scratch/output guards and source cache remain unchanged outside payloads;
- partial and cross-physical-block tiles preserve zero padding.

This is not the production implementation: its fixed GM scratch is intentionally
retained as an observable boundary. Passing this gate authorizes a subsequent
unified-core candidate that moves the same layouts through UB/L1 without the GM
round trip and adds FP32 online-softmax state.
