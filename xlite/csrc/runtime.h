/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_RUNTIME_H_
#define _XLITE_RUNTIME_H_

#include <cstdint>
#include "base.h"
#include "paged_decode_310p.h"
#include "matmul_plan.h"
#include <map>
#include <tuple>

#define XLITE_DEFAULT_PORT 10266
#define XLITE_DEFAULT_COMM_OPTIMIZE_LEN 6144
#define XLITE_ACTIVE_TOKENS_RATIO_PER_EP_THRESHOLD 1024

typedef void *aclrtContext;
typedef void *aclrtNotify;
typedef void *aclrtStream;
typedef void *aclrtEvent;
typedef void *aclmdlRI;
typedef void *HcclComm;
class XTensorPool;
class XcclComm;

enum XModelAttnType {
    XMODEL_ATTN_MHA,
    XMODEL_ATTN_MLA,
    XMODEL_ATTN_DSA,
    XMODEL_ATTN_HYBRID,
    XMODEL_ATTN_CXA,
    XMODEL_ATTN_MAX_TYPE,
};

enum XModelLayerAttnType {
    XMODEL_LAYER_ATTN_FULL = 0,
    XMODEL_LAYER_ATTN_LINEAR = 1,
};

struct XModelAttnMeta {
    int version = 0;

    std::vector<uint32_t> lens;
    std::vector<uint32_t> cachedLens;

    /* only for version 0 */
    std::vector<std::vector<uint32_t>> blockTables;

    /* only for version 1 */
    XTensor vllmBlockTables;
    XTensor vllmSlotMapping;
    XTensor vllmPosition;
};

enum commType {
    TP,
    EP,
    DP,
    MAX_COMM_TYPE,
};

class XRuntime
{
public:
    XRuntime(uint32_t devid, size_t sizeMB = 0, uint32_t rankId = 0, uint32_t tpSize = 1,
             uint32_t dpSize = 1, uint32_t moeTpSize = 1, uint32_t moeEpSize = 1);
    virtual ~XRuntime(void);
    void Init(size_t sizeMB);
    void InitAttn(uint64_t maxBatchedTokens, uint64_t maxBatch, uint64_t maxSeqLen,
                  uint32_t blockSize, uint32_t indexTopK);
    void PrepareAttn(XModelAttnMeta &attnMeta, uint64_t maxBatchedTokens, uint64_t maxBatch,
                     uint64_t maxSeqLen, uint32_t nHeads, uint32_t nKVheads, uint32_t blockSize,
                     uint32_t hiddenSize, uint32_t nRoutedExperts, uint32_t defDpSize,
                     int inputDtype, int weightsDtype, uint32_t indexTopK);
    void Synchronize(void);
    void EventWaitCurrStream(aclrtStream currStream);
    void EventRecordCurrStream(aclrtStream currStream);
    void MemcpyH2D(void *dst, void *src, size_t size);
    void SetDecodeAttentionBackend(const std::string &backend);
    void SetMatmulOptimization(const std::string &mode);
    void SetMatmulPlan(int64_t m, int64_t n, int64_t k, int64_t chunk, bool direct, bool enabled);
    std::string matmulOptimization = "legacy";
    bool matmulDiagnostics = false;
    std::map<std::tuple<int64_t, int64_t, int64_t>, XMatmulPlan> matmulPlans;
    std::map<std::string, XMatmulStats> matmulStats;
    std::string decodeAttentionBackend = "legacy";
    uint64_t pagedDecodeRequests = 0, pagedDecodeLaunches = 0, pagedDecodeMergeLaunches = 0;
    uint64_t legacyDecodeRequests = 0, decodeKvGatherBytes = 0;
#ifdef XLITE_310P_LLM_FP16_POC
    const uint8_t *CausalMaskHost310P(void);
    void PreparePagedDecodeMetadata();
    void WaitMetadataUpload();
    void *pagedMetadata = nullptr;
    void *pagedScratch = nullptr;
    uint32_t pagedCount = 0, pagedPartitions = 1, pagedPartitionLength = 128;
#endif
    void MemcpyD2H(void *dst, void *src, size_t size);
    void MemcpyD2HAsync(void *dst, void *src, size_t size);
    void UpdateCoreNum(float blockDimUtilization);

    void SetCurrentContext();
    void NotifyWaitPeerStream();
    void NotifyRecordPeerStream();

    void PrepareCommBuffers(uint64_t maxBatch, uint32_t hiddenSize, uint32_t nRoutedExperts,
                            uint32_t defDpSize, int inputDtype, int weightsDtype);
    void CaptureHcclAllGather(void *send, void *recv, uint32_t m, int hcclDtype,
                              enum commType type);
    void RunHcclAllGatherInGraph(void *send, void *recv, uint32_t m, int hcclDtype,
                                 enum commType type);
    void AllGatherInGraph(void *send, void *recv, uint32_t m, int hcclDtype, enum commType type);
    [[nodiscard]] bool AllGatherInGraphActive(enum commType type) const;

    void CaptureHcclReduceScatter(void *send, void *recv, uint32_t m, int hcclDtype,
                                  enum commType type);
    void RunHcclReduceScatterInGraph(void *send, void *recv, uint32_t m, int hcclDtype,
                                     enum commType type);
    void ReduceScatterInGraph(void *send, void *recv, uint32_t m, int hcclDtype,
                              enum commType type);
    [[nodiscard]] bool ReduceScatterInGraphActive(enum commType type) const;

    int InitTensorPool(size_t sizeMB);
    XTensor &GetTensor(std::vector<size_t> shape, enum XDtype dtype, DebugSrcLoc loc);
    void PutTensor(XTensor &t);
    bool TensorInPool(XTensor &t);
    int64_t GetTensorOffset(XTensor &t);

    void ConfigureSwizzle(uint32_t swizzle, bool useSwizzleTable);
    [[nodiscard]] bool ForceSyncAclnn(void) const
    {
        return _forceSyncAclnn;
    }
    [[nodiscard]] bool ForceSyncMatmul(void) const
    {
        return _forceSyncMatmul;
    }
    [[nodiscard]] bool ForceSyncAttention(void) const
    {
        return _forceSyncAttention;
    }
    [[nodiscard]] bool StressWorkspaceReuse(void) const
    {
        return _stressWorkspaceReuse;
    }
    [[nodiscard]] bool SyncForwardBoundary(void) const
    {
        return _syncForwardBoundary;
    }
    void RecordAclnnLaunch(void)
    {
        _aclnnLaunches++;
    }
    void RecordAclnnWorkspace(void *ptr)
    {
        if (ptr != nullptr && ptr == _lastAclnnWorkspace) {
            _workspaceReuses++;
        }
        _lastAclnnWorkspace = ptr;
    }
    void RecordForcedSync(void)
    {
        _forcedSyncLaunches++;
    }
    [[nodiscard]] uint64_t AclnnLaunches(void) const
    {
        return _aclnnLaunches;
    }
    [[nodiscard]] uint64_t StreamSynchronizations(void) const
    {
        return _streamSynchronizations;
    }
    [[nodiscard]] uint64_t AttentionMetadataD2HBytes(void) const
    {
        return _attentionMetadataD2HBytes;
    }
    [[nodiscard]] uint64_t WorkspaceReuses(void) const
    {
        return _workspaceReuses;
    }
    [[nodiscard]] uint64_t ForcedSyncLaunches(void) const
    {
        return _forcedSyncLaunches;
    }
    [[nodiscard]] uint64_t ForwardBoundarySynchronizations(void) const
    {
        return _forwardBoundarySynchronizations;
    }
    void ResetRuntimeStats(void)
    {
        _aclnnLaunches = 0;
        _streamSynchronizations = 0;
        _attentionMetadataD2HBytes = 0;
        _workspaceReuses = 0;
        _forcedSyncLaunches = 0;
        _forwardBoundarySynchronizations = 0;
        _lastAclnnWorkspace = nullptr;
        pagedDecodeRequests = pagedDecodeLaunches = pagedDecodeMergeLaunches = 0;
        legacyDecodeRequests = decodeKvGatherBytes = 0;
        matmulStats.clear();
    }

    [[nodiscard]] virtual bool IsDummyRuntime() const
    {
        return false;
    }
    bool Inited(void)
    {
        return _inited;
    };
    uint32_t rankId(void)
    {
        return _rankId;
    };
    uint32_t tpSize(void)
    {
        return _tpSize;
    };
    uint32_t dpSize(void)
    {
        return _dpSize;
    };
    uint32_t moeTpSize(void)
    {
        return _moeTpSize;
    };
    uint32_t moeEpSize(void)
    {
        return _moeEpSize;
    };
    aclrtStream stream = nullptr;
    struct GraphCaptureEntry {
        aclmdlRI modelRI = nullptr;
        void *sendAddr = nullptr;
        void *recvAddr = nullptr;
    };
    std::vector<GraphCaptureEntry> _agGraphs;
    XTensor _agSendBuf;
    XTensor _agRecvBuf;
    uint64_t _agPerTokenBytes = 0;
    std::vector<GraphCaptureEntry> _rsGraphs;
    XTensor _rsSendBuf;
    XTensor _rsRecvBuf;
    uint64_t _rsPerTokenBytes = 0;
    uint32_t _rsHiddenSize = 0;
    uint32_t aicNum;
    uint32_t aivNum;
    uint32_t originAicNum;
    uint32_t originAivNum;
    uint32_t reportedAivNum;
    HcclComm _tpComm = nullptr;
    HcclComm _dpComm = nullptr;
    HcclComm _epComm = nullptr;
    uint32_t commOptimizeLen = XLITE_DEFAULT_COMM_OPTIMIZE_LEN;
    bool enableCommOptimize;
    XTensor hiddenStatePad;
    XTensor hiddenStateSlice;
    uint32_t batchedTokens;
    uint32_t defaultMatmulSwizzle = 0x600;
    bool disableSwizzleTable = false;
    bool enableMoEAllToAll = false;
    double activeTokensRatioPerEp = 1.0f;

    XcclComm *_tpXcclComm = nullptr;
    XcclComm *_dpXcclComm = nullptr;
    XcclComm *_epXcclComm = nullptr;

    // for multi-task parallel
    bool multiTaskParallel = false;
    uint32_t taskId = 0;
    aclrtNotify peerNotify = nullptr;
    aclrtNotify notify = nullptr;

    // ATTN
    bool _attnInitialized = false;
    bool _decodeStep = false;
    // Host-side: true when this step is decode (seqlen==1 and all cached_lens>0).
    // Avoids D2H sync via GetFirstAttnPosition in every linear layer.
    bool _linearDecodeStep = false;
    // Host copy of cached_lens from PrepareAttn. Linear layers clear conv/ssm
    // only for requests with cached==0 (fresh prefill); chunked-prefill
    // continuation (cached>0, multi-token) reuses state via recurrent GDN.
    std::vector<uint32_t> _cachedLensHost;
    // Retained for the batch=1 ACLNN correctness backend. The 910B launch path
    // continues to consume the device-side tensors below.
    std::vector<uint32_t> _lensHost;
    std::vector<uint32_t> _blockTablesHost;
#ifdef XLITE_310P_LLM_FP16_POC
    // Stable, page-locked sources for asynchronous attention-metadata H2D copies.
    // The public host vectors above remain the source of truth for the ACLNN
    // correctness backend and are shared by every layer in one forward.
    XTensor _positionPinnedHost;
    XTensor _slotMappingPinnedHost;
    XTensor _cachedLensPinnedHost;
    XTensor _lensPinnedHost;
    XTensor _queryStartLocPinnedHost;
    XTensor _blockTablesPinnedHost;
#endif
    uint32_t _maxNumBlocks;
    uint32_t _batch;
    uint32_t _tileSizeOfCachedKV;
    XTensor _attnPosition;     // uint64_t
    XTensor _attnBlockTables;  // uint32_t
    XTensor _attnSlotMapping;  // uint32_t
    XTensor _position;         // uint64_t
    XTensor _blockTables;      // uint32_t
    XTensor _slotMapping;      // uint32_t
    XTensor _cachedLens;       // uint32_t
    XTensor _lens;             // uint32_t
    XTensor _queryStartLoc;    // uint32_t
    XTensor _dsaTopkBuffer;    // int32_t, cross-layer shared topk
    bool _dsaTopkValid = false;

    // for MoE
    XTensor _tokensPerEpGroupAllEpHost;

    // DP metadata
    uint64_t currTokens = 1;
    // maxTokensDp: initialized as max tokens across DP ranks; updated to the token number
    // to pad to for MoE when _dpSize > 1 and _nRoutedExperts > 0 and !enableMoEAllToAll
    uint64_t maxTokensDp = 1;

protected:
    int GetNodeIps(void);
    int InitHcclComm(void);
    int InitXcclComm(void);
    void FiniXcclComm(void);
    uint32_t _devid;
    aclrtEvent _event = nullptr;
#ifdef XLITE_310P_LLM_FP16_POC
    // Keep the two cross-stream directions independent. Reusing one event for
    // alternating record/wait/reset chains can release the PyTorch consumer
    // before the preceding ACLNN MatMul work is complete.
    aclrtEvent _inputReadyEvent = nullptr;
    aclrtEvent _outputReadyEvent = nullptr;
    void *_causalMaskPinnedHost = nullptr;
    void *_pagedPinnedHost = nullptr;
    aclrtEvent _metadataUploadedEvent = nullptr;
    bool _metadataUploadPending = false;
#endif
    aclrtContext context = nullptr;
    bool _initOutside = false;
    bool _inited = false;
    bool _graphCommEnabled = true;
    bool _forceSyncAclnn = false;
    bool _forceSyncMatmul = false;
    bool _forceSyncAttention = false;
    bool _stressWorkspaceReuse = false;
    bool _syncForwardBoundary = false;
    uint64_t _aclnnLaunches = 0;
    uint64_t _streamSynchronizations = 0;
    uint64_t _attentionMetadataD2HBytes = 0;
    uint64_t _workspaceReuses = 0;
    uint64_t _forcedSyncLaunches = 0;
    uint64_t _forwardBoundarySynchronizations = 0;
    void *_lastAclnnWorkspace = nullptr;
    XTensorPool *_pool = nullptr;
    uint32_t _rankId;
    uint32_t _tpSize;
    uint32_t _dpSize;
    uint32_t _moeTpSize;
    uint32_t _moeEpSize;
    uint32_t _rankSize;
    uint32_t _nDevPerNode = 0;
    uint32_t _port = XLITE_DEFAULT_PORT;
    std::vector<std::string> _ips;
};

class XDummyRuntime : public XRuntime
{
public:
    using XRuntime::XRuntime;

    [[nodiscard]] bool IsDummyRuntime() const override
    {
        return true;
    }
    void InitDummyRuntime(size_t sizeMB);
    size_t maxUsedSize(void);

private:
    int InitDummyXcclComm(void);
};
#endif
