/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_RUNTIME_H_
#define _XLITE_RUNTIME_H_

#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include "base.h"

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

// CXA 5-tuple block-size indices: (indexer_state, indexer_k, compress_kv, state, swa_kv).
enum CxaCacheIndex {
    CXA_INDEXER_STATE = 0,
    CXA_INDEXER_K = 1,
    CXA_COMPRESS_KV = 2,
    CXA_STATE = 3,
    CXA_SWA_KV = 4,
};

struct XModelAttnMeta {
    int version = 0;
    enum XModelAttnType attnType = XMODEL_ATTN_MHA;

    std::vector<uint32_t> lensCpu;
    std::vector<uint32_t> cachedLensCpu;

    /* only for version 0 */
    std::vector<std::vector<uint32_t>> blockTablesCpu;

    /* for version 1/2 */
    XTensor position;

    /* only for version 2 */
    std::vector<XTensor> blockTables;
    std::vector<XTensor> slotMapping;
    XTensor queryStartLoc;
    XTensor lens;
    XTensor cachedLens;
};

enum commType {
    TP,
    EP,
    DP,
    MAX_COMM_TYPE,
};

#ifdef XLITE_ARCH_310P
enum class XMatmulBackend310P {
    ACLNN,
    M200_ASR,
};
enum class XDecodeAttentionBackend310P {
    LEGACY,
    BATCHED_ACLNN,
    NATIVE_ATB,
    DIRECT_ATB,
};
#endif

class XRuntime
{
public:
    XRuntime(uint32_t devid, size_t sizeMB = 0, uint32_t rankId = 0, uint32_t tpSize = 1,
             uint32_t dpSize = 1, uint32_t moeTpSize = 1, uint32_t moeEpSize = 1);
    virtual ~XRuntime(void);
    void Init(size_t sizeMB);
    void InitAttn(XModelAttnMeta &attnMeta, uint64_t maxBatchedTokens, uint64_t maxBatch,
                  uint64_t maxSeqLen, const std::vector<uint32_t> &blockSizes, uint32_t indexTopK);
    void PrepareAttn(XModelAttnMeta &attnMeta, uint64_t maxBatchedTokens, uint64_t maxBatch,
                     uint64_t maxSeqLen, uint32_t nHeads, uint32_t nKVheads,
                     std::vector<uint32_t> blockSizes, uint32_t hiddenSize, uint32_t nRoutedExperts,
                     uint32_t defDpSize, int inputDtype, int weightsDtype, uint32_t indexTopK);
    void Synchronize(void);
    void EventWaitCurrStream(aclrtStream currStream);
    void EventRecordCurrStream(aclrtStream currStream);
    void MemcpyH2D(void *dst, void *src, size_t size);
#ifdef XLITE_310P_LLM_FP16_POC
    const uint8_t *CausalMaskHost310P(void);
#endif
    void MemcpyD2H(void *dst, void *src, size_t size);
    void MemcpyD2HAsync(void *dst, void *src, size_t size);
    void UpdateCoreNum(float blockDimUtilization);
#ifdef XLITE_DEBUG_ON
    void VerifyAttnMetaV2(const XModelAttnMeta &attnMeta, std::vector<uint32_t> blockSizes);
#endif

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
#ifdef XLITE_ARCH_310P
    void SetMatmulBackend310P(const std::string &backend);
    void SetDecodeAttentionBackend310P(const std::string &backend);
    void SetAclnnMatmulAsync310P(bool enabled)
    {
        _aclnnMatmulAsync310P = enabled;
    }
    void SetDirectAtbSetupReuse310P(bool enabled)
    {
        _directAtbSetupReuse310P = enabled;
    }
    void SetBatchedPrefillAttention310P(bool enabled)
    {
        _enableBatchedPrefillAttention310P = enabled;
    }
    [[nodiscard]] const char *MatmulBackend310PName(void) const;
    [[nodiscard]] bool UseAclnnMatmulAsync310P(void) const
    {
        return _aclnnMatmulAsync310P;
    }
    void ReapAclnnMatmulLeases310P(bool waitOldest);
    void RetireAclnnMatmulResources310P(std::vector<XTensor *> tensors,
                                        std::function<void()> cleanup,
                                        bool lmHead);
    [[nodiscard]] bool UseM200Matmul310P(void) const
    {
        return _matmulBackend310P == XMatmulBackend310P::M200_ASR;
    }
    [[nodiscard]] bool UseBatchedDecodeAttention310P(void) const
    {
        return _decodeAttentionBackend310P == XDecodeAttentionBackend310P::BATCHED_ACLNN;
    }
    [[nodiscard]] bool UseNativeAtbDecodeAttention310P(void) const
    {
        return _decodeAttentionBackend310P == XDecodeAttentionBackend310P::NATIVE_ATB;
    }
    [[nodiscard]] bool UseDirectAtbDecodeAttention310P(void) const
    {
        return _decodeAttentionBackend310P == XDecodeAttentionBackend310P::DIRECT_ATB;
    }
    [[nodiscard]] bool UseDirectAtbSetupReuse310P(void) const
    {
        return _directAtbSetupReuse310P;
    }
    [[nodiscard]] bool UseBatchedPrefillAttention310P(void) const
    {
        return _enableBatchedPrefillAttention310P;
    }
    [[nodiscard]] bool UseNativeKvDecodeAttention310P(void) const
    {
        return UseNativeAtbDecodeAttention310P() || UseDirectAtbDecodeAttention310P();
    }
    using NativeAtbAttentionCallback = std::function<bool(
        XTensor &, XTensor &, XTensor &, XTensor &, XTensor &, XTensor &, XTensor &,
        uint32_t, uint32_t, uint32_t, uint32_t)>;
    NativeAtbAttentionCallback nativeAtbAttentionCallback;
    using NativeAtbRopeStageCallback =
        std::function<bool(XTensor &, uint32_t, void *&, void *&, void *&)>;
    NativeAtbRopeStageCallback nativeAtbRopeStageCallback;
    void RecordM200Matmul310P(uint32_t m)
    {
        ++m200MatmulRequests;
        if (m < m200MatmulRequestsByM.size()) {
            ++m200MatmulRequestsByM[m];
        }
    }
    void RecordAclnnMatmul310P(uint32_t m)
    {
        ++aclnnMatmulRequests;
        if (m < aclnnMatmulRequestsByM.size()) {
            ++aclnnMatmulRequestsByM[m];
        }
    }
    void RecordAclnnMatmulSynchronization(bool lmHead)
    {
        ++_aclnnMatmulSynchronizations;
        if (lmHead) {
            ++_lmHeadSynchronizations;
        }
    }
    void RecordBatchedPrefillAttention(uint32_t requests)
    {
        _batchedPrefillAttentionRequests += requests;
        ++_batchedPrefillAttentionLaunches;
    }
#endif
    void RecordAttentionAclnnLaunch(void)
    {
        ++_attentionAclnnLaunches;
    }
    void RecordAttentionWorkspace(void *ptr)
    {
        if (ptr != nullptr && ptr == _lastAttentionWorkspace) {
            ++_attentionWorkspaceReuses;
        }
        _lastAttentionWorkspace = ptr;
    }
    void RecordAttentionForcedSync(void)
    {
        ++_attentionForcedSynchronizations;
    }
#ifdef XLITE_ARCH_310P
    void RecordBatchedDecodeAttention(uint32_t requests)
    {
        _batchedDecodeAttentionRequests += requests;
        ++_batchedDecodeAttentionLaunches;
    }
    void RecordLegacyAttentionRequest(bool decode)
    {
        ++_legacyAttentionRequests;
        if (decode) {
            ++_legacyDecodeAttentionRequests;
        } else {
            ++_legacyPrefillAttentionRequests;
        }
    }
    void RecordDecodeKvGatherBytes(uint64_t bytes)
    {
        _decodeKvGatherBytes += bytes;
    }
    void RecordNativeAtbAttention(uint32_t requests)
    {
        _nativeAtbDecodeRequests += requests;
        ++_nativeAtbDecodeLaunches;
    }
    void RecordNativeAtbCacheWrite(void)
    {
        ++_nativeAtbCacheWrites;
    }
    void RecordNativeAtbStagingBytes(uint64_t bytes)
    {
        _nativeAtbStagingBytes += bytes;
    }
    void RecordDirectAtbSetup(void)
    {
        ++_directAtbSetupCount;
    }
    void RecordDirectAtbSetupReuse(void)
    {
        ++_directAtbSetupReuseCount;
    }
    void RecordDirectAtbStagingBytes(uint64_t bytes)
    {
        _directAtbStagingBytes += bytes;
    }
    void RecordDirectAtbFusedRopeStaging(uint64_t bytes)
    {
        _directAtbFusedRopeStagingBytes += bytes;
    }
    void RecordDirectAtbExecute(bool attention, uint32_t requests = 0)
    {
        ++_directAtbExecuteCount;
        if (attention) {
            _directAtbDecodeRequests += requests;
            ++_directAtbAttentionLaunches;
        } else {
            ++_directAtbReshapeLaunches;
        }
    }
    void RecordDirectAtbMixedDecode(uint32_t requests)
    {
        _directAtbMixedDecodeRequests += requests;
        ++_directAtbMixedDecodeLaunches;
    }
    void RecordDirectAtbCompact(uint64_t bytes)
    {
        ++_directAtbCompactLaunches;
        _directAtbCompactBytes += bytes;
    }
    void RecordDirectAtbScatter(uint64_t bytes)
    {
        ++_directAtbScatterLaunches;
        _directAtbScatterBytes += bytes;
    }
    void RecordDirectAtbMetadataH2D(uint64_t bytes, uint64_t launches = 1)
    {
        _directAtbMetadataH2DLaunches += launches;
        _directAtbMetadataH2DBytes += bytes;
    }
    void RecordDirectAtbPlan(bool reused, bool attention)
    {
        if (reused) {
            ++_directAtbPlanReuses;
            if (attention) {
                ++_directAtbPagedPlanReuses;
            } else {
                ++_directAtbReshapePlanReuses;
            }
        } else {
            ++_directAtbPlanRebuilds;
            if (attention) {
                ++_directAtbPagedPlanRebuilds;
            } else {
                ++_directAtbReshapePlanRebuilds;
            }
        }
    }
#endif
    [[nodiscard]] bool ForceSyncAttention(void) const
    {
        return _forceSyncAttention;
    }
    [[nodiscard]] uint64_t StreamSynchronizations(void) const
    {
        return _streamSynchronizations;
    }
    [[nodiscard]] uint64_t PrepareAttnSynchronizations(void) const
    {
        return _prepareAttnSynchronizations;
    }
    [[nodiscard]] uint64_t AttentionMetadataD2HBytes(void) const
    {
        return _attentionMetadataD2HBytes;
    }
    [[nodiscard]] uint64_t AttentionAclnnLaunches(void) const
    {
        return _attentionAclnnLaunches;
    }
    [[nodiscard]] uint64_t AttentionForcedSynchronizations(void) const
    {
        return _attentionForcedSynchronizations;
    }
    [[nodiscard]] uint64_t AttentionWorkspaceReuses(void) const
    {
        return _attentionWorkspaceReuses;
    }
#ifdef XLITE_ARCH_310P
    [[nodiscard]] uint64_t BatchedDecodeAttentionRequests(void) const
    {
        return _batchedDecodeAttentionRequests;
    }
    [[nodiscard]] uint64_t BatchedPrefillAttentionRequests(void) const
    {
        return _batchedPrefillAttentionRequests;
    }
    [[nodiscard]] uint64_t BatchedPrefillAttentionLaunches(void) const
    {
        return _batchedPrefillAttentionLaunches;
    }
    [[nodiscard]] uint64_t PrefillShapeForwardCalls(void) const
    {
        return _prefillShapeForwardCalls;
    }
    [[nodiscard]] uint64_t PrefillShapeRequests(void) const { return _prefillShapeRequests; }
    [[nodiscard]] uint64_t PrefillExactGroupableRequests(void) const
    {
        return _prefillExactGroupableRequests;
    }
    [[nodiscard]] uint64_t PrefillActualQueryTokens(void) const
    {
        return _prefillActualQueryTokens;
    }
    [[nodiscard]] uint64_t PrefillPaddedQueryTokens(void) const
    {
        return _prefillPaddedQueryTokens;
    }
    [[nodiscard]] const std::map<std::string, uint64_t> &PrefillShapeHistogram(void) const
    {
        return _prefillShapeHistogram;
    }
    [[nodiscard]] const std::map<std::string, uint64_t> &PrefillBatchShapeHistogram(void) const
    {
        return _prefillBatchShapeHistogram;
    }
    [[nodiscard]] uint64_t AclnnMatmulSynchronizations(void) const
    {
        return _aclnnMatmulSynchronizations;
    }
    [[nodiscard]] uint64_t LmHeadSynchronizations(void) const
    {
        return _lmHeadSynchronizations;
    }
    [[nodiscard]] uint64_t AclnnMatmulEventLeases(void) const
    {
        return _aclnnMatmulEventLeases;
    }
    [[nodiscard]] uint64_t AclnnMatmulEventRetirements(void) const
    {
        return _aclnnMatmulEventRetirements;
    }
    [[nodiscard]] uint64_t AclnnMatmulEventWaits(void) const
    {
        return _aclnnMatmulEventWaits;
    }
    [[nodiscard]] uint64_t AclnnMatmulPeakInflight(void) const
    {
        return _aclnnMatmulPeakInflight;
    }
    [[nodiscard]] uint64_t BatchedDecodeAttentionLaunches(void) const
    {
        return _batchedDecodeAttentionLaunches;
    }
    [[nodiscard]] uint64_t LegacyAttentionRequests(void) const
    {
        return _legacyAttentionRequests;
    }
    [[nodiscard]] uint64_t LegacyDecodeAttentionRequests(void) const
    {
        return _legacyDecodeAttentionRequests;
    }
    [[nodiscard]] uint64_t LegacyPrefillAttentionRequests(void) const
    {
        return _legacyPrefillAttentionRequests;
    }
    [[nodiscard]] uint64_t DecodeKvGatherBytes(void) const
    {
        return _decodeKvGatherBytes;
    }
    [[nodiscard]] uint64_t NativeAtbDecodeRequests(void) const
    {
        return _nativeAtbDecodeRequests;
    }
    [[nodiscard]] uint64_t NativeAtbDecodeLaunches(void) const
    {
        return _nativeAtbDecodeLaunches;
    }
    [[nodiscard]] uint64_t NativeAtbCacheWrites(void) const
    {
        return _nativeAtbCacheWrites;
    }
    [[nodiscard]] uint64_t NativeAtbStagingBytes(void) const
    {
        return _nativeAtbStagingBytes;
    }
    [[nodiscard]] uint64_t DirectAtbSetupCount(void) const { return _directAtbSetupCount; }
    [[nodiscard]] uint64_t DirectAtbSetupReuseCount(void) const
    {
        return _directAtbSetupReuseCount;
    }
    [[nodiscard]] uint64_t DirectAtbExecuteCount(void) const { return _directAtbExecuteCount; }
    [[nodiscard]] uint64_t DirectAtbDecodeRequests(void) const { return _directAtbDecodeRequests; }
    [[nodiscard]] uint64_t DirectAtbAttentionLaunches(void) const
    {
        return _directAtbAttentionLaunches;
    }
    [[nodiscard]] uint64_t DirectAtbReshapeLaunches(void) const
    {
        return _directAtbReshapeLaunches;
    }
    [[nodiscard]] uint64_t DirectAtbStagingBytes(void) const
    {
        return _directAtbStagingBytes;
    }
    [[nodiscard]] uint64_t DirectAtbFusedRopeStagingBytes(void) const
    {
        return _directAtbFusedRopeStagingBytes;
    }
    [[nodiscard]] uint64_t DirectAtbPlanReuses(void) const { return _directAtbPlanReuses; }
    [[nodiscard]] uint64_t DirectAtbPlanRebuilds(void) const { return _directAtbPlanRebuilds; }
    [[nodiscard]] uint64_t DirectAtbPagedPlanReuses(void) const
    {
        return _directAtbPagedPlanReuses;
    }
    [[nodiscard]] uint64_t DirectAtbPagedPlanRebuilds(void) const
    {
        return _directAtbPagedPlanRebuilds;
    }
    [[nodiscard]] uint64_t DirectAtbReshapePlanReuses(void) const
    {
        return _directAtbReshapePlanReuses;
    }
    [[nodiscard]] uint64_t DirectAtbReshapePlanRebuilds(void) const
    {
        return _directAtbReshapePlanRebuilds;
    }
    [[nodiscard]] uint64_t DirectAtbMixedDecodeRequests(void) const
    {
        return _directAtbMixedDecodeRequests;
    }
    [[nodiscard]] uint64_t DirectAtbMixedDecodeLaunches(void) const
    {
        return _directAtbMixedDecodeLaunches;
    }
    [[nodiscard]] uint64_t DirectAtbCompactLaunches(void) const
    {
        return _directAtbCompactLaunches;
    }
    [[nodiscard]] uint64_t DirectAtbCompactBytes(void) const
    {
        return _directAtbCompactBytes;
    }
    [[nodiscard]] uint64_t DirectAtbScatterLaunches(void) const
    {
        return _directAtbScatterLaunches;
    }
    [[nodiscard]] uint64_t DirectAtbScatterBytes(void) const
    {
        return _directAtbScatterBytes;
    }
    [[nodiscard]] uint64_t DirectAtbMetadataH2DBytes(void) const
    {
        return _directAtbMetadataH2DBytes;
    }
    [[nodiscard]] uint64_t DirectAtbMetadataH2DLaunches(void) const
    {
        return _directAtbMetadataH2DLaunches;
    }
#endif
    [[nodiscard]] uint64_t ForwardInputEvents(void) const { return _forwardInputEvents; }
    [[nodiscard]] uint64_t ForwardOutputEvents(void) const { return _forwardOutputEvents; }

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
#ifdef XLITE_ARCH_310P
    // Opaque state owned by the ASR-only M200 Cube MatMul backend.
    void *_m200MatmulState = nullptr;
    uint64_t m200MatmulRequests = 0;
    uint64_t m200MatmulKernelLaunches = 0;
    uint64_t aclnnMatmulRequests = 0;
    std::array<uint64_t, 21> m200MatmulRequestsByM{};
    std::array<uint64_t, 21> aclnnMatmulRequestsByM{};
#endif
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

    // cross-layer buffer
    XTensor _dsaTopkBuffer;  // int32, buffer for storing topk results, allocated via aclrtMalloc
    XTensor *dsaPerLayerTopk = nullptr;  // ptr to the **current** layer's topk result, either a
                                         // nullptr (no topk needed) or a ptr to _dsaTopkBuffer

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
    // Retained for the 310P ACLNN correctness backend. Version 0/1 metadata is
    // already available on the host, so the backend need not copy it back per
    // decoder layer. The normal device-side metadata remains authoritative for
    // every other architecture.
    std::vector<uint32_t> _lensHost;
    std::vector<uint32_t> _blockTablesHost;
#ifdef XLITE_ARCH_310P
    // Computed once in PrepareAttn and shared by all decoder layers. Direct
    // ATB uses these indices to compact only the decode rows of a mixed batch.
    std::vector<uint32_t> _queryOffsetsHost;
    std::vector<uint32_t> _decodeRequestIndicesHost;
    std::vector<uint8_t> _directAtbProcessedRequests;
    XTensor _decodeQueryOffsets;
    XTensor _decodeBlockTables;
    XTensor _decodeTotalLens;
    uint32_t _decodeTableColumns = 0;
    bool _decodeMetadataReady = false;
#endif
#ifdef XLITE_310P_LLM_FP16_POC
    // Stable page-locked sources for asynchronous attention metadata H2D.
    // The std::vectors above remain the host source of truth shared by all
    // decoder layers in one forward.
    XTensor _positionPinnedHost;
    XTensor _slotMappingPinnedHost;
    XTensor _cachedLensPinnedHost;
    XTensor _totalLensPinnedHost;
    XTensor _lensPinnedHost;
    XTensor _queryStartLocPinnedHost;
    XTensor _blockTablesPinnedHost;
    XTensor _decodeQueryOffsetsPinnedHost;
    XTensor _decodeBlockTablesPinnedHost;
    XTensor _decodeTotalLensPinnedHost;
#endif
    uint32_t _batch = 0;
    uint32_t _maxTotalLens;
    uint32_t _tileSizeOfCachedKV;
    XTensor
        _attnPosition;  // [batchedTokens] int64, ref: v0/1 -> _position, v2 -> attnMeta.position
    std::vector<XTensor> _attnBlockTables;  // [batch, maxNumBlocks] int32, per-kv-cache, ref: v0/1
                                            // -> {_blockTables}, v2 -> attnMeta.blockTables
    std::vector<XTensor> _attnSlotMapping;  // [batchedTokens] int32, per-kv-cache, ref: v0/1 ->
                                            // {_slotMapping}, v2 -> attnMeta.slotMapping
    XTensor _attnLens;        // [batch] int32, ref: v0/1 -> _lens, v2 -> attnMeta.lens
    XTensor _attnCachedLens;  // [batch] int32, ref: v0/1 -> _cachedLens, v2 -> attnMeta.cachedLens
    // Total valid KV length (cached + query). Native 310P paged attention consumes this
    // directly, avoiding a host callback or per-layer device add.
    XTensor _attnTotalLens;
    XTensor _attnQueryStartLoc;  // [batch] int32, ref: v0/1 -> _queryStartLoc, v2 ->
                                 // attnMeta.queryStartLoc
    XTensor _position;           // [maxBatchedTokens] int64, internal buffer (malloc+free in dtor)
    XTensor _blockTables;  // [maxBatch * DIV_ROUND_UP(maxSeqLen, blockSize)] int32, internal buffer
    XTensor _slotMapping;  // [maxBatchedTokens] int32, internal buffer
    XTensor _cachedLens;   // [maxBatch] int32, internal buffer
    XTensor _totalLens;    // [maxBatch] int32, internal buffer
    XTensor _lens;         // [maxBatch] int32, internal buffer
    XTensor _queryStartLoc;  // [maxBatch] int32, internal buffer
    // Host copy of per-request query lengths from PrepareAttn (mixed-length linear attn).
    std::vector<uint32_t> _hostLens;

    // for MoE
    XTensor _tokensPerEpGroupAllEpHost;

    // DP metadata
    uint64_t currTokens = 1;
    // maxTokensDp: initialized as max tokens across DP ranks; updated to the token number
    // to pad to for MoE when _dpSize > 1 and _nRoutedExperts > 0 and !enableMoEAllToAll
    uint64_t maxTokensDp = 1;

protected:
#ifdef XLITE_ARCH_310P
    XMatmulBackend310P _matmulBackend310P = XMatmulBackend310P::M200_ASR;
    XDecodeAttentionBackend310P _decodeAttentionBackend310P =
        XDecodeAttentionBackend310P::LEGACY;
    bool _directAtbSetupReuse310P = false;
    // The V2 batch path is retained for targeted shape diagnosis, but real
    // long/chunked ASR prefills have not passed transcript equivalence yet.
    bool _enableBatchedPrefillAttention310P = false;
    // ACLNN may retain workspace and tensor descriptors until queued work has
    // completed. Event leases preserve those lifetimes without synchronizing
    // the complete Runtime stream after every MatMul.
    bool _aclnnMatmulAsync310P = false;
    struct AclnnMatmulLease310P {
        aclrtEvent event = nullptr;
        std::vector<XTensor *> tensors;
        std::function<void()> cleanup;
    };
    std::deque<AclnnMatmulLease310P> _aclnnMatmulLeases310P;
#endif
    int GetNodeIps(void);
    int InitHcclComm(void);
    int InitXcclComm(void);
    void FiniXcclComm(void);
    uint32_t _devid;
    aclrtEvent _event = nullptr;
#ifdef XLITE_310P_LLM_FP16_POC
    // Input and output handoffs must not reset the same event while the other
    // direction is still pending.
    aclrtEvent _inputReadyEvent = nullptr;
    aclrtEvent _outputReadyEvent = nullptr;
    void *_causalMaskPinnedHost = nullptr;
#endif
    aclrtContext context = nullptr;
    bool _initOutside = false;
    bool _inited = false;
    bool _graphCommEnabled = true;
    bool _forceSyncAttention = false;
    uint64_t _streamSynchronizations = 0;
    uint64_t _forwardInputEvents = 0;
    uint64_t _forwardOutputEvents = 0;
    uint64_t _prepareAttnSynchronizations = 0;
    uint64_t _attentionMetadataD2HBytes = 0;
    uint64_t _attentionAclnnLaunches = 0;
    uint64_t _attentionForcedSynchronizations = 0;
    uint64_t _attentionWorkspaceReuses = 0;
#ifdef XLITE_ARCH_310P
    uint64_t _batchedDecodeAttentionRequests = 0;
    uint64_t _batchedDecodeAttentionLaunches = 0;
    uint64_t _batchedPrefillAttentionRequests = 0;
    uint64_t _batchedPrefillAttentionLaunches = 0;
    uint64_t _prefillShapeForwardCalls = 0;
    uint64_t _prefillShapeRequests = 0;
    uint64_t _prefillExactGroupableRequests = 0;
    uint64_t _prefillActualQueryTokens = 0;
    uint64_t _prefillPaddedQueryTokens = 0;
    std::map<std::string, uint64_t> _prefillShapeHistogram;
    std::map<std::string, uint64_t> _prefillBatchShapeHistogram;
    uint64_t _aclnnMatmulSynchronizations = 0;
    uint64_t _lmHeadSynchronizations = 0;
    uint64_t _aclnnMatmulEventLeases = 0;
    uint64_t _aclnnMatmulEventRetirements = 0;
    uint64_t _aclnnMatmulEventWaits = 0;
    uint64_t _aclnnMatmulPeakInflight = 0;
    uint64_t _legacyAttentionRequests = 0;
    uint64_t _legacyDecodeAttentionRequests = 0;
    uint64_t _legacyPrefillAttentionRequests = 0;
    uint64_t _decodeKvGatherBytes = 0;
    uint64_t _nativeAtbDecodeRequests = 0;
    uint64_t _nativeAtbDecodeLaunches = 0;
    uint64_t _nativeAtbCacheWrites = 0;
    uint64_t _nativeAtbStagingBytes = 0;
    uint64_t _directAtbSetupCount = 0;
    uint64_t _directAtbSetupReuseCount = 0;
    uint64_t _directAtbExecuteCount = 0;
    uint64_t _directAtbDecodeRequests = 0;
    uint64_t _directAtbAttentionLaunches = 0;
    uint64_t _directAtbReshapeLaunches = 0;
    uint64_t _directAtbStagingBytes = 0;
    uint64_t _directAtbFusedRopeStagingBytes = 0;
    uint64_t _directAtbPlanReuses = 0;
    uint64_t _directAtbPlanRebuilds = 0;
    uint64_t _directAtbPagedPlanReuses = 0;
    uint64_t _directAtbPagedPlanRebuilds = 0;
    uint64_t _directAtbReshapePlanReuses = 0;
    uint64_t _directAtbReshapePlanRebuilds = 0;
    uint64_t _directAtbMixedDecodeRequests = 0;
    uint64_t _directAtbMixedDecodeLaunches = 0;
    uint64_t _directAtbCompactLaunches = 0;
    uint64_t _directAtbCompactBytes = 0;
    uint64_t _directAtbScatterLaunches = 0;
    uint64_t _directAtbScatterBytes = 0;
    uint64_t _directAtbMetadataH2DLaunches = 0;
    uint64_t _directAtbMetadataH2DBytes = 0;
#endif
    void *_lastAttentionWorkspace = nullptr;
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
