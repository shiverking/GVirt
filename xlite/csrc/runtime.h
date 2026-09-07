/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_RUNTIME_H_
#define _XLITE_RUNTIME_H_

#include <cstdint>
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
    uint32_t _batch;
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
    XTensor _attnQueryStartLoc;  // [batch] int32, ref: v0/1 -> _queryStartLoc, v2 ->
                                 // attnMeta.queryStartLoc
    XTensor _position;           // [maxBatchedTokens] int64, internal buffer (malloc+free in dtor)
    XTensor _blockTables;  // [maxBatch * DIV_ROUND_UP(maxSeqLen, blockSize)] int32, internal buffer
    XTensor _slotMapping;  // [maxBatchedTokens] int32, internal buffer
    XTensor _cachedLens;   // [maxBatch] int32, internal buffer
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
    int GetNodeIps(void);
    int InitHcclComm(void);
    int InitXcclComm(void);
    void FiniXcclComm(void);
    uint32_t _devid;
    aclrtEvent _event = nullptr;
    aclrtContext context = nullptr;
    bool _initOutside = false;
    bool _inited = false;
    bool _graphCommEnabled = true;
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
