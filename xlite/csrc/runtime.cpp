/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <cmath>
#include <cstring>
#include <sstream>
#include "ascend.h"
#include "base.h"
#include "runtime.h"
#include "op.h"
#include "sock.h"
#include "ccl.h"
#include "auto_tuner.h"
#ifdef XLITE_310P_LLM_FP16_POC
#include "aclnn_310p.h"
#endif

#define XLITE_DEFAULT_IP "127.0.0.1"
#define XLITE_DP_PORT_OFFSET 200
#define XLITE_EP_PORT_OFFSET 300
#define XLITE_CCL_PORT_OFFSET 400

XRuntime::XRuntime(uint32_t devid, size_t sizeMB, uint32_t rankId, uint32_t tpSize, uint32_t dpSize,
                   uint32_t moeTpSize, uint32_t moeEpSize)
    : _devid(devid), _rankId(rankId), _tpSize(tpSize), _dpSize(dpSize), _moeTpSize(moeTpSize),
      _moeEpSize(moeEpSize)
{
#ifdef XLITE_310P_LLM_FP16_POC
    if (rankId != 0 || tpSize != 1 || dpSize != 1 || moeTpSize != 1 || moeEpSize != 1) {
        throw std::runtime_error(
            "Ascend310P llm_fp16 POC only supports rank=0, TP=1, DP=1 and no MoE parallelism");
    }
    defaultMatmulSwizzle = 0;
    disableSwizzleTable = true;
    _forceSyncAclnn = isEnvironmentVariableTrue(std::getenv("XLITE_310P_FORCE_SYNC_ACLNN"));
    // Keep the validated fallback unless async is explicitly selected. The
    // mask upload ordering fix must pass an end-to-end hardware acceptance
    // before changing this default; the failure did not establish a CANN bug.
    _forceSyncMatmul =
        isEnvironmentVariableTrue(std::getenv("XLITE_310P_FORCE_SYNC_MATMUL")) ||
        !isEnvironmentVariableTrue(std::getenv("XLITE_310P_ASYNC_MATMUL"));
    _forceSyncAttention =
        isEnvironmentVariableTrue(std::getenv("XLITE_310P_FORCE_SYNC_ATTENTION"));
    _stressWorkspaceReuse =
        isEnvironmentVariableTrue(std::getenv("XLITE_310P_STRESS_WORKSPACE_REUSE"));
    _syncForwardBoundary =
        isEnvironmentVariableTrue(std::getenv("XLITE_310P_FORCE_SYNC_FORWARD"));
#endif
    if (sizeMB != 0) {
        Init(sizeMB);
    }
}

void XRuntime::Init(size_t sizeMB)
{
    if (_inited) {
        return;
    }
    aclError initRet = aclInit(nullptr);
    uint32_t count;
    if (initRet == ACL_ERROR_REPEAT_INITIALIZE) {
        _initOutside = true;
    } else {
        CHECK_ACL(initRet);
    }
    CHECK_ACL(aclrtSetDevice(_devid));
    CHECK_ACL(aclrtCreateStream(&stream));
    CHECK_ACL(aclrtGetDeviceCount(&count));
    _nDevPerNode = count;

    if (sizeMB != 0) {
        _pool = new XTensorPool(sizeMB << MB_BIT, _rankId);
        if (_pool->Init()) {
            throw std::runtime_error("XRuntime: tensor pool initialization failed");
        }
#ifdef XLITE_310P_LLM_FP16_POC
        XTensor &reserve = GetTensor({XlitePaged310P::ReserveBytes}, INT8, DBG_LOC);
        pagedScratch = reserve.ptr;
        pagedMetadata = static_cast<uint8_t *>(reserve.ptr) + XlitePaged310P::ScratchBytes;
#endif
    }

    _rankSize = _tpSize * _dpSize;
    if (InitHcclComm()) {
        delete _pool;
        throw std::runtime_error("XRuntime: HCCL initialization failed");
    }

    if (sizeMB != 0) {
        if (InitXcclComm()) {
            delete _pool;
            throw std::runtime_error("XRuntime: XCCL initialization failed");
        }
    }

    int64_t val;
    CHECK_ACL(aclGetDeviceCapability(_devid, ACL_DEVICE_INFO_AI_CORE_NUM, &val));
    aicNum = static_cast<uint32_t>(val);
    CHECK_ACL(aclGetDeviceCapability(_devid, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &val));
    aivNum = static_cast<uint32_t>(val);
    reportedAivNum = aivNum;
#ifdef XLITE_310P_LLM_FP16_POC
    // Ascend310P/M200 uses unified AI cores.  The official 310P Add kernel
    // sample launches all eight AI cores successfully, while some runtime
    // stacks report seven through ACL_DEVICE_INFO_VECTOR_CORE_NUM.  Keep that
    // raw value for diagnostics, but use the unified AI-core count for launch.
    if (aicNum == 0) {
        throw std::runtime_error("Ascend 310P reported zero AI cores");
    }
    aivNum = aicNum;
#endif
    originAicNum = aicNum;
    originAivNum = aivNum;

#ifdef XLITE_310P_LLM_FP16_POC
    CHECK_ACL(aclrtCreateEventWithFlag(&_inputReadyEvent, ACL_EVENT_SYNC));
    CHECK_ACL(aclrtCreateEventWithFlag(&_outputReadyEvent, ACL_EVENT_SYNC));
    CHECK_ACL(aclrtCreateEventWithFlag(&_metadataUploadedEvent, ACL_EVENT_SYNC));
    CHECK_ACL(aclrtMallocHost(&_pagedPinnedHost, XlitePaged310P::MetadataBytes));
#else
    CHECK_ACL(aclrtCreateEvent(&_event));
#endif
    CHECK_ACL(aclrtCreateNotify(&notify, 0));
    CHECK_ACL(aclrtGetCurrentContext(&context));

    const char *envCommOptimizeLen = std::getenv("XLITE_COMM_OPTIMIZE_LEN");
    if (envCommOptimizeLen) {
        char *endPtr = nullptr;
        long val = strtol(envCommOptimizeLen, &endPtr, 10);
        if (endPtr != envCommOptimizeLen && *endPtr == '\0' && val >= 0) {
            commOptimizeLen = static_cast<uint32_t>(val);
        }
    }

    const char *envMoEAllToAll = std::getenv("XLITE_MOE_ALLTOALL");
    if (isEnvironmentVariableTrue(envMoEAllToAll)) {
        enableMoEAllToAll = true;
        if (_rankId == 0) {
            std::cout << "Xlite MoE AllToAll Enabled!" << std::endl;
        }
    }

    const char *ratioPerEPEnv = std::getenv("XLITE_ACTIVE_TOKENS_RATIO_PER_EP");
    if (ratioPerEPEnv) {
        char *endPtr = nullptr;
        double val = strtod(ratioPerEPEnv, &endPtr);
        double min = 1 / static_cast<double>(_moeEpSize);
        double max = 1.0f;
        if (endPtr != ratioPerEPEnv && *endPtr == '\0' && std::isfinite(val)) {
            activeTokensRatioPerEp = val;
        }
        if (activeTokensRatioPerEp < min) {
            activeTokensRatioPerEp = min;
        }
        if (activeTokensRatioPerEp > max) {
            activeTokensRatioPerEp = max;
        }
    }

    if (isEnvironmentVariableFalse(std::getenv("XLITE_ENABLE_GRAPH_COMM"))) {
        _graphCommEnabled = false;
    }

    _inited = true;
}

XRuntime::~XRuntime(void)
{
    // All allocations below may still be referenced by work queued on the
    // runtime stream.  Drain it before destroying or freeing any resource.
    if (stream) {
        (void)aclrtSynchronizeStream(stream);
    }
    FiniXcclComm();

    if (_tpSize > 1 && _tpComm) {
        HcclCommDestroy(_tpComm);
    }
    if (_dpSize > 1 && _dpComm) {
        HcclCommDestroy(_dpComm);
    }
    if (_moeEpSize > 1 && _epComm) {
        HcclCommDestroy(_epComm);
    }

    for (auto &e : _agGraphs) {
        if (e.modelRI) {
            (void)aclmdlRIDestroy(e.modelRI);
            e.modelRI = nullptr;
        }
    }
    _agGraphs.clear();
    for (auto &e : _rsGraphs) {
        if (e.modelRI) {
            (void)aclmdlRIDestroy(e.modelRI);
            e.modelRI = nullptr;
        }
    }
    _rsGraphs.clear();
    if (_attnInitialized) {
        (void)aclrtFree(_position.ptr);
        (void)aclrtFree(_slotMapping.ptr);
        (void)aclrtFree(_cachedLens.ptr);
        (void)aclrtFree(_lens.ptr);
        (void)aclrtFree(_queryStartLoc.ptr);
        (void)aclrtFree(_blockTables.ptr);
        (void)aclrtFreeHost(_tokensPerEpGroupAllEpHost.ptr);
        (void)aclrtFree(_agSendBuf.ptr);
        (void)aclrtFree(_agRecvBuf.ptr);
        (void)aclrtFree(_dsaTopkBuffer.ptr);
#ifdef XLITE_310P_LLM_FP16_POC
        (void)aclrtFreeHost(_positionPinnedHost.ptr);
        (void)aclrtFreeHost(_slotMappingPinnedHost.ptr);
        (void)aclrtFreeHost(_cachedLensPinnedHost.ptr);
        (void)aclrtFreeHost(_lensPinnedHost.ptr);
        (void)aclrtFreeHost(_queryStartLocPinnedHost.ptr);
        (void)aclrtFreeHost(_blockTablesPinnedHost.ptr);
#endif
    }
    delete _pool;
    if (_event) {
        (void)aclrtDestroyEvent(_event);
    }
#ifdef XLITE_310P_LLM_FP16_POC
    if (_inputReadyEvent) {
        (void)aclrtDestroyEvent(_inputReadyEvent);
    }
    if (_outputReadyEvent) {
        (void)aclrtDestroyEvent(_outputReadyEvent);
    }
    if (_metadataUploadedEvent) {
        (void)aclrtDestroyEvent(_metadataUploadedEvent);
    }
    if (_pagedPinnedHost) {
        (void)aclrtFreeHost(_pagedPinnedHost);
    }
    if (_causalMaskPinnedHost) {
        (void)aclrtFreeHost(_causalMaskPinnedHost);
    }
#endif
    if (notify) {
        (void)aclrtDestroyNotify(notify);
    }
    if (stream) {
        (void)aclrtDestroyStream(stream);
    }
    (void)aclrtResetDevice(static_cast<int32_t>(_devid));

    if (!_initOutside) {
        (void)aclFinalize();
    }
}

int XRuntime::GetNodeIps(void)
{
    const char *envDevs = std::getenv("XLITE_DEVS_PER_NODE");
    const char *envIps = std::getenv("XLITE_NODE_IPS");
    const char *envPort = std::getenv("XLITE_PORT");

    if (envDevs) {
        char *endPtr = nullptr;
        long val = strtol(envDevs, &endPtr, 10);
        if (endPtr != envDevs && *endPtr == '\0' && val >= 0) {
            _nDevPerNode = static_cast<uint32_t>(val);
        }
    }

    if (envPort) {
        char *endPtr = nullptr;
        long val = strtol(envPort, &endPtr, 10);
        if (endPtr != envPort && *endPtr == '\0' && val >= 0) {
            _port = static_cast<uint32_t>(val);
        }
    }

    if (_rankSize <= _nDevPerNode) {
        _ips.push_back(std::string(XLITE_DEFAULT_IP));
        return 0;
    }

    if (!envIps) {
        throw std::runtime_error(std::string(__func__) +
                                 ": please set XLITE_NODE_IPS in multi-node environment.");
    }

    std::string ipsStr(envIps);
    std::istringstream iss(ipsStr);
    std::string ip;
    while (std::getline(iss, ip, ',')) {
        _ips.push_back(ip);
    }

    if (_ips.size() != DIV_ROUND_UP(_rankSize, _nDevPerNode)) {
        throw std::runtime_error(std::string(__func__) + ": XLITE_NODE_IPS not match " +
                                 std::to_string(_rankSize) + " / " + std::to_string(_nDevPerNode));
    }
    return 0;
}

void XRuntime::FiniXcclComm(void)
{
    delete _tpXcclComm;
    delete _dpXcclComm;
}

int XRuntime::InitXcclComm(void)
{
    std::string ip;
    uint32_t port;
    const char *envDisableXccl = std::getenv("XLITE_DISABLE_XCCL");
    const char *envDeterministic = std::getenv("HCCL_DETERMINISTIC");
    void *myXTensorPtr = _pool->Ptr();
    size_t myXTensorSize = _pool->Size();
    char ipcXTensorKey[EXPORT_KEY_LEN];

    if (_rankSize == 1 || _rankSize > XLITE_CCL_MAX_RANK_SIZE) {
        return 0;
    }

    if (isEnvironmentVariableTrue(envDisableXccl) || isEnvironmentVariableTrue(envDeterministic)) {
        return 0;
    }

    bool enableTpXccl = (_tpSize > 1 && _tpSize <= _nDevPerNode);
    bool enableDpXccl = (_dpSize > 1 && _rankSize <= _nDevPerNode);

    if (!enableTpXccl && !enableDpXccl) {
        return 0;
    }

    CHECK_ACL(aclrtIpcMemGetExportKey(myXTensorPtr, myXTensorSize, ipcXTensorKey, EXPORT_KEY_LEN,
                                      ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION));

    static uint32_t portOffset = 0;
    if (enableTpXccl) {
        ip = _ips[ROUND_DOWN(_rankId, _tpSize) / _nDevPerNode];
        port = _port + XLITE_CCL_PORT_OFFSET + _rankId / _tpSize + portOffset;
        _tpXcclComm = new XcclComm(_rankId % _tpSize, _tpSize);
        if (_tpXcclComm->Init(ip, port, myXTensorPtr, ipcXTensorKey)) {
            return -EFAULT;
        }
    }

    if (enableDpXccl) {
        ip = _ips[_rankId % _tpSize / _nDevPerNode];
        port =
            _port + XLITE_CCL_PORT_OFFSET + XLITE_DP_PORT_OFFSET + _rankId % _tpSize + portOffset;
        _dpXcclComm = new XcclComm(_rankId / _tpSize, _dpSize);
        if (_dpXcclComm->Init(ip, port, myXTensorPtr, ipcXTensorKey)) {
            return -EFAULT;
        }
    }
    portOffset += 500;

    return 0;
}

int XRuntime::InitHcclComm(void)
{
    std::string ip;
    uint32_t port;
    HcclRootInfo rootInfo;

    int ret = GetNodeIps();
    if (ret) {
        return ret;
    }

    static uint32_t portOffset = 0;
    if (_tpSize > 1) {
        ip = _ips[ROUND_DOWN(_rankId, _tpSize) / _nDevPerNode];
        port = _port + _rankId / _tpSize + portOffset;

        if (_rankId % _tpSize == 0) {
            CHECK_HCCL(HcclGetRootInfo(&rootInfo));
        }
        XSock *sock = new XSock(_rankId % _tpSize, _tpSize, ip, port);
        sock->Broadcast(&rootInfo, sizeof(rootInfo));
        delete sock;
        CHECK_HCCL(HcclCommInitRootInfo(_tpSize, &rootInfo, _rankId % _tpSize, &_tpComm));
    }

    if (_dpSize > 1) {
        ip = _ips[_rankId % _tpSize / _nDevPerNode];
        port = _port + XLITE_DP_PORT_OFFSET + _rankId % _tpSize + portOffset;

        if (_rankId / _tpSize == 0) {
            CHECK_HCCL(HcclGetRootInfo(&rootInfo));
        }
        XSock *sock = new XSock(_rankId / _tpSize, _dpSize, ip, port);
        sock->Broadcast(&rootInfo, sizeof(rootInfo));
        delete sock;
        CHECK_HCCL(HcclCommInitRootInfo(_dpSize, &rootInfo, _rankId / _tpSize, &_dpComm));
    }

    if (_moeEpSize > 1) {
        ip = _ips[_rankId % _moeTpSize];
        port = _port + XLITE_EP_PORT_OFFSET + _rankId % _moeTpSize + portOffset;

        if (_rankId / _moeTpSize == 0) {
            CHECK_HCCL(HcclGetRootInfo(&rootInfo));
        }
        XSock *sock = new XSock(_rankId / _moeTpSize, _moeEpSize, ip, port);
        sock->Broadcast(&rootInfo, sizeof(rootInfo));
        delete sock;
        CHECK_HCCL(HcclCommInitRootInfo(_moeEpSize, &rootInfo, _rankId / _moeTpSize, &_epComm));
    }
    portOffset += 500;

    return 0;
}

void XRuntime::InitAttn(uint64_t maxBatchedTokens, uint64_t maxBatch, uint64_t maxSeqLen,
                        uint32_t blockSize, uint32_t indexTopK)
{
    std::vector<uint32_t> vgatherIndices;
    size_t size;
    void *ptr;

    size = maxBatchedTokens * XDtypeBit(INT64) / 8;
    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _position.Init({maxBatchedTokens}, INT64, ptr);

    size = maxBatchedTokens * XDtypeBit(INT32) / 8;
    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _slotMapping.Init({maxBatchedTokens}, INT32, ptr);

    size = maxBatch * XDtypeBit(INT32) / 8;
    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _cachedLens.Init({maxBatch}, INT32, ptr);

    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _lens.Init({maxBatch}, INT32, ptr);

    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _queryStartLoc.Init({maxBatch}, INT32, ptr);

    size = maxBatch * DIV_ROUND_UP(maxSeqLen, blockSize) * XDtypeBit(INT32) / 8;
    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _blockTables.Init({maxBatch * DIV_ROUND_UP(maxSeqLen, blockSize)}, INT32, ptr);

#ifdef XLITE_310P_LLM_FP16_POC
    auto allocPinned = [](XTensor &tensor, std::vector<size_t> shape, enum XDtype dtype) {
        size_t numel = 1;
        for (size_t dim : shape) {
            numel *= dim;
        }
        void *hostPtr = nullptr;
        CHECK_ACL(aclrtMallocHost(&hostPtr, numel * XDtypeBit(dtype) / 8));
        tensor.Init(std::move(shape), dtype, hostPtr);
    };
    allocPinned(_positionPinnedHost, {maxBatchedTokens}, INT64);
    allocPinned(_slotMappingPinnedHost, {maxBatchedTokens}, INT32);
    allocPinned(_cachedLensPinnedHost, {maxBatch}, INT32);
    allocPinned(_lensPinnedHost, {maxBatch}, INT32);
    allocPinned(_queryStartLocPinnedHost, {maxBatch}, INT32);
    allocPinned(_blockTablesPinnedHost,
                {maxBatch * DIV_ROUND_UP(maxSeqLen, blockSize)}, INT32);
#endif

    size = _moeEpSize * _moeEpSize * XDtypeBit(INT32) / 8;
    CHECK_ACL(aclrtMallocHost(&ptr, size));
    _tokensPerEpGroupAllEpHost.Init({_moeEpSize * _moeEpSize}, INT32, ptr);

    if (indexTopK > 0) {
        size = maxBatchedTokens * indexTopK * XDtypeBit(INT32) / 8;
        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        _dsaTopkBuffer.Init({maxBatchedTokens, indexTopK}, INT32, ptr);
    }
}

void XRuntime::PrepareCommBuffers(uint64_t maxBatch, uint32_t hiddenSize, uint32_t nRoutedExperts,
                                  uint32_t defDpSize, int inputDtype, int weightsDtype)
{
    enum XDtype inDtype = static_cast<enum XDtype>(inputDtype);
    enum XDtype wDtype = static_cast<enum XDtype>(weightsDtype);
    uint64_t perTokenInput = hiddenSize * XDtypeBit(inDtype) / 8;
    uint64_t perTokenWeights = static_cast<uint64_t>(nRoutedExperts) * XDtypeBit(wDtype) / 8;
    uint64_t perTokenRouting = static_cast<uint64_t>(nRoutedExperts) / 8;  // BIT1: 1 bit/expert
    _agPerTokenBytes =
        perTokenInput + perTokenWeights + perTokenRouting;  // needed by Select fallback
    // AG buffers serve in-graph path (m <= maxBatch); eager uses pool tensors.
    uint64_t agSendBytes = maxBatch * _agPerTokenBytes;
    uint64_t agRecvBytes = agSendBytes * defDpSize;

    // RS aliases AG buffers; AG and RS per-token costs differ (INT8-packed vs
    // inputDtype hidden), so size each to the max to keep the alias from overflowing.
    enum XDtype rsDtype = inDtype;  // RS dtype == inputDtype (embed.dtype)
    _rsPerTokenBytes = static_cast<uint64_t>(hiddenSize) * XDtypeBit(rsDtype) / 8;
    _rsHiddenSize = hiddenSize;
    uint64_t rsSendBytes = maxBatch * defDpSize * _rsPerTokenBytes;
    uint64_t rsRecvBytes = maxBatch * _rsPerTokenBytes;

    uint64_t sendBytes = std::max(agSendBytes, rsSendBytes);
    uint64_t recvBytes = std::max(agRecvBytes, rsRecvBytes);
    void *sendPtr = nullptr;
    void *recvPtr = nullptr;
    CHECK_ACL(aclrtMalloc(&sendPtr, sendBytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMalloc(&recvPtr, recvBytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    _agSendBuf.Init({sendBytes}, INT8, sendPtr);
    _agRecvBuf.Init({recvBytes}, INT8, recvPtr);
    // RS alias wrappers: each views only its own logical slice of the oversized buffer.
    _rsSendBuf.Init({rsSendBytes}, INT8, _agSendBuf.ptr);  // alias, no extra malloc
    _rsRecvBuf.Init({rsRecvBytes}, INT8, _agRecvBuf.ptr);  // alias, no extra malloc
}

void XRuntime::PrepareAttn(XModelAttnMeta &attnMeta, uint64_t maxBatchedTokens, uint64_t maxBatch,
                           uint64_t maxSeqLen, uint32_t nHeads, uint32_t nKVHeads,
                           uint32_t blockSize, uint32_t hiddenSize, uint32_t nRoutedExperts,
                           uint32_t defDpSize, int inputDtype, int weightsDtype, uint32_t indexTopK)
{
    if (!_attnInitialized) {
        InitAttn(maxBatchedTokens, maxBatch, maxSeqLen, blockSize, indexTopK);
        bool agInGraph = AllGatherInGraphActive(DP);
        bool rsInGraph = ReduceScatterInGraphActive(DP);
        if (defDpSize > 1 && (agInGraph || rsInGraph)) {
            PrepareCommBuffers(maxBatch, hiddenSize, nRoutedExperts, defDpSize, inputDtype,
                               weightsDtype);
            if (agInGraph && maxBatch > 0 && _agPerTokenBytes != 0) {
                int hcclDtypeInt = static_cast<int>(XDtype2HcclDtype(INT8));
                _agGraphs.assign(maxBatch + 1, GraphCaptureEntry{});  // index 0 unused, 1..maxBatch
                for (uint32_t m = 1; m <= static_cast<uint32_t>(maxBatch); m++) {
                    CaptureHcclAllGather(_agSendBuf.ptr, _agRecvBuf.ptr, m, hcclDtypeInt, DP);
                }
            }
            if (rsInGraph && maxBatch > 0 && hiddenSize > 0) {
                int rsHcclDtype =
                    static_cast<int>(XDtype2HcclDtype(static_cast<enum XDtype>(inputDtype)));
                _rsGraphs.assign(maxBatch + 1, GraphCaptureEntry{});  // index 0 unused, 1..maxBatch
                for (uint32_t m = 1; m <= static_cast<uint32_t>(maxBatch); m++) {
                    CaptureHcclReduceScatter(_rsSendBuf.ptr, _rsRecvBuf.ptr, m, rsHcclDtype, DP);
                }
            }
        }
        _attnInitialized = true;
    }
#ifdef XLITE_310P_LLM_FP16_POC
    WaitMetadataUpload();
#endif
    // Reset cross-layer topk state per step, first full layer repopulates it.
    if (indexTopK > 0) {
        _dsaTopkValid = false;
    }
    uint32_t batch = attnMeta.lens.size();
    std::vector<uint32_t> lens(batch);
    std::vector<uint32_t> cachedLens(batch);
    std::vector<uint32_t> queryStartLoc(batch);
    std::vector<uint32_t> numBlocks(batch);
    std::vector<uint32_t> slotMapping, blockTables;
    std::vector<uint64_t> position;
    uint32_t queryStart, blockId, id, k;
    size_t size;

    if (batch == 0 || batch > maxBatch || attnMeta.cachedLens.size() != batch) {
        throw std::runtime_error(std::string(__func__) + ":" + std::to_string(__LINE__) +
                                 ": invalid batchSize: " + std::to_string(batch));
    }

    batchedTokens = 0;
    _maxNumBlocks = 0;
    _batch = batch;
    queryStart = 0;
    bool allCached = true;
    bool anyMultiToken = false;
    for (uint32_t i = 0; i < batch; i++) {
        lens[i] = attnMeta.lens[i];
        cachedLens[i] = attnMeta.cachedLens[i];
        queryStartLoc[i] = queryStart;
        queryStart += lens[i];
        numBlocks[i] = DIV_ROUND_UP(lens[i] + cachedLens[i], blockSize);
        _maxNumBlocks = numBlocks[i] > _maxNumBlocks ? numBlocks[i] : _maxNumBlocks;
        batchedTokens += lens[i];
        if (cachedLens[i] == 0) {
            allCached = false;
        }
        if (lens[i] != 1) {
            anyMultiToken = true;
        }
    }
    _decodeStep = !anyMultiToken;
    // Decode only when every request has cache and this step is a single token.
    _linearDecodeStep = allCached && !anyMultiToken && batch > 0;
    _lensHost = lens;
    _cachedLensHost = cachedLens;

    if (batchedTokens == 0 || batchedTokens > maxBatchedTokens) {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                 ": invalid attnMeta batched tokens(" +
                                 std::to_string(batchedTokens) + ") > maxBatchedTokens(" +
                                 std::to_string(maxBatchedTokens) + ")");
    }

    if (IsDummyRuntime() || _maxNumBlocks * blockSize <= MAX_KV_TILE_SIZE) {
        _tileSizeOfCachedKV = MAX_KV_TILE_SIZE;
    } else {
        uint32_t localHeads = std::max(nHeads / _tpSize, static_cast<uint32_t>(1));
        uint32_t localKvHeads = std::max(nKVHeads / _tpSize, static_cast<uint32_t>(1));
        _tileSizeOfCachedKV = GetTileSizeOfCachedKV(cachedLens, lens, localHeads / localKvHeads,
                                                    localKvHeads, blockSize, aicNum);
    }

    size = batch * XDtypeBit(INT32) / 8;
#ifdef XLITE_310P_LLM_FP16_POC
    std::memcpy(_lensPinnedHost.ptr, lens.data(), size);
    std::memcpy(_cachedLensPinnedHost.ptr, cachedLens.data(), size);
    std::memcpy(_queryStartLocPinnedHost.ptr, queryStartLoc.data(), size);
    void *lensSrc = _lensPinnedHost.ptr;
    void *cachedLensSrc = _cachedLensPinnedHost.ptr;
    void *queryStartLocSrc = _queryStartLocPinnedHost.ptr;
#else
    void *lensSrc = lens.data();
    void *cachedLensSrc = cachedLens.data();
    void *queryStartLocSrc = queryStartLoc.data();
#endif
    CHECK_ACL(aclrtMemcpyAsync(_lens.ptr, size, lensSrc, size, ACL_MEMCPY_HOST_TO_DEVICE, stream));
    CHECK_ACL(aclrtMemcpyAsync(_cachedLens.ptr, size, cachedLensSrc, size,
                               ACL_MEMCPY_HOST_TO_DEVICE, stream));
    CHECK_ACL(aclrtMemcpyAsync(_queryStartLoc.ptr, size, queryStartLocSrc, size,
                               ACL_MEMCPY_HOST_TO_DEVICE, stream));

    position.resize(batchedTokens);
    slotMapping.resize(batchedTokens);
    k = 0;
    if (attnMeta.blockTables.size() < batch) {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                 ": invalid blocktable(" +
                                 std::to_string(attnMeta.blockTables.size()) + ")");
    }
    for (uint32_t i = 0; i < batch; i++) {
        if (attnMeta.blockTables[i].size() < numBlocks[i]) {
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                     ": block table too small (" +
                                     std::to_string(attnMeta.blockTables[i].size()) + " < " +
                                     std::to_string(numBlocks[i]) + ")");
        }
        for (uint32_t j = 0; j < lens[i]; j++) {
            position[k] = cachedLens[i] + j;
            blockId = position[k] / blockSize;
            id = position[k] % blockSize;
            slotMapping[k++] = attnMeta.blockTables[i][blockId] * blockSize + id;
        }
    }
    size = batchedTokens * XDtypeBit(INT32) / 8;
#ifdef XLITE_310P_LLM_FP16_POC
    std::memcpy(_slotMappingPinnedHost.ptr, slotMapping.data(), size);
    void *slotMappingSrc = _slotMappingPinnedHost.ptr;
#else
    void *slotMappingSrc = slotMapping.data();
#endif
    CHECK_ACL(aclrtMemcpyAsync(_slotMapping.ptr, size, slotMappingSrc, size,
                               ACL_MEMCPY_HOST_TO_DEVICE, stream));
    _attnSlotMapping = _slotMapping;

    blockTables.resize(batch * _maxNumBlocks);
    for (uint32_t i = 0; i < batch; i++) {
        for (uint32_t j = 0; j < numBlocks[i]; j++) {
            blockTables[i * _maxNumBlocks + j] = attnMeta.blockTables[i][j];
        }
    }
    _blockTablesHost = blockTables;
    size = batch * _maxNumBlocks * XDtypeBit(INT32) / 8;
#ifdef XLITE_310P_LLM_FP16_POC
    std::memcpy(_blockTablesPinnedHost.ptr, blockTables.data(), size);
    void *blockTablesSrc = _blockTablesPinnedHost.ptr;
#else
    void *blockTablesSrc = blockTables.data();
#endif
    CHECK_ACL(aclrtMemcpyAsync(_blockTables.ptr, size, blockTablesSrc, size,
                               ACL_MEMCPY_HOST_TO_DEVICE, stream));
    _attnBlockTables = _blockTables;
    switch (attnMeta.version) {
        case 0: {
            size = batchedTokens * XDtypeBit(INT64) / 8;
#ifdef XLITE_310P_LLM_FP16_POC
            std::memcpy(_positionPinnedHost.ptr, position.data(), size);
            void *positionSrc = _positionPinnedHost.ptr;
#else
            void *positionSrc = position.data();
#endif
            CHECK_ACL(aclrtMemcpyAsync(_position.ptr, size, positionSrc, size,
                                       ACL_MEMCPY_HOST_TO_DEVICE, stream));
            _attnPosition = _position;
            break;
        }
        case 1:
            _attnPosition = attnMeta.vllmPosition;
            break;
        default:
            throw std::runtime_error(
                std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                ": invalid attnMeta version: " + std::to_string(attnMeta.version));
    }
#ifdef XLITE_310P_LLM_FP16_POC
    PreparePagedDecodeMetadata();
#endif
}

void XRuntime::SetDecodeAttentionBackend(const std::string &backend)
{
    if (backend != "legacy" && backend != "paged_310p") {
        throw std::invalid_argument("decode_attention_backend must be legacy or paged_310p");
    }
#ifndef XLITE_310P_LLM_FP16_POC
    if (backend != "legacy") {
        throw std::runtime_error("this build has no paged_310p backend");
    }
#endif
    if (!_lensHost.empty() && backend != decodeAttentionBackend) {
        throw std::runtime_error("select decode attention backend before preparing metadata");
    }
    decodeAttentionBackend = backend;
}

#ifdef XLITE_310P_LLM_FP16_POC
void XRuntime::WaitMetadataUpload()
{
    if (_metadataUploadPending && !IsDummyRuntime()) {
        CHECK_ACL(aclrtSynchronizeEvent(_metadataUploadedEvent));
        CHECK_ACL(aclrtResetEvent(_metadataUploadedEvent, stream));
        _metadataUploadPending = false;
    }
}

void XRuntime::PreparePagedDecodeMetadata()
{
    using namespace XlitePaged310P;
    if (IsDummyRuntime()) {
        return;
    }
    WaitMetadataUpload();
    pagedCount = 0;
    if (decodeAttentionBackend == "paged_310p") {
        if (_lensHost.size() > MaxBatch || _cachedLensHost.size() != _lensHost.size() ||
            _maxNumBlocks > MaxBlocks || _maxNumBlocks == 0 ||
            _blockTablesHost.size() != _lensHost.size() * _maxNumBlocks) {
            throw std::runtime_error("paged_310p invalid host metadata dimensions");
        }
        if (pagedScratch == nullptr) {
            throw std::runtime_error("paged_310p runtime pool has not been initialized");
        }
        auto *meta = static_cast<uint32_t *>(_pagedPinnedHost);
        std::memset(meta, 0, MetadataBytes);
        uint32_t row = 0, maxLength = 1;
        for (uint32_t request = 0; request < _lensHost.size(); ++request) {
            const uint64_t length = uint64_t(_lensHost[request]) + _cachedLensHost[request];
            if (_lensHost[request] == 0 || length > 2048 ||
                (length + 127) / 128 > _maxNumBlocks) {
                throw std::runtime_error("paged_310p invalid KV length/block table");
            }
            std::memcpy(meta + TableOffset + request * MaxBlocks,
                        _blockTablesHost.data() + request * _maxNumBlocks,
                        _maxNumBlocks * sizeof(uint32_t));
            if (_lensHost[request] == 1) {
                meta[pagedCount * RecordWords] = request;
                meta[pagedCount * RecordWords + 1] = row;
                meta[pagedCount * RecordWords + 2] = static_cast<uint32_t>(length);
                maxLength = std::max(maxLength, static_cast<uint32_t>(length));
                ++pagedCount;
            }
            row += _lensHost[request];
        }
        pagedPartitions = maxLength > 512 ? 4 : 1;
        pagedPartitionLength = ((maxLength + pagedPartitions * 128 - 1) /
                                 (pagedPartitions * 128)) * 128;
        CHECK_ACL(aclrtMemcpyAsync(pagedMetadata, MetadataBytes, meta, MetadataBytes,
                                   ACL_MEMCPY_HOST_TO_DEVICE, stream));
    }
    // Also protects the ordinary lens/slot/position staging buffers.
    CHECK_ACL(aclrtRecordEvent(_metadataUploadedEvent, stream));
    _metadataUploadPending = true;
}
#endif

void XRuntime::Synchronize(void)
{
    _streamSynchronizations++;
    CHECK_ACL(aclrtSynchronizeStream(stream));
}

void XRuntime::EventWaitCurrStream(aclrtStream currStream)
{
#ifdef XLITE_310P_LLM_FP16_POC
    CHECK_ACL(aclrtRecordEvent(_inputReadyEvent, currStream));
    CHECK_ACL(aclrtStreamWaitEvent(stream, _inputReadyEvent));
    CHECK_ACL(aclrtResetEvent(_inputReadyEvent, stream));
#else
    CHECK_ACL(aclrtRecordEvent(_event, currStream));
    CHECK_ACL(aclrtStreamWaitEvent(stream, _event));
    CHECK_ACL(aclrtResetEvent(_event, stream));
#endif
}

void XRuntime::EventRecordCurrStream(aclrtStream currStream)
{
#ifdef XLITE_310P_LLM_FP16_POC
    if (_syncForwardBoundary) {
        _forwardBoundarySynchronizations++;
        Synchronize();
    }
    CHECK_ACL(aclrtRecordEvent(_outputReadyEvent, stream));
    CHECK_ACL(aclrtStreamWaitEvent(currStream, _outputReadyEvent));
    CHECK_ACL(aclrtResetEvent(_outputReadyEvent, currStream));
#else
    CHECK_ACL(aclrtRecordEvent(_event, stream));
    CHECK_ACL(aclrtStreamWaitEvent(currStream, _event));
    CHECK_ACL(aclrtResetEvent(_event, currStream));
#endif
}

static inline HcclComm HcclCommFor(XRuntime &rt, enum commType type)
{
    switch (type) {
        case TP:
            return rt._tpComm;
        case DP:
            return rt._dpComm;
        case EP:
            return rt._epComm;
        default:
            return nullptr;
    }
}

static inline bool XcclActiveFor(const XRuntime &rt, enum commType type, enum XDtype dtype)
{
    XcclComm *xccl = nullptr;
    switch (type) {
        case TP:
            xccl = rt._tpXcclComm;
            break;
        case DP:
            xccl = rt._dpXcclComm;
            break;
        case EP:
            xccl = rt._epXcclComm;
            break;
        default:
            return false;
    }
    return xccl != nullptr && dtype != INT64;
}

bool XRuntime::AllGatherInGraphActive(enum commType type) const
{
    return !IsDummyRuntime() && _graphCommEnabled && !XcclActiveFor(*this, type, INT8);
}

bool XRuntime::ReduceScatterInGraphActive(enum commType type) const
{
    return !IsDummyRuntime() && _graphCommEnabled && !XcclActiveFor(*this, type, BF16);
}

void XRuntime::CaptureHcclAllGather(void *send, void *recv, uint32_t m, int hcclDtype,
                                    enum commType type)
{
    uint64_t sendBytes = static_cast<uint64_t>(m) * _agPerTokenBytes;
    if (m == 0 || m >= _agGraphs.size() || _agPerTokenBytes == 0) {
        throw std::runtime_error(
            std::string(__func__) + ": invalid m=" + std::to_string(m) +
            " (need 1<=m<=" + std::to_string(_agGraphs.size() - 1) +
            " and _agPerTokenBytes>0; PrepareAttn pre-captures m=1..maxBatch)");
    }
    if (_agGraphs[m].modelRI != nullptr) {
        return;  // already captured for this m
    }
    HcclComm comm = HcclCommFor(*this, type);
    if (comm == nullptr) {
        throw std::runtime_error(std::string(__func__) + ": HCCL comm is null for commType=" +
                                 std::to_string(static_cast<int>(type)) +
                                 " (need XLITE_DISABLE_XCCL and the matching size>1)");
    }
    HcclDataType dtype = static_cast<HcclDataType>(hcclDtype);
    aclmdlRI modelRI = nullptr;
    CHECK_ACL(aclmdlRICaptureBegin(stream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL));
    CHECK_HCCL(HcclAllGather(send, recv, sendBytes, dtype, comm, stream));
    CHECK_ACL(aclmdlRICaptureEnd(stream, &modelRI));
    _agGraphs[m] = GraphCaptureEntry{modelRI, send, recv};
}

void XRuntime::RunHcclAllGatherInGraph(void *send, void *recv, uint32_t m, int hcclDtype,
                                       enum commType type)
{
    if (m == 0 || m >= _agGraphs.size()) {
        throw std::runtime_error(std::string(__func__) + ": invalid m=" + std::to_string(m) +
                                 " (need 1<=m<=" + std::to_string(_agGraphs.size() - 1) + ")");
    }
    const GraphCaptureEntry &e = _agGraphs[m];
    if (e.modelRI == nullptr) {
        throw std::runtime_error(std::string(__func__) + ": no captured graph for m=" +
                                 std::to_string(m) + "; call CaptureHcclAllGather first");
    }
    if (send != e.sendAddr || recv != e.recvAddr) {
        std::stringstream ss;
        ss << __func__ << ": FIXED-ADDRESS VIOLATION for m=" << m << " dtype=" << hcclDtype
           << " commType=" << static_cast<int>(type) << " rank=" << _rankId
           << " : replay send=" << send << " recv=" << recv << " but captured send=" << e.sendAddr
           << " recv=" << e.recvAddr
           << ". The pool's bump pointer drifted (call sequence changed before AllGather) — "
           << "re-capture or fix the Get/Put ordering of packedSend/packedRecv.";
        throw std::runtime_error(ss.str());
    }
    CHECK_ACL(aclmdlRIExecuteAsync(e.modelRI, stream));
}

void XRuntime::AllGatherInGraph(void *send, void *recv, uint32_t m, int hcclDtype,
                                enum commType type)
{
    if (IsDummyRuntime()) {
        return;
    }
    RunHcclAllGatherInGraph(send, recv, m, hcclDtype, type);
}

void XRuntime::CaptureHcclReduceScatter(void *send, void *recv, uint32_t m, int hcclDtype,
                                        enum commType type)
{
    if (m == 0 || m >= _rsGraphs.size() || _rsHiddenSize == 0) {
        throw std::runtime_error(std::string(__func__) + ": invalid m=" + std::to_string(m) +
                                 " (need 1<=m<=" + std::to_string(_rsGraphs.size() - 1) +
                                 " and _rsHiddenSize>0; PrepareAttn pre-captures m=1..maxBatch)");
    }
    if (_rsGraphs[m].modelRI != nullptr) {
        return;  // already captured for this m
    }
    HcclComm comm = HcclCommFor(*this, type);
    if (comm == nullptr) {
        throw std::runtime_error(std::string(__func__) + ": HCCL comm is null for commType=" +
                                 std::to_string(static_cast<int>(type)) +
                                 " (need XLITE_DISABLE_XCCL and the matching size>1)");
    }
    HcclDataType dtype = static_cast<HcclDataType>(hcclDtype);
    uint64_t recvCount =
        static_cast<uint64_t>(m) * _rsHiddenSize;  // element count (dtype-agnostic)
    aclmdlRI modelRI = nullptr;
    CHECK_ACL(aclmdlRICaptureBegin(stream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL));
    CHECK_HCCL(HcclReduceScatter(send, recv, recvCount, dtype, HCCL_REDUCE_SUM, comm, stream));
    CHECK_ACL(aclmdlRICaptureEnd(stream, &modelRI));
    _rsGraphs[m] = GraphCaptureEntry{modelRI, send, recv};
}

void XRuntime::RunHcclReduceScatterInGraph(void *send, void *recv, uint32_t m, int hcclDtype,
                                           enum commType type)
{
    if (m == 0 || m >= _rsGraphs.size()) {
        throw std::runtime_error(std::string(__func__) + ": invalid m=" + std::to_string(m) +
                                 " (need 1<=m<=" + std::to_string(_rsGraphs.size() - 1) + ")");
    }
    const GraphCaptureEntry &e = _rsGraphs[m];
    if (e.modelRI == nullptr) {
        throw std::runtime_error(std::string(__func__) + ": no captured RS graph for m=" +
                                 std::to_string(m) + "; call CaptureHcclReduceScatter first");
    }
    if (send != e.sendAddr || recv != e.recvAddr) {
        std::stringstream ss;
        ss << __func__ << ": FIXED-ADDRESS VIOLATION for m=" << m << " dtype=" << hcclDtype
           << " commType=" << static_cast<int>(type) << " rank=" << _rankId
           << " : replay send=" << send << " recv=" << recv << " but captured send=" << e.sendAddr
           << " recv=" << e.recvAddr;
        throw std::runtime_error(ss.str());
    }
    CHECK_ACL(aclmdlRIExecuteAsync(e.modelRI, stream));
}

void XRuntime::ReduceScatterInGraph(void *send, void *recv, uint32_t m, int hcclDtype,
                                    enum commType type)
{
    if (IsDummyRuntime()) {
        return;
    }
    RunHcclReduceScatterInGraph(send, recv, m, hcclDtype, type);
}

void XRuntime::MemcpyH2D(void *dst, void *src, size_t size)
{
    CHECK_ACL(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE));
}

#ifdef XLITE_310P_LLM_FP16_POC
const uint8_t *XRuntime::CausalMaskHost310P(void)
{
    // Immutable for the runtime lifetime: queued H2D copies from any layer or
    // request can safely reference this storage until the stream is drained
    // by the destructor. No per-layer host allocation or staging overwrite.
    constexpr size_t side = XLITE_310P_MAX_SEQ_LEN;
    if (_causalMaskPinnedHost == nullptr) {
        CHECK_ACL(aclrtMallocHost(&_causalMaskPinnedHost, side * side));
        auto *mask = static_cast<uint8_t *>(_causalMaskPinnedHost);
        std::memset(mask, 1, side * side);
        for (size_t row = 0; row < side; ++row) {
            std::memset(mask + row * side, 0, row + 1);
        }
    }
    return static_cast<const uint8_t *>(_causalMaskPinnedHost);
}
#endif

void XRuntime::MemcpyD2H(void *dst, void *src, size_t size)
{
    CHECK_ACL(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST));
}

void XRuntime::MemcpyD2HAsync(void *dst, void *src, size_t size)
{
    CHECK_ACL(aclrtMemcpyAsync(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST, stream));
}

void XRuntime::UpdateCoreNum(float blockDimUtilization)
{
    aicNum =
        static_cast<uint32_t>(std::round(static_cast<float>(originAicNum) * blockDimUtilization));
    aivNum =
        static_cast<uint32_t>(std::round(static_cast<float>(originAivNum) * blockDimUtilization));
}

void XRuntime::SetCurrentContext()
{
    CHECK_ACL(aclrtSetCurrentContext(context));
}

void XRuntime::NotifyWaitPeerStream()
{
    CHECK_ACL(aclrtWaitAndResetNotify(notify, stream, 0));
}

void XRuntime::NotifyRecordPeerStream()
{
    CHECK_ACL(aclrtRecordNotify(peerNotify, stream));
}

int XRuntime::InitTensorPool(size_t sizeMB)
{
    if (sizeMB != 0) {
        Init(sizeMB);
    }
    return 0;
}

XTensor &XRuntime::GetTensor(std::vector<size_t> shape, enum XDtype dtype, DebugSrcLoc loc)
{
    return _pool->GetTensor(std::move(shape), dtype, loc);
}

void XRuntime::PutTensor(XTensor &t)
{
    _pool->PutTensor(t);
}

bool XRuntime::TensorInPool(XTensor &t)
{
    return _pool->TensorInPool(t);
}

int64_t XRuntime::GetTensorOffset(XTensor &t)
{
    if (!_pool->TensorInPool(t)) {
        return -1;
    }
    uint64_t poolStart = reinterpret_cast<uint64_t>(_pool->Ptr());
    uint64_t tensorStart = reinterpret_cast<uint64_t>(t.ptr);
    return static_cast<int64_t>(tensorStart - poolStart);
}

void XRuntime::ConfigureSwizzle(uint32_t swizzle, bool useSwizzleTable)
{
#ifdef XLITE_310P_LLM_FP16_POC
    if (swizzle != 0 || useSwizzleTable) {
        throw std::runtime_error("Ascend310P llm_fp16 POC does not support matmul swizzle tuning");
    }
#endif
    defaultMatmulSwizzle = swizzle;
    disableSwizzleTable = !useSwizzleTable;
}

void XDummyRuntime::InitDummyRuntime(size_t sizeMB)
{
    if (_inited) {
        return;
    }
    aclError initRet = aclInit(nullptr);
    uint32_t count;
    if (initRet == ACL_ERROR_REPEAT_INITIALIZE) {
        _initOutside = true;
    } else {
        CHECK_ACL(initRet);
    }
    CHECK_ACL(aclrtGetDeviceCount(&count));
    _nDevPerNode = count;

    _pool = new XDummyTensorPool(sizeMB << MB_BIT, _rankId);
    if (_pool->Init()) {
        throw std::runtime_error("XDummyRuntime: tensor pool initialization failed");
    }
    _rankSize = _tpSize * _dpSize;

    (void)InitDummyXcclComm();

    int64_t val;
    CHECK_ACL(aclGetDeviceCapability(_devid, ACL_DEVICE_INFO_AI_CORE_NUM, &val));
    aicNum = static_cast<uint32_t>(val);
    CHECK_ACL(aclGetDeviceCapability(_devid, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &val));
    aivNum = static_cast<uint32_t>(val);
    reportedAivNum = aivNum;
#ifdef XLITE_310P_LLM_FP16_POC
    if (aicNum == 0) {
        throw std::runtime_error("Ascend 310P reported zero AI cores");
    }
    aivNum = aicNum;
#endif
    originAicNum = aicNum;
    originAivNum = aivNum;

    const char *envCommOptimizeLen = std::getenv("XLITE_COMM_OPTIMIZE_LEN");
    if (envCommOptimizeLen) {
        char *endPtr = nullptr;
        long val = strtol(envCommOptimizeLen, &endPtr, 10);
        if (endPtr != envCommOptimizeLen && *endPtr == '\0' && val >= 0) {
            commOptimizeLen = static_cast<uint32_t>(val);
        }
    }

    const char *ratioPerEPEnv = std::getenv("XLITE_ACTIVE_TOKENS_RATIO_PER_EP");
    if (ratioPerEPEnv) {
        char *endPtr = nullptr;
        double val = strtod(ratioPerEPEnv, &endPtr);
        double min = 1 / static_cast<double>(_moeEpSize);
        double max = 1.0f;
        if (endPtr != ratioPerEPEnv && *endPtr == '\0' && std::isfinite(val)) {
            activeTokensRatioPerEp = val;
        }
        if (activeTokensRatioPerEp < min) {
            activeTokensRatioPerEp = min;
        }
        if (activeTokensRatioPerEp > max) {
            activeTokensRatioPerEp = max;
        }
    }

    _inited = true;
}

int XDummyRuntime::InitDummyXcclComm(void)
{
    const char *envDisableXccl = std::getenv("XLITE_DISABLE_XCCL");
    const char *envDeterministic = std::getenv("HCCL_DETERMINISTIC");

    if (_rankSize == 1 || _rankSize > XLITE_CCL_MAX_RANK_SIZE) {
        return 0;
    }

    if (isEnvironmentVariableTrue(envDisableXccl) || isEnvironmentVariableTrue(envDeterministic)) {
        return 0;
    }

    bool enableTpXccl = (_tpSize > 1 && _tpSize <= _nDevPerNode);
    bool enableDpXccl = (_dpSize > 1 && _rankSize <= _nDevPerNode);

    if (!enableTpXccl && !enableDpXccl) {
        return 0;
    }

    if (enableTpXccl) {
        _tpXcclComm = new XcclComm(_rankId % _tpSize, _tpSize);
    }

    if (enableDpXccl) {
        _dpXcclComm = new XcclComm(_rankId / _tpSize, _dpSize);
    }
    return 0;
}

size_t XDummyRuntime::maxUsedSize(void)
{
    auto *dummyPool = dynamic_cast<XDummyTensorPool *>(_pool);
    return dummyPool ? dummyPool->maxUsedSize : 0;
}
