/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <cmath>
#include <sstream>
#include "ascend.h"
#include "base.h"
#include "runtime.h"
#include "op.h"
#include "sock.h"
#include "ccl.h"
#include "auto_tuner.h"

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
    originAicNum = aicNum;
    originAivNum = aivNum;

    CHECK_ACL(aclrtCreateEvent(&_event));
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

    delete _pool;
    if (_event) {
        (void)aclrtDestroyEvent(_event);
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
    if (notify) {
        (void)aclrtDestroyNotify(notify);
    }
    if (stream) {
        (void)aclrtDestroyStream(stream);
    }
    (void)aclrtResetDevice(static_cast<int32_t>(_devid));

    if (_position.ptr) {
        (void)aclrtFree(_position.ptr);
    }
    if (_slotMapping.ptr) {
        (void)aclrtFree(_slotMapping.ptr);
    }
    if (_cachedLens.ptr) {
        (void)aclrtFree(_cachedLens.ptr);
    }
    if (_lens.ptr) {
        (void)aclrtFree(_lens.ptr);
    }
    if (_queryStartLoc.ptr) {
        (void)aclrtFree(_queryStartLoc.ptr);
    }
    if (_blockTables.ptr) {
        (void)aclrtFree(_blockTables.ptr);
    }
    if (_tokensPerEpGroupAllEpHost.ptr) {
        (void)aclrtFreeHost(_tokensPerEpGroupAllEpHost.ptr);
    }
    if (_agSendBuf.ptr) {
        (void)aclrtFree(_agSendBuf.ptr);
    }
    if (_agRecvBuf.ptr) {
        (void)aclrtFree(_agRecvBuf.ptr);
    }
    if (_dsaTopkBuffer.ptr) {
        (void)aclrtFree(_dsaTopkBuffer.ptr);
    }

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

void XRuntime::InitAttn(XModelAttnMeta &attnMeta, uint64_t maxBatchedTokens, uint64_t maxBatch,
                        uint64_t maxSeqLen, const std::vector<uint32_t> &blockSizes,
                        uint32_t indexTopK)
{
    std::vector<uint32_t> vgatherIndices;
    size_t size;
    void *ptr;

    switch (attnMeta.version) {
        case 0:
            size = maxBatchedTokens * XDtypeBit(INT64) / 8;
            CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
            _position.Init({maxBatchedTokens}, INT64, ptr);
        case 1:
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

            size = maxBatch * DIV_ROUND_UP(maxSeqLen, blockSizes[0]) * XDtypeBit(INT32) / 8;
            CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
            _blockTables.Init({maxBatch, DIV_ROUND_UP(maxSeqLen, blockSizes[0])}, INT32, ptr);
            break;
        case 2:
            break;
        default:
            throw std::runtime_error(std::string(__func__) +
                                     ": unsupported attention meta version " +
                                     std::to_string(attnMeta.version));
    }

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

static void CheckAttnMetaV2(const XModelAttnMeta &attnMeta, uint32_t batch, uint32_t batchedTokens,
                            std::vector<uint32_t> blockSizes, uint32_t maxTotalLens)
{
    if (attnMeta.lens.shape.size() != 1 || attnMeta.lens.shape[0] < batch) {
        throw std::runtime_error(
            std::string(__FILE__) + ":" + std::to_string(__LINE__) +
            ": lens shape mismatch, expect 1D len >= " + std::to_string(batch));
    }
    if (attnMeta.cachedLens.shape.size() != 1 || attnMeta.cachedLens.shape[0] < batch) {
        throw std::runtime_error(
            std::string(__FILE__) + ":" + std::to_string(__LINE__) +
            ": cachedLens shape mismatch, expect 1D len >= " + std::to_string(batch));
    }
    if (attnMeta.queryStartLoc.shape.size() != 1 || attnMeta.queryStartLoc.shape[0] < batch) {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                 ": queryStartLoc shape mismatch, got dim " +
                                 std::to_string(attnMeta.queryStartLoc.shape.size()) + " len " +
                                 std::to_string(attnMeta.queryStartLoc.shape.empty()
                                                    ? 0
                                                    : attnMeta.queryStartLoc.shape[0]) +
                                 ", expect 1D len >= " + std::to_string(batch));
    }
    if (attnMeta.position.shape.size() != 1 || attnMeta.position.shape[0] < batchedTokens) {
        throw std::runtime_error(
            std::string(__FILE__) + ":" + std::to_string(__LINE__) +
            ": position shape mismatch, expect 1D len >= " + std::to_string(batchedTokens));
    }
    uint32_t expectedKvCacheNum = blockSizes.size();
    if (attnMeta.slotMapping.size() != expectedKvCacheNum) {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                 ": slotMapping size mismatch, expect " +
                                 std::to_string(expectedKvCacheNum) + " got " +
                                 std::to_string(attnMeta.slotMapping.size()));
    }
    if (attnMeta.blockTables.size() != expectedKvCacheNum) {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                 ": blockTables size mismatch, expect " +
                                 std::to_string(expectedKvCacheNum) + " got " +
                                 std::to_string(attnMeta.blockTables.size()));
    }
    for (size_t i = 0; i < expectedKvCacheNum; i++) {
        if (attnMeta.slotMapping[i].shape.size() != 1 ||
            attnMeta.slotMapping[i].shape[0] < batchedTokens) {
            throw std::runtime_error(
                std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                ": slotMapping shape mismatch, expect 1D len >= " + std::to_string(batchedTokens));
        }
        uint32_t maxNumBlocks = DIV_ROUND_UP(maxTotalLens, blockSizes[i]);
        if (attnMeta.blockTables[i].shape.size() != 2 || attnMeta.blockTables[i].shape[0] < batch ||
            attnMeta.blockTables[i].shape[1] < maxNumBlocks) {
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                     ": blockTables shape mismatch, expect 2D [batch, >= " +
                                     std::to_string(maxNumBlocks) + "]");
        }
    }
}

#ifdef XLITE_DEBUG_ON
// Verify v2 device tensors equal what v1's host-side algorithm would produce.
void XRuntime::VerifyAttnMetaV2(const XModelAttnMeta &attnMeta, std::vector<uint32_t> blockSizes)
{
    if (IsDummyRuntime()) {
        return;
    }
    const uint32_t batch = _batch;
    const uint32_t batchedTokens = this->batchedTokens;
    const uint32_t maxNumBlocks = attnMeta.blockTables[0].shape[1];
    aclrtStream stream = this->stream;
    uint32_t rankId = this->rankId();
    const auto fail = [&](const char *field, uint32_t idx, uint64_t device, uint64_t expected,
                          int line) {
        std::stringstream ss;
        ss << std::string(__FILE__) << ":" << std::to_string(line)
           << ": VerifyAttnMetaV2: " << field << " mismatch (R" << rankId << ") idx=" << idx
           << " device=" << device << " expected=" << expected;
        throw std::runtime_error(ss.str());
    };

    // lens / cachedLens
    {
        std::vector<uint32_t> host(batch);
        size_t bytes = batch * XDtypeBit(INT32) / 8;
        CHECK_ACL(aclrtMemcpyAsync(host.data(), bytes, attnMeta.lens.ptr, bytes,
                                   ACL_MEMCPY_DEVICE_TO_HOST, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        for (uint32_t i = 0; i < batch; i++) {
            if (host[i] != attnMeta.lensCpu[i]) {
                fail("lens", i, host[i], attnMeta.lensCpu[i], __LINE__);
            }
        }
    }
    {
        std::vector<uint32_t> host(batch);
        size_t bytes = batch * XDtypeBit(INT32) / 8;
        CHECK_ACL(aclrtMemcpyAsync(host.data(), bytes, attnMeta.cachedLens.ptr, bytes,
                                   ACL_MEMCPY_DEVICE_TO_HOST, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        for (uint32_t i = 0; i < batch; i++) {
            if (host[i] != attnMeta.cachedLensCpu[i]) {
                fail("cachedLens", i, host[i], attnMeta.cachedLensCpu[i], __LINE__);
            }
        }
    }

    // queryStartLoc
    {
        std::vector<uint32_t> host(batch);
        size_t bytes = batch * XDtypeBit(INT32) / 8;
        CHECK_ACL(aclrtMemcpyAsync(host.data(), bytes, attnMeta.queryStartLoc.ptr, bytes,
                                   ACL_MEMCPY_DEVICE_TO_HOST, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        uint32_t acc = 0;
        for (uint32_t i = 0; i < batch; i++) {
            if (host[i] != acc) {
                fail("queryStartLoc", i, host[i], acc, __LINE__);
            }
            acc += attnMeta.lensCpu[i];
        }
    }

    // blockTables
    std::vector<uint32_t> blockTablesHost(batch * maxNumBlocks);
    {
        size_t bytes = batch * maxNumBlocks * XDtypeBit(INT32) / 8;
        CHECK_ACL(aclrtMemcpyAsync(blockTablesHost.data(), bytes, attnMeta.blockTables[0].ptr,
                                   bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    // slotMapping
    {
        std::vector<uint32_t> host(batchedTokens);
        size_t bytes = batchedTokens * XDtypeBit(INT32) / 8;
        CHECK_ACL(aclrtMemcpyAsync(host.data(), bytes, attnMeta.slotMapping[0].ptr, bytes,
                                   ACL_MEMCPY_DEVICE_TO_HOST, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        uint32_t k = 0;
        for (uint32_t i = 0; i < batch; i++) {
            for (uint32_t j = 0; j < attnMeta.lensCpu[i]; j++) {
                uint32_t pos = attnMeta.cachedLensCpu[i] + j;
                uint32_t blockId = pos / blockSizes[0];
                uint32_t id = pos % blockSizes[0];
                uint32_t expect = blockTablesHost[i * maxNumBlocks + blockId] * blockSizes[0] + id;
                if (host[k] != expect) {
                    fail("slotMapping", k, host[k], expect, __LINE__);
                }
                k++;
            }
        }
    }

    // position
    {
        std::vector<uint64_t> host(batchedTokens);
        size_t bytes = batchedTokens * XDtypeBit(INT64) / 8;
        CHECK_ACL(aclrtMemcpyAsync(host.data(), bytes, attnMeta.position.ptr, bytes,
                                   ACL_MEMCPY_DEVICE_TO_HOST, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        uint32_t k = 0;
        for (uint32_t i = 0; i < batch; i++) {
            for (uint32_t j = 0; j < attnMeta.lensCpu[i]; j++) {
                uint64_t expect = static_cast<uint64_t>(attnMeta.cachedLensCpu[i]) + j;
                if (host[k] != expect) {
                    fail("position", k, host[k], expect, __LINE__);
                }
                k++;
            }
        }
    }
}
#endif  // XLITE_DEBUG_ON

void XRuntime::PrepareAttn(XModelAttnMeta &attnMeta, uint64_t maxBatchedTokens, uint64_t maxBatch,
                           uint64_t maxSeqLen, uint32_t nHeads, uint32_t nKVHeads,
                           std::vector<uint32_t> blockSizes, uint32_t hiddenSize,
                           uint32_t nRoutedExperts, uint32_t defDpSize, int inputDtype,
                           int weightsDtype, uint32_t indexTopK)
{
    if (!_attnInitialized) {
        InitAttn(attnMeta, maxBatchedTokens, maxBatch, maxSeqLen, blockSizes, indexTopK);
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
    uint32_t batch = attnMeta.lensCpu.size();
    std::vector<uint32_t> lens(batch);
    std::vector<uint32_t> cachedLens(batch);
    std::vector<uint32_t> queryStartLoc(batch);
    std::vector<uint32_t> numBlocks(batch);
    std::vector<uint32_t> totalLens(batch);
    std::vector<uint32_t> slotMapping, blockTables;
    std::vector<uint64_t> position;
    uint32_t queryStart, blockId, id, k;
    size_t size;

    if (batch == 0) {
        throw std::runtime_error(std::string(__func__) + ":" + std::to_string(__LINE__) +
                                 ": invalid batchSize: " + std::to_string(batch));
    }

    if (attnMeta.attnType == XMODEL_ATTN_CXA && blockSizes.size() < CXA_SWA_KV + 1) {
        throw std::runtime_error(
            std::string(__func__) + ":" + std::to_string(__LINE__) +
            ": invalid blockSizes for CXA, expect size >= " + std::to_string(CXA_SWA_KV + 1));
    }

    batchedTokens = 0;
    _maxTotalLens = 0;
    _batch = batch;
    queryStart = 0;
    bool allCached = true;
    bool anyMultiToken = false;
    for (uint32_t i = 0; i < batch; i++) {
        lens[i] = attnMeta.lensCpu[i];
        cachedLens[i] = attnMeta.cachedLensCpu[i];
        totalLens[i] = lens[i] + cachedLens[i];
        queryStartLoc[i] = queryStart;
        queryStart += lens[i];
        _maxTotalLens = totalLens[i] > _maxTotalLens ? totalLens[i] : _maxTotalLens;
        batchedTokens += lens[i];
        numBlocks[i] = DIV_ROUND_UP(totalLens[i], blockSizes[0]);
        if (cachedLens[i] == 0) {
            allCached = false;
        }
        if (lens[i] != 1) {
            anyMultiToken = true;
        }
    }
    _hostLens = lens;
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

    if (IsDummyRuntime() || _maxTotalLens <= MAX_KV_TILE_SIZE) {
        _tileSizeOfCachedKV = MAX_KV_TILE_SIZE;
    } else {
        uint32_t localHeads = std::max(nHeads / _tpSize, static_cast<uint32_t>(1));
        uint32_t localKvHeads = std::max(nKVHeads / _tpSize, static_cast<uint32_t>(1));
        _tileSizeOfCachedKV = GetTileSizeOfCachedKV(
            cachedLens, lens, localHeads / localKvHeads, localKvHeads,
            attnMeta.attnType == XMODEL_ATTN_CXA ? blockSizes[CXA_COMPRESS_KV] : blockSizes[0],
            aicNum);
    }

    size = batch * XDtypeBit(INT32) / 8;

    switch (attnMeta.version) {
        case 0:
        case 1: {
            uint32_t maxNumBlocks = DIV_ROUND_UP(_maxTotalLens, blockSizes[0]);
            CHECK_ACL(aclrtMemcpyAsync(_lens.ptr, size, lens.data(), size,
                                       ACL_MEMCPY_HOST_TO_DEVICE, stream));
            CHECK_ACL(aclrtMemcpyAsync(_cachedLens.ptr, size, cachedLens.data(), size,
                                       ACL_MEMCPY_HOST_TO_DEVICE, stream));
            CHECK_ACL(aclrtMemcpyAsync(_queryStartLoc.ptr, size, queryStartLoc.data(), size,
                                       ACL_MEMCPY_HOST_TO_DEVICE, stream));
            _attnLens = _lens;
            _attnCachedLens = _cachedLens;
            _attnQueryStartLoc = _queryStartLoc;

            position.resize(batchedTokens);
            slotMapping.resize(batchedTokens);
            k = 0;
            if (attnMeta.blockTablesCpu.size() < batch) {
                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                         ": invalid blocktable(" +
                                         std::to_string(attnMeta.blockTablesCpu.size()) + ")");
            }
            for (uint32_t i = 0; i < batch; i++) {
                if (attnMeta.blockTablesCpu[i].size() < numBlocks[i]) {
                    throw std::runtime_error(std::string(__FILE__) + ":" +
                                             std::to_string(__LINE__) +
                                             ": block table too small (" +
                                             std::to_string(attnMeta.blockTablesCpu[i].size()) +
                                             " < " + std::to_string(numBlocks[i]) + ")");
                }
                for (uint32_t j = 0; j < lens[i]; j++) {
                    position[k] = cachedLens[i] + j;
                    blockId = position[k] / blockSizes[0];
                    id = position[k] % blockSizes[0];
                    slotMapping[k++] = attnMeta.blockTablesCpu[i][blockId] * blockSizes[0] + id;
                }
            }
            size = batchedTokens * XDtypeBit(INT32) / 8;
            CHECK_ACL(aclrtMemcpyAsync(_slotMapping.ptr, size, slotMapping.data(), size,
                                       ACL_MEMCPY_HOST_TO_DEVICE, stream));
            _attnSlotMapping = {_slotMapping};

            blockTables.resize(batch * maxNumBlocks);
            for (uint32_t i = 0; i < batch; i++) {
                for (uint32_t j = 0; j < numBlocks[i]; j++) {
                    blockTables[i * maxNumBlocks + j] = attnMeta.blockTablesCpu[i][j];
                }
            }
            _blockTablesHost = blockTables;
            size = batch * maxNumBlocks * XDtypeBit(INT32) / 8;
            CHECK_ACL(aclrtMemcpyAsync(_blockTables.ptr, size, blockTables.data(), size,
                                       ACL_MEMCPY_HOST_TO_DEVICE, stream));
            _attnBlockTables = {_blockTables};
            _attnBlockTables[0].View({batch, maxNumBlocks});
            if (attnMeta.version == 0) {
                size = batchedTokens * XDtypeBit(INT64) / 8;
                CHECK_ACL(aclrtMemcpyAsync(_position.ptr, size, position.data(), size,
                                           ACL_MEMCPY_HOST_TO_DEVICE, stream));
                _attnPosition = _position;
            } else {
                _attnPosition = attnMeta.position;
            }
            break;
        }
        case 2: {
            // Version 2: lens / cachedLens / queryStartLoc / slotMapping / blockTables are
            // pre-built on the Python side as device tensors; shape-check and alias (zero copy).
            CheckAttnMetaV2(attnMeta, batch, batchedTokens, blockSizes, _maxTotalLens);
#ifdef XLITE_DEBUG_ON
            VerifyAttnMetaV2(attnMeta, blockSizes);
#endif
            _attnLens = attnMeta.lens;
            _attnCachedLens = attnMeta.cachedLens;
            _attnQueryStartLoc = attnMeta.queryStartLoc;
            _attnSlotMapping = attnMeta.slotMapping;
            _attnBlockTables = attnMeta.blockTables;
            _attnPosition = attnMeta.position;
            break;
        }
        default:
            throw std::runtime_error(
                std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                ": invalid attnMeta version: " + std::to_string(attnMeta.version));
    }
}

void XRuntime::Synchronize(void)
{
    CHECK_ACL(aclrtSynchronizeStream(stream));
}

void XRuntime::EventWaitCurrStream(aclrtStream currStream)
{
    CHECK_ACL(aclrtRecordEvent(_event, currStream));
    CHECK_ACL(aclrtStreamWaitEvent(stream, _event));
    CHECK_ACL(aclrtResetEvent(_event, stream));
}

void XRuntime::EventRecordCurrStream(aclrtStream currStream)
{
    CHECK_ACL(aclrtRecordEvent(_event, stream));
    CHECK_ACL(aclrtStreamWaitEvent(currStream, _event));
    CHECK_ACL(aclrtResetEvent(_event, currStream));
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
