/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "kernel_operator.h"
#include "kernel_macro.h"
#include "kernel_param.h"

using namespace AscendC;

template <typename Dtype>
class AllReduce
{
public:
    __aicore__ inline AllReduce()
    {
    }

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint64_t count, uint32_t rankId,
                                uint32_t rankSize, uint64_t generation, GM_ADDR param,
                                uint32_t copySize, bool fetchOffset)
    {
        __gm__ struct XcclParam *xcclParam = (__gm__ struct XcclParam *)param;
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

        DataCacheCleanAndInvalid(xcclParam->ipcMems);
        DataCacheCleanAndInvalid(xcclParam->ipcXTensorMems);
        for (uint32_t r = 0; r < rankSize; r++) {
            this->param.ipcMems[r] = xcclParam->ipcMems[r];
            this->param.ipcXTensorMems[r] = xcclParam->ipcXTensorMems[r];
        }
        this->count = count;
        this->myRankId = rankId;
        this->rankSize = rankSize;
        this->coreIdx = block_idx;
        this->coreNum = block_num;
        this->countPerRank = DIV_ROUND_UP(count, rankSize);
        this->blockNum = rankSize;
        this->offsetCurrRank = countPerRank * myRankId;
        this->generation = generation;
        this->skipMyRank = input == output;
        copySize = ROUND_DOWN(copySize, UB_BUF_ALIGN_SIZE);
        this->copyCount = copySize / sizeof(Dtype);

        uint64_t off = 0;
        flagBuf.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
        flagBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += UB_BUF_ALIGN_SIZE;
        uint64_t ubBufBytes = COPY_SIZE;
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            ubBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
            ubBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += ubBufBytes;
        }

        for (uint32_t r = 0; r < rankSize; r++) {
            this->ipcFlagBuf[r].SetGlobalBuffer(
                (__gm__ uint32_t *)(this->param.ipcMems[r] + XLITE_IPC_MEM_FLAG_OFFSET));
        }

        if (coreIdx == 0) {
            if (fetchOffset) {
                __gm__ uint64_t *inputOffset = (__gm__ uint64_t *)(this->param.ipcMems[myRankId]);
                __gm__ uint64_t *outputOffset =
                    (__gm__ uint64_t *)(this->param.ipcMems[myRankId] + sizeof(uint64_t));
                *inputOffset = (uint64_t)input - (uint64_t)this->param.ipcXTensorMems[myRankId];
                *outputOffset = (uint64_t)output - (uint64_t)this->param.ipcXTensorMems[myRankId];
                DataCacheCleanAndInvalid(inputOffset);
                DataCacheCleanAndInvalid(outputOffset);
            }
            SetIpcFlag(0, generation);
        }

        uint32_t idx = 0;
        for (uint32_t r = 0; r < rankSize; r++) {
            if (r == myRankId) {
                continue;
            }
            rankIdxMapping[idx++] = r;
        }

        WorkSplit(rankSize - 1, &syncWorkStart, &syncWorkEnd);
        for (uint32_t workIdx = syncWorkStart; workIdx < syncWorkEnd; workIdx++) {
            WaitIpcFlag(rankIdxMapping[workIdx], 0, generation);
        }
        PipeBarrier<PIPE_ALL>();

        CrossCoreSetFlag<0x0, PIPE_MTE3>(1);
        CrossCoreWaitFlag(1);

        for (uint32_t r = 0; r < rankSize; r++) {
            if (r == myRankId) {
                inputBuf[r].SetGlobalBuffer((__gm__ Dtype *)input);
                outputBuf[r].SetGlobalBuffer((__gm__ Dtype *)output);
                continue;
            }
            struct XcclIpcMemData localIpcMemData;
            if (fetchOffset) {
                __gm__ uint64_t *inputOffset = (__gm__ uint64_t *)(this->param.ipcMems[r]);
                __gm__ uint64_t *outputOffset =
                    (__gm__ uint64_t *)(this->param.ipcMems[r] + sizeof(uint64_t));
                DataCacheCleanAndInvalid(inputOffset);
                DataCacheCleanAndInvalid(outputOffset);
                localIpcMemData.inputOffset = *inputOffset;
                localIpcMemData.outputOffset = *outputOffset;
            } else {
                localIpcMemData.inputOffset =
                    (uint64_t)input - (uint64_t)this->param.ipcXTensorMems[myRankId];
                localIpcMemData.outputOffset =
                    (uint64_t)output - (uint64_t)this->param.ipcXTensorMems[myRankId];
            }
            inputBuf[r].SetGlobalBuffer(
                (__gm__ Dtype *)(this->param.ipcXTensorMems[r] + localIpcMemData.inputOffset));
            outputBuf[r].SetGlobalBuffer(
                (__gm__ Dtype *)(this->param.ipcXTensorMems[r] + localIpcMemData.outputOffset));
        }

        // each rank process countPerRank elements
        this->countCurrRank = countPerRank;
        if ((myRankId + 1) * countPerRank > count) {
            this->countCurrRank = count - myRankId * countPerRank;
        }
        this->countLastRank = count - countPerRank * (rankSize - 1);
    }

    __aicore__ inline void SetIpcFlag(uint32_t flagId, uint32_t value)
    {
        flagBuf.SetValue(0, value);
        PipeBarrier<PIPE_ALL>();
        DataCopyParams copyParams;
        copyParams.blockLen = sizeof(uint32_t);
        copyParams.blockCount = 1;
        DataCopyPad(ipcFlagBuf[myRankId][flagId], flagBuf, copyParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void WaitIpcFlag(uint32_t rankId, uint32_t flagId, uint32_t expectValue)
    {
        uint32_t flagValue = 0;
        do {
            DataCopyParams copyParams;
            copyParams.blockLen = sizeof(uint32_t);
            copyParams.blockCount = 1;
            DataCopyPadParams padParams;
            padParams.isPad = false;
            DataCopyPad(flagBuf, ipcFlagBuf[rankId][flagId], copyParams, padParams);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
            flagValue = flagBuf.GetValue(0);
        } while (flagValue < expectValue);
    }

    // split work among cores
    __aicore__ inline void WorkSplit(uint32_t workNum, uint32_t *start, uint32_t *end)
    {
        uint32_t remain = workNum % coreNum;
        uint32_t avg = workNum / coreNum;
        if (coreIdx < remain) {
            *start = coreIdx * avg + coreIdx;
            *end = *start + avg + 1;
        } else {
            *start = coreIdx * avg + remain;
            *end = *start + avg;
        }
        if (*end > workNum) {
            *end = workNum;
        }
    }

    __aicore__ inline void CopyGMtoUbuf(LocalTensor<Dtype> dst, GlobalTensor<Dtype> src,
                                        uint64_t count)
    {
        if (count * sizeof(Dtype) % BLOCK_SIZE == 0) {
            DataCopy(dst, src, count);
        } else {
            DataCopyParams copyParams;
            copyParams.blockLen = count * sizeof(Dtype);
            copyParams.blockCount = 1;
            DataCopyPadParams padParams;
            padParams.isPad = false;
            DataCopyPad(dst, src, copyParams, padParams);
        }
    }

    __aicore__ inline void CopyUbufToGM(GlobalTensor<Dtype> dst, LocalTensor<Dtype> src,
                                        uint64_t count)
    {
        if (count * sizeof(Dtype) % BLOCK_SIZE == 0) {
            DataCopy(dst, src, count);
        } else {
            DataCopyParams copyParams;
            copyParams.blockLen = count * sizeof(Dtype);
            copyParams.blockCount = 1;
            DataCopyPad(dst, src, copyParams);
        }
    }

    __aicore__ inline void Run()
    {
        int curr = 0;
        // reduce-scatter phase
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0 + i);
        }

        if (!skipMyRank) {
            uint64_t countPerCore = DIV_ROUND_UP(countCurrRank, coreNum);
            uint64_t offset = countPerCore * coreIdx;
            uint64_t count = countPerCore;
            if (offset + count > countCurrRank) {
                count = countCurrRank - offset;
            }
            uint64_t tmpCopyCount = COPY_SIZE / sizeof(Dtype);
            uint32_t copyNum = DIV_ROUND_UP(count, tmpCopyCount);
            for (uint32_t copyIdx = 0; copyIdx < copyNum; copyIdx++) {
                uint64_t copyOffset = copyIdx * tmpCopyCount;
                uint64_t gmOffset = offsetCurrRank + offset + copyOffset;
                uint64_t currCopyCount = tmpCopyCount;
                if (copyOffset + currCopyCount > count) {
                    currCopyCount = count - copyOffset;
                }
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0 + curr);
                CopyGMtoUbuf(ubBuf[curr], inputBuf[myRankId][gmOffset], currCopyCount);
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0 + curr);
                CopyUbufToGM(outputBuf[myRankId][gmOffset], ubBuf[curr], currCopyCount);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0 + curr);
                curr = (curr + 1) % PINGPONG_BUF_NUM;
            }
            CrossCoreSetFlag<0x0, PIPE_MTE3>(1);
            CrossCoreWaitFlag(1);
        }
        SetAtomicAdd<Dtype>();

        uint32_t rankPerCore, corePerRank;
        if (rankSize > coreNum) {
            rankPerCore = DIV_ROUND_UP(rankSize, coreNum);
            corePerRank = 1;
        } else {
            rankPerCore = 1;
            corePerRank = coreNum / rankSize;
        }

        for (int r = 0; r < rankPerCore; r++) {
            uint32_t processRankIdx =
                rankSize > coreNum ? coreIdx * rankPerCore + r : coreIdx / corePerRank;
            if (processRankIdx == myRankId || processRankIdx >= rankSize) {
                continue;
            }
            uint32_t taskIdx = rankSize > coreNum ? 0 : coreIdx % corePerRank;
            uint64_t countPerTask = DIV_ROUND_UP(countCurrRank, corePerRank);
            uint64_t taskOffset = taskIdx * countPerTask;
            uint64_t taskCount = countPerTask;
            if (taskOffset + taskCount > countCurrRank) {
                taskCount = countCurrRank - taskOffset;
            }
            uint32_t copyNum = DIV_ROUND_UP(taskCount, copyCount);
            uint64_t offset = myRankId * countPerRank + taskOffset;
            for (uint32_t copyIdx = 0; copyIdx < copyNum; copyIdx++) {
                uint64_t copyOffset = copyIdx * copyCount;
                uint64_t gmOffset = offset + copyOffset;
                uint64_t currCopyCount = copyCount;
                if (copyOffset + currCopyCount > taskCount) {
                    currCopyCount = taskCount - copyOffset;
                }
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0 + curr);
                CopyGMtoUbuf(ubBuf[curr], inputBuf[processRankIdx][gmOffset], currCopyCount);
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0 + curr);
                CopyUbufToGM(outputBuf[myRankId][gmOffset], ubBuf[curr], currCopyCount);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0 + curr);
                curr = (curr + 1) % PINGPONG_BUF_NUM;
            }
        }

        // clear atomic
        SetAtomicNone();

        CrossCoreSetFlag<0x0, PIPE_MTE3>(1);
        CrossCoreWaitFlag(1);
        if (coreIdx == 0) {
            SetIpcFlag(1, generation);
        }
        for (int r = 0; r < rankPerCore; r++) {
            uint32_t processRankIdx =
                rankSize > coreNum ? coreIdx * rankPerCore + r : coreIdx / corePerRank;
            if (processRankIdx == myRankId || processRankIdx >= rankSize) {
                continue;
            }
            WaitIpcFlag(processRankIdx, 1, generation);
        }
        PipeBarrier<PIPE_ALL>();

        // allgather phase
        for (int r = 0; r < rankPerCore; r++) {
            uint32_t processRankIdx =
                rankSize > coreNum ? coreIdx * rankPerCore + r : coreIdx / corePerRank;
            if (processRankIdx == myRankId || processRankIdx >= rankSize) {
                continue;
            }
            uint64_t countCurrProcRank =
                (processRankIdx == rankSize - 1) ? countLastRank : countPerRank;
            uint32_t taskIdx = rankSize > coreNum ? 0 : coreIdx % corePerRank;
            uint64_t countPerTask = DIV_ROUND_UP(countCurrProcRank, corePerRank);
            uint64_t taskOffset = taskIdx * countPerTask;
            uint64_t outOffset = processRankIdx * countPerRank + taskOffset;
            uint64_t taskCount = countPerTask;
            if (taskOffset + taskCount > countCurrRank) {
                taskCount = countCurrProcRank - taskOffset;
            }
            uint32_t copyNum = DIV_ROUND_UP(taskCount, copyCount);

            for (uint32_t copyIdx = 0; copyIdx < copyNum; copyIdx++) {
                uint64_t copyOffset = copyIdx * copyCount;
                uint64_t gmOffset = outOffset + copyOffset;
                uint64_t currCopyCount = copyCount;
                if (copyOffset + currCopyCount > taskCount) {
                    currCopyCount = taskCount - copyOffset;
                }
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0 + curr);
                CopyGMtoUbuf(ubBuf[curr], outputBuf[processRankIdx][gmOffset], currCopyCount);
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0 + curr);
                CopyUbufToGM(outputBuf[myRankId][gmOffset], ubBuf[curr], currCopyCount);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0 + curr);
                curr = (curr + 1) % PINGPONG_BUF_NUM;
            }
        }

        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0 + i);
        }

        PipeBarrier<PIPE_ALL>();
    }

private:
    GlobalTensor<Dtype> inputBuf[XLITE_CCL_MAX_RANK_SIZE];
    GlobalTensor<Dtype> outputBuf[XLITE_CCL_MAX_RANK_SIZE];
    GlobalTensor<uint32_t> ipcFlagBuf[XLITE_CCL_MAX_RANK_SIZE];
    LocalTensor<uint32_t> flagBuf;
    LocalTensor<Dtype> ubBuf[PINGPONG_BUF_NUM];
    uint64_t count;
    uint64_t offsetCurrRank;
    uint64_t countCurrRank;
    uint64_t countPerRank;
    uint64_t countLastRank;
    uint64_t generation;
    uint32_t myRankId;
    uint32_t rankSize;
    uint32_t coreIdx;
    uint32_t coreNum;
    uint32_t blockNum;
    uint32_t rankIdxMapping[XLITE_CCL_MAX_RANK_SIZE];
    uint32_t syncWorkStart;
    uint32_t syncWorkEnd;
    uint32_t copyCount;
    struct XcclParam param;
    bool skipMyRank;
};

#if defined(XLITE_DEVICE_310P)
#define ALLREDUCE_FUNC_DEFINE(dtype)                                                              \
    extern "C" __global__ __aicore__ void allreduce_##dtype(                                     \
        GM_ADDR input, GM_ADDR output, uint64_t count, uint32_t rankId, uint32_t rankSize,        \
        uint64_t generation, GM_ADDR param, uint32_t copySize, bool fetchOffset)                  \
    {                                                                                             \
    }
#else
#define ALLREDUCE_FUNC_DEFINE(dtype)                                                               \
    extern "C" __global__ __aicore__ void allreduce_##dtype(                                       \
        GM_ADDR input, GM_ADDR output, uint64_t count, uint32_t rankId, uint32_t rankSize,         \
        uint64_t generation, GM_ADDR param, uint32_t copySize, bool fetchOffset)                   \
    {                                                                                              \
        if (count == 0 || rankSize <= 1 || copySize == 0) {                                        \
            return;                                                                                \
        }                                                                                          \
        AllReduce<dtype> op;                                                                       \
        op.Init(input, output, count, rankId, rankSize, generation, param, copySize, fetchOffset); \
        op.Run();                                                                                  \
    }
#endif

ALLREDUCE_FUNC_DEFINE(int8_t);
ALLREDUCE_FUNC_DEFINE(int16_t);
ALLREDUCE_FUNC_DEFINE(int32_t);
ALLREDUCE_FUNC_DEFINE(float16_t);
ALLREDUCE_FUNC_DEFINE(bfloat16_t);
ALLREDUCE_FUNC_DEFINE(float);
