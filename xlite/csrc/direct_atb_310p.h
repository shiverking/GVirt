/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_DIRECT_ATB_310P_H_
#define _XLITE_DIRECT_ATB_310P_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

typedef void *aclrtStream;

class XliteDirectAtb310P
{
public:
    using WorkspaceAcquire = std::function<void *(size_t)>;
    using WorkspaceRelease = std::function<void(void *)>;

    XliteDirectAtb310P();
    ~XliteDirectAtb310P();
    XliteDirectAtb310P(const XliteDirectAtb310P &) = delete;
    XliteDirectAtb310P &operator=(const XliteDirectAtb310P &) = delete;

    void SetStream(aclrtStream stream);
    void SetSetupReuse(bool enabled);
    bool ReshapeAndCache(uint32_t layer, void *key, void *value, uint32_t tokens,
                         void *keyCache, void *valueCache, uint32_t cacheBlocks, void *slots,
                         const WorkspaceAcquire &acquire, const WorkspaceRelease &release);
    bool PagedAttention(uint32_t layer, void *query, uint32_t batch, void *keyCache,
                        void *valueCache, uint32_t cacheBlocks, void *blockTables,
                        uint32_t tableColumns, void *contextLens, void *output,
                        const WorkspaceAcquire &acquire, const WorkspaceRelease &release);

    [[nodiscard]] uint64_t SetupCount() const;
    [[nodiscard]] uint64_t SetupReuseCount() const;
    [[nodiscard]] uint64_t ExecuteCount() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
