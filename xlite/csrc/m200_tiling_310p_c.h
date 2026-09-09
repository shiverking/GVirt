#ifndef _XLITE_M200_TILING_310P_C_H_
#define _XLITE_M200_TILING_310P_C_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int XliteM200BuildTiling310P(uint32_t m, uint32_t n, uint32_t k,
                             void *buffer, size_t capacity, size_t *written,
                             uint32_t *usedCores, uint64_t *systemWorkspaceBytes,
                             char *error, size_t errorCapacity);

#ifdef __cplusplus
}
#endif

#endif
