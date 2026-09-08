#pragma once

// Shared host/device ABI. Each partial has max, sum, six padding floats,
// and 128 unnormalised output floats; every record is DMA aligned.
namespace XlitePaged310P {
constexpr unsigned MaxBatch = 20;
constexpr unsigned Heads = 16;
constexpr unsigned MaxBlocks = 16;
constexpr unsigned RecordWords = 4;
constexpr unsigned TableOffset = MaxBatch * RecordWords;
constexpr unsigned MetadataBytes = (TableOffset + MaxBatch * MaxBlocks) * 4;
constexpr unsigned PartialFloats = 136;
constexpr unsigned ScratchBytes = MaxBatch * Heads * 4 * PartialFloats * 4;
constexpr unsigned ReserveBytes = 1024 * 1024;
static_assert(ScratchBytes + MetadataBytes <= ReserveBytes, "paged decode reserve");
}
