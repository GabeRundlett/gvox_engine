#pragma once

#include "voxels.inl"

#if defined(__cplusplus)

struct VoxelWorld {
    VoxelWorldBuffers buffers;

    void record_startup(RecordContext &);
    void begin_frame(daxa::Device &, VoxelWorldOutput const &);
    void record_frame(RecordContext &, daxa::TaskBufferView, daxa::TaskImageView);
};

#endif
