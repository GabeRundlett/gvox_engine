#pragma once

#include "voxels.inl"

#if defined(__cplusplus)

struct VoxelParticles;

struct VoxelWorld {
    VoxelWorldBuffers buffers;

    void record_startup(GpuContext &);
    void begin_frame(daxa::Device &, VoxelWorldOutput const &);
    void record_frame(GpuContext &, daxa::TaskBufferView, daxa::TaskImageView, VoxelParticles &);
};

#endif
