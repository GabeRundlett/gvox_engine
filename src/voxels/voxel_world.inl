#pragma once

#include <voxels/impl/voxel_world.inl>

#if defined(__cplusplus)

template <typename T>
concept IsVoxelWorld = requires(T x, GpuContext &g, VoxelParticles &p) {
    { x.buffers };
    { x.record_startup(g) };
    { x.begin_frame(g.device, VoxelWorldOutput{}) };
    { x.record_frame(g, daxa::TaskBufferView{}, daxa::TaskImageView{}, p) };
};

#endif
