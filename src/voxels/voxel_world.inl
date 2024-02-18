#pragma once

#include <voxels/impl/voxel_world.inl>

#if defined(__cplusplus)

template <typename T>
concept IsVoxelWorld = requires(T x, RecordContext &r) {
    { x.buffers };
    { x.record_startup(r) };
    { x.begin_frame(r.gpu_context->device, VoxelWorldOutput{}) };
    { x.use_buffers(r) };
    { x.record_frame(r, daxa::TaskBufferView{}, daxa::TaskImageView{}) };
};

#endif
