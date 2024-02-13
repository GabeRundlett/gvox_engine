#pragma once

#include <voxels/impl/voxel_world.inl>

#if defined(__cplusplus)

template <typename T>
concept IsVoxelWorld = requires(T x, RecordContext &r) {
    { x.buffers };
    { x.create(r.device) };
    { x.destroy(r.device) };
    {
        x.for_each_buffer([](daxa::BufferId) {})
    };
    { x.record_startup(r) };
    { x.begin_frame(r.device, VoxelWorldOutput{}) };
    { x.use_buffers(r) };
    { x.record_frame(r, daxa::TaskBufferView{}, daxa::TaskImageView{}) };
};

#endif
