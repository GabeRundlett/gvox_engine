#pragma once

#include <voxels/impl/voxel_world.inl>

#if defined(__cplusplus)

template <typename T>
concept IsVoxelWorld = requires(T x, RecordContext &r, bool b) {
    { x.buffers };
    { x.create(r.device) };
    { x.destroy(r.device) };
    {
        x.for_each_buffer([](daxa::BufferId) {})
    };
    { x.record_startup(r) };
    { x.check_for_realloc(r.device, VoxelWorldOutput{}) } -> std::same_as<bool>;
    { x.dynamic_buffers_realloc(r.task_graph, b) };
    { x.use_buffers(r) };
    { x.record_frame(r, daxa::TaskBufferView{}, daxa::TaskImageView{}) };
};

#endif
