#pragma once

#include <shared/voxels/impl/voxels.inl>

#if !defined(VOXELS_BUFFER_PTRS)
#error "The implementation must define a way for the users to construct the read-only pointers!"
#endif

#if !defined(VOXELS_RW_BUFFER_PTRS)
#error "The implementation must define a way for the users to construct the read-write pointers!"
#endif

#if defined(__cplusplus)

template <typename T>
concept IsVoxelWorld = requires(T x, AsyncPipelineManager &p, RecordContext &r, bool b) {
    { T(p) };
    { x.buffers } -> std::same_as<typename T::Buffers &>;
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
