#include "voxel_world.inl"

void VoxelWorld::record_startup(RecordContext &) {
}

void VoxelWorld::begin_frame(daxa::Device &, VoxelWorldOutput const &) {
}

void VoxelWorld::use_buffers(RecordContext &record_ctx) {
    buffers.voxel_globals = record_ctx.gpu_context->find_or_add_temporal_buffer({
        .size = static_cast<daxa_u32>(sizeof(VoxelWorldGlobals)),
        .name = "voxel_globals",
    });

    record_ctx.task_graph.use_persistent_buffer(buffers.voxel_globals.task_resource);
}

void VoxelWorld::record_frame(RecordContext &, daxa::TaskBufferView, daxa::TaskImageView) {
}
