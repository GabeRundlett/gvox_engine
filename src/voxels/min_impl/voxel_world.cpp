#include "voxel_world.inl"

void VoxelWorld::record_startup(RecordContext &record_ctx) {
    record_ctx.task_graph.use_persistent_buffer(buffers.voxel_globals.task_resource);
}

void VoxelWorld::begin_frame(daxa::Device &, VoxelWorldOutput const &) {
}

void VoxelWorld::record_frame(RecordContext &record_ctx, daxa::TaskBufferView, daxa::TaskImageView) {
    buffers.voxel_globals = record_ctx.gpu_context->find_or_add_temporal_buffer({
        .size = static_cast<daxa_u32>(sizeof(VoxelWorldGlobals)),
        .name = "voxel_globals",
    });
    record_ctx.task_graph.use_persistent_buffer(buffers.voxel_globals.task_resource);
}
