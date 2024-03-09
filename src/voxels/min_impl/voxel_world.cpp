#include "voxel_world.inl"

void VoxelWorld::record_startup(GpuContext &gpu_context) {
    gpu_context.frame_task_graph.use_persistent_buffer(buffers.voxel_globals.task_resource);
}

void VoxelWorld::begin_frame(daxa::Device &, VoxelWorldOutput const &) {
}

void VoxelWorld::record_frame(GpuContext &gpu_context, daxa::TaskBufferView, daxa::TaskImageView) {
    buffers.voxel_globals = gpu_context.find_or_add_temporal_buffer({
        .size = sizeof(VoxelWorldGlobals),
        .name = "voxel_globals",
    });
    gpu_context.frame_task_graph.use_persistent_buffer(buffers.voxel_globals.task_resource);
}
