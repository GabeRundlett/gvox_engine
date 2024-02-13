#pragma once

#include "voxels.inl"

#if defined(__cplusplus)

struct VoxelWorld {
    VoxelWorldBuffers buffers;

    void create(daxa::Device &device) {
        buffers.voxel_globals = device.create_buffer({
            .size = static_cast<daxa_u32>(sizeof(daxa_u32)),
            .name = "voxel_globals",
        });

        buffers.task_voxel_globals.set_buffers({.buffers = std::array{buffers.voxel_globals}});
    }
    void destroy(daxa::Device &device) const {
        if (!buffers.voxel_globals.is_empty()) {
            device.destroy_buffer(buffers.voxel_globals);
        }
    }

    void for_each_buffer(auto func) {
        func(buffers.voxel_globals);
    }

    void record_startup(RecordContext &) {
    }

    void begin_frame(daxa::Device &, VoxelWorldOutput const &) {
    }

    void use_buffers(RecordContext &record_ctx) {
        record_ctx.task_graph.use_persistent_buffer(buffers.task_voxel_globals);
    }

    void record_frame(RecordContext &, daxa::TaskBufferView, daxa::TaskImageView) {
    }
};

#endif
