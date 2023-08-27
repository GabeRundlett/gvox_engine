#pragma once

#if defined(__cplusplus)

struct VoxelWorld {
    struct Buffers {
        daxa::BufferId voxel_globals;
        daxa::TaskBuffer task_voxel_globals{{.name = "task_voxel_globals"}};
    };

    Buffers buffers;

    VoxelWorld(AsyncPipelineManager &pipeline_manager) {
    }

    void create(daxa::Device &device) {
        buffers.voxel_globals = device.create_buffer({
            .size = static_cast<u32>(sizeof(u32)),
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

    void startup(RecordContext &record_ctx) {
    }

    auto check_for_realloc(daxa::Device &device, VoxelWorldOutput const &gpu_output) -> bool {
        return false;
    }

    void dynamic_buffers_realloc(daxa::TaskGraph &temp_task_graph, bool &needs_vram_calc) {
    }

    void use_buffers(RecordContext &record_ctx) {
        record_ctx.task_graph.use_persistent_buffer(buffers.task_voxel_globals);
    }

    void update(RecordContext &record_ctx, daxa::TaskBufferView task_gvox_model_buffer, daxa::TaskImageView task_value_noise_image) {
    }
};

#endif
