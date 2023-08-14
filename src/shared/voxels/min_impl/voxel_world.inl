#pragma once

#if defined(__cplusplus)

struct VoxelWorld {
    struct Buffers {
        daxa::BufferId _dummy;
        daxa::TaskBuffer task_dummy{{.name = "task_dummy"}};
    };

    Buffers buffers;

    VoxelWorld(daxa::PipelineManager &pipeline_manager) {
    }

    void create(daxa::Device &device) {
        buffers._dummy = device.create_buffer({
            .size = static_cast<u32>(sizeof(u32)),
            .name = "_dummy",
        });

        buffers.task_dummy.set_buffers({.buffers = std::array{buffers._dummy}});
    }
    void destroy(daxa::Device &device) const {
        if (!buffers._dummy.is_empty()) {
            device.destroy_buffer(buffers._dummy);
        }
    }

    void for_each_buffer(auto func) {
        func(buffers._dummy);
    }

    void startup(RecordContext &record_ctx) {
    }

    auto check_for_realloc(daxa::Device &device, VoxelWorldOutput const &gpu_output) -> bool {
        return false;
    }

    void dynamic_buffers_realloc(daxa::TaskGraph &temp_task_graph, bool &needs_vram_calc) {
    }

    void use_buffers(RecordContext &record_ctx) {
        record_ctx.task_graph.use_persistent_buffer(buffers.task_dummy);
    }

    void update(RecordContext &record_ctx, daxa::TaskBufferView task_gvox_model_buffer, daxa::TaskImageView task_value_noise_image) {
    }
};

#endif
