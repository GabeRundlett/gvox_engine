#pragma once

#if defined(__cplusplus)

struct VoxelWorld : AppUi::DebugDisplayProvider {
    struct Buffers {
        daxa::BufferId voxel_globals;
        daxa::TaskBuffer task_voxel_globals{{.name = "task_voxel_globals"}};
    };

    Buffers buffers;

    VoxelWorld(AsyncPipelineManager &) {
    }
    virtual ~VoxelWorld() override = default;

    virtual void add_ui() override {
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

    void startup(RecordContext &) {
    }

    auto check_for_realloc(daxa::Device &, VoxelWorldOutput const &) -> bool {
        return false;
    }

    void dynamic_buffers_realloc(daxa::TaskGraph &, bool &) {
    }

    void use_buffers(RecordContext &record_ctx) {
        record_ctx.task_graph.use_persistent_buffer(buffers.task_voxel_globals);
    }

    void update(RecordContext &, daxa::TaskBufferView, daxa::TaskImageView) {
    }
};

#endif
