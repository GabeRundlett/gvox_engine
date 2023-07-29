#pragma once

#include <shared/core.inl>

DAXA_DECL_TASK_USES_BEGIN(ChunkHierarchyComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(gvox_model, daxa_BufferPtr(GpuGvoxModel), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_BUFFER(voxel_chunks, daxa_RWBufferPtr(VoxelLeafChunk), COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_USES_END()

#if defined(__cplusplus)

struct ChunkHierarchyComputeTaskState {
    daxa::PipelineManager &pipeline_manager;
    AppUi &ui;
    std::shared_ptr<daxa::ComputePipeline> pipeline_l0;
    std::shared_ptr<daxa::ComputePipeline> pipeline_l1;

    void compile_pipeline(std::shared_ptr<daxa::ComputePipeline> &pipeline, char const *const depth_str) {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"chunk_hierarchy.comp.glsl"},
                .compile_options = {.defines = {{"CHUNK_HIERARCHY_COMPUTE", "1"}, {"CHUNK_LEVEL", depth_str}}},
            },
            .name = "chunk_hierarchy",
        });
        if (compile_result.is_err()) {
            ui.console.add_log(compile_result.message());
            return;
        }
        pipeline = compile_result.value();
        if (!compile_result.value()->is_valid()) {
            ui.console.add_log(compile_result.message());
        }
    }

    ChunkHierarchyComputeTaskState(daxa::PipelineManager &a_pipeline_manager, AppUi &a_ui) : pipeline_manager{a_pipeline_manager}, ui{a_ui} {
        compile_pipeline(pipeline_l0, "0");
        compile_pipeline(pipeline_l1, "1");
    }
    auto pipeline_l0_is_valid() -> bool { return pipeline_l0 && pipeline_l0->is_valid(); }
    auto pipeline_l1_is_valid() -> bool { return pipeline_l1 && pipeline_l1->is_valid(); }

    void record_commands_l0(daxa::CommandList &cmd_list, daxa::BufferId globals_buffer_id) {
        if (!pipeline_l0_is_valid()) {
            return;
        }
        cmd_list.set_pipeline(*pipeline_l0);
        cmd_list.dispatch_indirect({
            .indirect_buffer = globals_buffer_id,
            .offset = offsetof(GpuGlobals, chunk_thread_pool_state) + offsetof(ChunkThreadPoolState, work_items_l0_queued),
        });
    }
    void record_commands_l1(daxa::CommandList &cmd_list, daxa::BufferId globals_buffer_id) {
        if (!pipeline_l1_is_valid()) {
            return;
        }
        cmd_list.set_pipeline(*pipeline_l1);
        cmd_list.dispatch_indirect({
            .indirect_buffer = globals_buffer_id,
            .offset = offsetof(GpuGlobals, chunk_thread_pool_state) + offsetof(ChunkThreadPoolState, work_items_l1_queued),
        });
    }
};

struct ChunkHierarchyComputeTaskL0 : ChunkHierarchyComputeUses {
    ChunkHierarchyComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        state->record_commands_l0(cmd_list, uses.globals.buffer());
    }
};

struct ChunkHierarchyComputeTaskL1 : ChunkHierarchyComputeUses {
    ChunkHierarchyComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        state->record_commands_l1(cmd_list, uses.globals.buffer());
    }
};

#endif
