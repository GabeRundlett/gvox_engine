#pragma once

#include "../core.inl"

DAXA_INL_TASK_USE_BEGIN(ChunkHierarchyComputeUses, DAXA_CBUFFER_SLOT0)
DAXA_INL_TASK_USE_BUFFER(settings, daxa_BufferPtr(GpuSettings), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(gvox_model, daxa_BufferPtr(GpuGvoxModel), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ_WRITE)
DAXA_INL_TASK_USE_BUFFER(voxel_chunks, daxa_RWBufferPtr(VoxelLeafChunk), COMPUTE_SHADER_READ_WRITE)
DAXA_INL_TASK_USE_END()

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
            ui.console.add_log(compile_result.to_string());
            return;
        }
        pipeline = compile_result.value();
    }

    ChunkHierarchyComputeTaskState(daxa::PipelineManager &a_pipeline_manager, AppUi &a_ui) : pipeline_manager{a_pipeline_manager}, ui{a_ui} {}

    void record_commands_l0(daxa::CommandList &cmd_list, daxa::BufferId globals_buffer_id) {
        if (!pipeline_l0) {
            compile_pipeline(pipeline_l0, "0");
            if (!pipeline_l0)
                return;
        }
        cmd_list.set_pipeline(*pipeline_l0);
        cmd_list.dispatch_indirect({
            .indirect_buffer = globals_buffer_id,
            .offset = offsetof(GpuGlobals, chunk_thread_pool_state) + offsetof(ChunkThreadPoolState, work_items_l0_queued),
        });
    }
    void record_commands_l1(daxa::CommandList &cmd_list, daxa::BufferId globals_buffer_id) {
        if (!pipeline_l1) {
            compile_pipeline(pipeline_l1, "1");
            if (!pipeline_l1)
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
        cmd_list.set_constant_buffer(ti.uses.constant_buffer_set_info());
        state->record_commands_l0(cmd_list, uses.globals.buffer());
    }
};

struct ChunkHierarchyComputeTaskL1 : ChunkHierarchyComputeUses {
    ChunkHierarchyComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_constant_buffer(ti.uses.constant_buffer_set_info());
        state->record_commands_l1(cmd_list, uses.globals.buffer());
    }
};

#endif
