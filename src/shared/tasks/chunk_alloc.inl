#pragma once

#include "../core.inl"

DAXA_INL_TASK_USE_BEGIN(ChunkAllocComputeUses, DAXA_CBUFFER_SLOT0)
DAXA_INL_TASK_USE_BUFFER(settings, daxa_BufferPtr(GpuSettings), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(globals, daxa_BufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(temp_voxel_chunks, daxa_BufferPtr(TempVoxelChunk), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(voxel_chunks, daxa_RWBufferPtr(VoxelChunk), COMPUTE_SHADER_READ_WRITE)
DAXA_INL_TASK_USE_BUFFER(voxel_malloc_global_allocator, daxa_RWBufferPtr(VoxelMalloc_GlobalAllocator), COMPUTE_SHADER_READ_WRITE)
DAXA_INL_TASK_USE_END()

#if defined(__cplusplus)

struct ChunkAllocComputeTaskState {
    daxa::PipelineManager &pipeline_manager;
    AppUi &ui;
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    void compile_pipeline() {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"chunk_alloc.comp.glsl"},
                .compile_options = {.defines = {{"CHUNK_ALLOC_COMPUTE", "1"}}},
            },
            .name = "chunk_alloc",
        });
        if (compile_result.is_err()) {
            ui.console.add_log(compile_result.to_string());
            return;
        }
        pipeline = compile_result.value();
    }

    ChunkAllocComputeTaskState(daxa::PipelineManager &a_pipeline_manager, AppUi &a_ui) : pipeline_manager{a_pipeline_manager}, ui{a_ui} {}

    void record_commands(daxa::CommandList &cmd_list, daxa::BufferId globals_buffer_id) {
        if (!pipeline) {
            compile_pipeline();
            if (!pipeline)
                return;
        }
        cmd_list.set_pipeline(*pipeline);
        cmd_list.dispatch_indirect({
            .indirect_buffer = globals_buffer_id,
            // NOTE: This should always have the same value as the chunk edit dispatch, so we're re-using it here
            .offset = offsetof(GpuGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, chunk_edit_dispatch),
        });
    }
};

struct ChunkAllocComputeTask : ChunkAllocComputeUses {
    ChunkAllocComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_constant_buffer(ti.uses.constant_buffer_set_info());
        state->record_commands(cmd_list, uses.globals.buffer());
    }
};

#endif
