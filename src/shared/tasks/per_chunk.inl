#pragma once

#include "../core.inl"

DAXA_INL_TASK_USE_BEGIN(PerChunkComputeUses, DAXA_CBUFFER_SLOT0)
DAXA_INL_TASK_USE_BUFFER(settings, daxa_BufferPtr(GpuSettings), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(gvox_model, daxa_BufferPtr(GpuGvoxModel), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ_WRITE)
DAXA_INL_TASK_USE_BUFFER(voxel_chunks, daxa_RWBufferPtr(VoxelLeafChunk), COMPUTE_SHADER_READ_WRITE)
DAXA_INL_TASK_USE_END()

#if defined(__cplusplus)

struct PerChunkComputeTaskState {
    daxa::PipelineManager &pipeline_manager;
    AppUi &ui;
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    void compile_pipeline() {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"per_chunk.comp.glsl"},
                .compile_options = {.defines = {{"PER_CHUNK_COMPUTE", "1"}}},
            },
            .name = "per_chunk",
        });
        if (compile_result.is_err()) {
            ui.console.add_log(compile_result.to_string());
            return;
        }
        pipeline = compile_result.value();
    }

    PerChunkComputeTaskState(daxa::PipelineManager &a_pipeline_manager, AppUi &a_ui) : pipeline_manager{a_pipeline_manager}, ui{a_ui} {}

    void record_commands(daxa::CommandList &cmd_list) {
        if (!pipeline) {
            compile_pipeline();
            if (!pipeline)
                return;
        }
        cmd_list.set_pipeline(*pipeline);
        auto chunk_nx = 1u << ui.settings.log2_chunks_per_axis;
        auto chunk_n = u32vec3{chunk_nx, chunk_nx, chunk_nx};
        cmd_list.dispatch((chunk_n.x + 7) / 8, (chunk_n.y + 7) / 8, (chunk_n.z + 7) / 8);
    }
};

struct PerChunkComputeTask : PerChunkComputeUses {
    PerChunkComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_constant_buffer(ti.uses.constant_buffer_set_info());
        state->record_commands(cmd_list);
    }
};

#endif
