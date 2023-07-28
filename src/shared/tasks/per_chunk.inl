#pragma once

#include <shared/core.inl>

DAXA_DECL_TASK_USES_BEGIN(PerChunkComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(settings, daxa_BufferPtr(GpuSettings), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(gvox_model, daxa_BufferPtr(GpuGvoxModel), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_BUFFER(voxel_chunks, daxa_RWBufferPtr(VoxelLeafChunk), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_IMAGE(value_noise_texture, REGULAR_2D_ARRAY, COMPUTE_SHADER_SAMPLED)
DAXA_DECL_TASK_USES_END()

struct PerChunkComputePush {
    daxa_SamplerId value_noise_sampler;
};

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
            ui.console.add_log(compile_result.message());
            return;
        }
        pipeline = compile_result.value();
        if (!compile_result.value()->is_valid()) {
            ui.console.add_log(compile_result.message());
        }
    }

    PerChunkComputeTaskState(daxa::PipelineManager &a_pipeline_manager, AppUi &a_ui) : pipeline_manager{a_pipeline_manager}, ui{a_ui} { compile_pipeline(); }
    auto pipeline_is_valid() -> bool { return pipeline && pipeline->is_valid(); }

    void record_commands(daxa::CommandList &cmd_list, daxa_SamplerId value_noise_sampler) {
        if (!pipeline_is_valid()) {
            return;
        }
        cmd_list.set_pipeline(*pipeline);
        cmd_list.push_constant(PerChunkComputePush{
            .value_noise_sampler = value_noise_sampler,
        });
        auto const dispatch_size = 1 << (ui.settings.log2_chunks_per_axis - 3);
        cmd_list.dispatch(dispatch_size, dispatch_size, dispatch_size);
    }
};

struct PerChunkComputeTask : PerChunkComputeUses {
    PerChunkComputeTaskState *state;
    daxa_SamplerId *value_noise_sampler;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        state->record_commands(cmd_list, *value_noise_sampler);
    }
};

#endif
