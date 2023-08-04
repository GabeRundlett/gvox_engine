#pragma once

#include <shared/core.inl>

#if PER_CHUNK_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(PerChunkComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(gvox_model, daxa_BufferPtr(GpuGvoxModel), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_BUFFER(voxel_chunks, daxa_RWBufferPtr(VoxelLeafChunk), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_IMAGE(value_noise_texture, REGULAR_2D_ARRAY, COMPUTE_SHADER_SAMPLED)
DAXA_DECL_TASK_USES_END()
#endif

#if defined(__cplusplus)

struct PerChunkComputeTaskState {
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    PerChunkComputeTaskState(daxa::PipelineManager &pipeline_manager) {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"per_chunk.comp.glsl"},
                .compile_options = {.defines = {{"PER_CHUNK_COMPUTE", "1"}}},
            },
            .name = "per_chunk",
        });
        if (compile_result.is_err()) {
            AppUi::Console::s_instance->add_log(compile_result.message());
            return;
        }
        pipeline = compile_result.value();
        if (!compile_result.value()->is_valid()) {
            AppUi::Console::s_instance->add_log(compile_result.message());
        }
    }
    auto pipeline_is_valid() -> bool { return pipeline && pipeline->is_valid(); }

    void record_commands(daxa::CommandList &cmd_list) {
        if (!pipeline_is_valid()) {
            return;
        }
        cmd_list.set_pipeline(*pipeline);
        auto const dispatch_size = 1 << (LOG2_CHUNKS_PER_LEVEL_PER_AXIS - 3);
        cmd_list.dispatch(dispatch_size, dispatch_size, dispatch_size);
    }
};

struct PerChunkComputeTask : PerChunkComputeUses {
    PerChunkComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        state->record_commands(cmd_list);
    }
};

#endif
