#pragma once

#include "../core.inl"

DAXA_DECL_TASK_USES_BEGIN(PerframeComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(settings, daxa_BufferPtr(GpuSettings), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(gpu_output, daxa_RWBufferPtr(GpuOutput), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_BUFFER(simulated_voxel_particles, daxa_RWBufferPtr(SimulatedVoxelParticle), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_BUFFER(voxel_malloc_global_allocator, daxa_RWBufferPtr(VoxelMalloc_GlobalAllocator), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_BUFFER(voxel_chunks, daxa_BufferPtr(VoxelLeafChunk), COMPUTE_SHADER_READ)
DAXA_DECL_TASK_USES_END()

#if defined(__cplusplus)

struct PerframeComputeTaskState {
    daxa::PipelineManager &pipeline_manager;
    AppUi &ui;
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    void compile_pipeline() {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"perframe.comp.glsl"},
                .compile_options = {.defines = {{"PERFRAME_COMPUTE", "1"}}},
            },
            .name = "perframe",
        });
        if (compile_result.is_err()) {
            ui.console.add_log(compile_result.to_string());
            return;
        }
        pipeline = compile_result.value();
    }

    PerframeComputeTaskState(daxa::PipelineManager &a_pipeline_manager, AppUi &a_ui) : pipeline_manager{a_pipeline_manager}, ui{a_ui} {}

    void record_commands(daxa::CommandList &cmd_list) {
        if (!pipeline) {
            compile_pipeline();
            if (!pipeline)
                return;
        }
        cmd_list.set_pipeline(*pipeline);
        cmd_list.dispatch(1, 1, 1);
    }
};

struct PerframeComputeTask : PerframeComputeUses {
    PerframeComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        state->record_commands(cmd_list);
    }
};

#endif
