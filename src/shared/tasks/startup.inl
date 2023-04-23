#pragma once

DAXA_INL_TASK_USE_BEGIN(StartupComputeUses, DAXA_CBUFFER_SLOT0)
DAXA_INL_TASK_USE_BUFFER(settings, daxa_BufferPtr(GpuSettings), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ_WRITE)
DAXA_INL_TASK_USE_BUFFER(voxel_chunks, daxa_RWBufferPtr(VoxelChunk), COMPUTE_SHADER_READ_WRITE)
DAXA_INL_TASK_USE_END()

#if defined(__cplusplus)

struct StartupComputeTaskState {
    daxa::PipelineManager &pipeline_manager;
    AppUi &ui;
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    void compile_pipeline() {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"startup.comp.glsl"},
                .compile_options = {.defines = {{"STARTUP_COMPUTE", "1"}}},
            },
            .name = "startup_task",
        });
        if (compile_result.is_err()) {
            ui.console.add_log(compile_result.to_string());
            return;
        }
        pipeline = compile_result.value();
    }

    StartupComputeTaskState(daxa::PipelineManager &a_pipeline_manager, AppUi &a_ui) : pipeline_manager{a_pipeline_manager}, ui{a_ui} {}

    void record_commands(daxa::CommandList &cmd_list) {
        if (!pipeline) {
            compile_pipeline();
            if (!pipeline)
                return;
        }
        cmd_list.set_pipeline(*pipeline);
        cmd_list.dispatch(1, 1, 1);
        ui.should_run_startup = false;
    }
};

struct StartupComputeTask : StartupComputeUses {
    StartupComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        state->record_commands(cmd_list);
    }
};

#endif
