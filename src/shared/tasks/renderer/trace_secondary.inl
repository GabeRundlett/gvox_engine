#pragma once

#include <shared/core.inl>

DAXA_DECL_TASK_USES_BEGIN(TraceSecondaryComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(settings, daxa_BufferPtr(GpuSettings), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(voxel_malloc_page_allocator, daxa_RWBufferPtr(VoxelMallocPageAllocator), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(voxel_chunks, daxa_BufferPtr(VoxelLeafChunk), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(blue_noise_vec2, REGULAR_3D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(g_buffer_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(depth_image, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(indirect_diffuse_image_id, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()

#if defined(__cplusplus)

struct TraceSecondaryComputeTaskState {
    daxa::PipelineManager &pipeline_manager;
    AppUi &ui;
    u32vec2 &render_size;
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    void compile_pipeline() {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"trace_secondary.comp.glsl"},
                .compile_options = {.defines = {{"TRACE_SECONDARY_COMPUTE", "1"}}},
            },
            .name = "trace_secondary",
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

    TraceSecondaryComputeTaskState(daxa::PipelineManager &a_pipeline_manager, AppUi &a_ui, u32vec2 &a_render_size) : pipeline_manager{a_pipeline_manager}, ui{a_ui}, render_size{a_render_size} { compile_pipeline(); }
    auto pipeline_is_valid() -> bool { return pipeline && pipeline->is_valid(); }

    void record_commands(daxa::CommandList &cmd_list) {
        if (!pipeline_is_valid()) {
            return;
        }
        cmd_list.set_pipeline(*pipeline);
        constexpr auto SCL_X = 8 * SHADING_SCL;
        constexpr auto SCL_Y = 8 * SHADING_SCL;
        assert((render_size.x % SCL_X) == 0 && (render_size.y % SCL_Y) == 0);
        cmd_list.dispatch(render_size.x / SCL_X, render_size.y / SCL_Y);
    }
};

struct TraceSecondaryComputeTask : TraceSecondaryComputeUses {
    TraceSecondaryComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        state->record_commands(cmd_list);
    }
};

#endif
