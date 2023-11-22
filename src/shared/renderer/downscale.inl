#pragma once

#include <shared/core.inl>

#if DOWNSCALE_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(DownscaleComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(src_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(dst_image_id, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif

#if defined(__cplusplus)

struct DownscaleComputeTaskState {
    AsyncManagedComputePipeline pipeline;

    void compile_pipeline() {
    }

    DownscaleComputeTaskState(AsyncPipelineManager &pipeline_manager, std::vector<daxa::ShaderDefine> &&extra_defines) {
        extra_defines.push_back({"DOWNSCALE_COMPUTE", "1"});
        pipeline = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"downscale.comp.glsl"},
                .compile_options = {.defines = extra_defines, .enable_debug_info = true},
            },
            .name = "downscale",
        });
    }

    void record_commands(daxa::CommandRecorder &recorder, daxa_u32vec2 render_size) {
        if (!pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(pipeline.get());
        // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
        recorder.dispatch({(render_size.x + 7) / 8, (render_size.y + 7) / 8});
    }
};

struct DownscaleComputeTask : DownscaleComputeUses {
    DownscaleComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        recorder.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        auto const &image_info = ti.get_device().info_image(uses.dst_image_id.image()).value();
        state->record_commands(recorder, {image_info.size.x, image_info.size.y});
    }
};

inline auto extract_downscaled_depth(RecordContext &ctx, DownscaleComputeTaskState &task_state, daxa::TaskImageView depth) -> daxa::TaskImageView {
    auto size = ctx.render_resolution;

    auto output_tex = ctx.task_graph.create_transient_image({
        .format = daxa::Format::R32_SFLOAT,
        .size = {size.x / SHADING_SCL, size.y / SHADING_SCL, 1},
        .name = "downscaled_depth",
    });

    ctx.task_graph.add_task(DownscaleComputeTask{
        {
            .uses = {
                .gpu_input = ctx.task_input_buffer,
                .globals = ctx.task_globals_buffer,
                .src_image_id = depth,
                .dst_image_id = output_tex,
            },
        },
        &task_state,
    });

    return output_tex;
}

inline auto extract_downscaled_gbuffer_view_normal_rgba8(RecordContext &ctx, DownscaleComputeTaskState &task_state, daxa::TaskImageView gbuffer) -> daxa::TaskImageView {
    auto size = ctx.render_resolution;

    auto output_tex = ctx.task_graph.create_transient_image({
        .format = daxa::Format::R8G8B8A8_SNORM,
        .size = {size.x / SHADING_SCL, size.y / SHADING_SCL, 1},
        .name = "downscaled_gbuffer_view_normal",
    });

    ctx.task_graph.add_task(DownscaleComputeTask{
        {
            .uses = {
                .gpu_input = ctx.task_input_buffer,
                .globals = ctx.task_globals_buffer,
                .src_image_id = gbuffer,
                .dst_image_id = output_tex,
            },
        },
        &task_state,
    });

    return output_tex;
}

inline auto extract_downscaled_ssao(RecordContext &ctx, DownscaleComputeTaskState &task_state, daxa::TaskImageView ssao_image) -> daxa::TaskImageView {
    auto size = ctx.render_resolution;

    auto output_tex = ctx.task_graph.create_transient_image({
        .format = daxa::Format::R16_SFLOAT,
        .size = {size.x / SHADING_SCL, size.y / SHADING_SCL, 1},
        .name = "downscaled_ssao_image",
    });

    ctx.task_graph.add_task(DownscaleComputeTask{
        {
            .uses = {
                .gpu_input = ctx.task_input_buffer,
                .globals = ctx.task_globals_buffer,
                .src_image_id = ssao_image,
                .dst_image_id = output_tex,
            },
        },
        &task_state,
    });

    return output_tex;
}

#endif
