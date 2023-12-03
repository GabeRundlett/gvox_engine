#pragma once

#include <shared/core.inl>

#if DOWNSCALE_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(DownscaleCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, src_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, dst_image_id)
DAXA_DECL_TASK_HEAD_END
struct DownscaleComputePush {
    DownscaleCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(DownscaleComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewId src_image_id = push.uses.src_image_id;
daxa_ImageViewId dst_image_id = push.uses.dst_image_id;
#endif
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
            .push_constant_size = sizeof(DownscaleComputePush),
            .name = "downscale",
        });
    }

    void record_commands(DownscaleComputePush const &push, daxa::CommandRecorder &recorder, daxa_u32vec2 render_size) {
        if (!pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(pipeline.get());
        recorder.push_constant(push);
        // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
        recorder.dispatch({(render_size.x + 7) / 8, (render_size.y + 7) / 8});
    }
};

struct DownscaleComputeTask {
    DownscaleCompute::Uses uses;
    DownscaleComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto const &image_info = ti.get_device().info_image(uses.dst_image_id.image()).value();
        auto push = DownscaleComputePush{};
        ti.copy_task_head_to(&push.uses);
        state->record_commands(push, recorder, {image_info.size.x, image_info.size.y});
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
        .uses = {
            .gpu_input = ctx.task_input_buffer,
            .globals = ctx.task_globals_buffer,
            .src_image_id = depth,
            .dst_image_id = output_tex,
        },
        .state = &task_state,
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
        .uses = {
            .gpu_input = ctx.task_input_buffer,
            .globals = ctx.task_globals_buffer,
            .src_image_id = gbuffer,
            .dst_image_id = output_tex,
        },
        .state = &task_state,
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
            .uses = {
                .gpu_input = ctx.task_input_buffer,
                .globals = ctx.task_globals_buffer,
                .src_image_id = ssao_image,
                .dst_image_id = output_tex,
            },
        .state = &task_state,
    });

    return output_tex;
}

#endif
