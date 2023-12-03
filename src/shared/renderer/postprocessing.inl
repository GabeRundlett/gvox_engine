#pragma once

#include <shared/core.inl>

#if COMPOSITING_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(CompositingCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), shadow_image_buffer)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, sky_lut)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, particles_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, ssao_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, dst_image_id)
DAXA_DECL_TASK_HEAD_END
struct CompositingComputePush {
    CompositingCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(CompositingComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_RWBufferPtr(daxa_u32) shadow_image_buffer = push.uses.shadow_image_buffer;
daxa_ImageViewId g_buffer_image_id = push.uses.g_buffer_image_id;
daxa_ImageViewId transmittance_lut = push.uses.transmittance_lut;
daxa_ImageViewId sky_lut = push.uses.sky_lut;
daxa_ImageViewId particles_image_id = push.uses.particles_image_id;
daxa_ImageViewId ssao_image_id = push.uses.ssao_image_id;
daxa_ImageViewId dst_image_id = push.uses.dst_image_id;
#endif
#endif

#if POSTPROCESSING_RASTER || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(PostprocessingRaster)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(FRAGMENT_SHADER_SAMPLED, REGULAR_2D, composited_image_id)
DAXA_TH_IMAGE_ID(FRAGMENT_SHADER_SAMPLED, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE_ID(COLOR_ATTACHMENT, REGULAR_2D, render_image)
DAXA_DECL_TASK_HEAD_END
struct PostprocessingRasterPush {
    PostprocessingRaster uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(PostprocessingRasterPush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewId composited_image_id = push.uses.composited_image_id;
daxa_ImageViewId g_buffer_image_id = push.uses.g_buffer_image_id;
daxa_ImageViewId render_image = push.uses.render_image;
#endif
#endif

#if defined(__cplusplus)

struct CompositingComputeTaskState {
    AsyncManagedComputePipeline pipeline;

    CompositingComputeTaskState(AsyncPipelineManager &pipeline_manager) {
        pipeline = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"postprocessing.comp.glsl"},
                .compile_options = {.defines = {{"COMPOSITING_COMPUTE", "1"}}},
            },
            .push_constant_size = sizeof(CompositingComputePush),
            .name = "compositing",
        });
    }

    void record_commands(CompositingComputePush const &push, daxa::CommandRecorder &recorder, daxa_u32vec2 render_size) {
        if (!pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(pipeline.get());
        recorder.push_constant(push);
        // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
        recorder.dispatch({(render_size.x + 7) / 8, (render_size.y + 7) / 8});
    }
};

struct PostprocessingRasterTaskState {
    AsyncManagedRasterPipeline pipeline;
    daxa::Format render_color_format;

    auto get_color_format() -> daxa::Format {
        return render_color_format;
    }

    PostprocessingRasterTaskState(AsyncPipelineManager &pipeline_manager, daxa::Format a_render_color_format = daxa::Format::R32G32B32A32_SFLOAT)
        : render_color_format{a_render_color_format} {
        pipeline = pipeline_manager.add_raster_pipeline({
            .vertex_shader_info = daxa::ShaderCompileInfo{.source = daxa::ShaderFile{"FULL_SCREEN_TRIANGLE_VERTEX_SHADER"}, .compile_options = {.defines = {{"POSTPROCESSING_RASTER", "1"}}}},
            .fragment_shader_info = daxa::ShaderCompileInfo{.source = daxa::ShaderFile{"postprocessing.comp.glsl"}, .compile_options = {.defines = {{"POSTPROCESSING_RASTER", "1"}}}},
            .color_attachments = {{
                .format = get_color_format(),
            }},
            .push_constant_size = sizeof(PostprocessingRasterPush),
            .name = "postprocessing",
        });
    }

    void record_commands(PostprocessingRasterPush const &push, daxa::CommandRecorder &recorder, daxa::ImageId render_image, daxa_u32vec2 size) {
        if (!pipeline.is_valid()) {
            return;
        }
        auto renderpass_recorder = std::move(recorder).begin_renderpass({
            .color_attachments = {{.image_view = render_image.default_view(), .load_op = daxa::AttachmentLoadOp::DONT_CARE, .clear_value = std::array<daxa_f32, 4>{0.0f, 0.0f, 0.0f, 0.0f}}},
            .render_area = {.x = 0, .y = 0, .width = size.x, .height = size.y},
        });
        renderpass_recorder.set_pipeline(pipeline.get());
        renderpass_recorder.push_constant(push);
        renderpass_recorder.draw({.vertex_count = 3});
        recorder = std::move(renderpass_recorder).end_renderpass();
    }
};

struct CompositingComputeTask {
    CompositingCompute::Uses uses;
    CompositingComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto const &image_info = ti.get_device().info_image(uses.dst_image_id.image()).value();
        auto push = CompositingComputePush{};
        ti.copy_task_head_to(&push.uses);
        state->record_commands(push, recorder, {image_info.size.x, image_info.size.y});
    }
};

struct PostprocessingRasterTask {
    PostprocessingRaster::Uses uses;
    PostprocessingRasterTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto const &image_info = ti.get_device().info_image(uses.render_image.image()).value();
        auto push = PostprocessingRasterPush{};
        ti.copy_task_head_to(&push.uses);
        state->record_commands(push, recorder, uses.render_image.image(), {image_info.size.x, image_info.size.y});
    }
};

struct Compositor {
    CompositingComputeTaskState compositing_compute_task_state;
    Compositor(AsyncPipelineManager &pipeline_manager)
        : compositing_compute_task_state{pipeline_manager} {
    }

    auto render(RecordContext &record_ctx, GbufferDepth &gbuffer_depth, daxa::TaskImageView sky_lut, daxa::TaskImageView transmittance_lut, daxa::TaskImageView ssao_image, daxa::TaskBufferView shadow_image_buffer, daxa::TaskImageView particles_image) -> daxa::TaskImageView {
        auto output_image = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "composited_image",
        });

        record_ctx.task_graph.add_task(CompositingComputeTask{
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                .shadow_image_buffer = shadow_image_buffer,
                .g_buffer_image_id = gbuffer_depth.gbuffer,
                .transmittance_lut = transmittance_lut,
                .sky_lut = sky_lut,
                .particles_image_id = particles_image,
                .ssao_image_id = ssao_image,
                .dst_image_id = output_image,
            },
            .state = &compositing_compute_task_state,
        });

        return output_image;
    }
};

#endif
