#pragma once

#include <shared/core.inl>

#if COMPOSITING_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(CompositingComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(g_buffer_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(sky_lut, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
// DAXA_TASK_USE_IMAGE(particles_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(ssao_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(shading_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(dst_image_id, REGULAR_2D, COMPUTE_SHADER_STORAGE_READ_WRITE)
DAXA_DECL_TASK_USES_END()
#endif

#if POSTPROCESSING_RASTER || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(PostprocessingRasterUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), FRAGMENT_SHADER_READ)
DAXA_TASK_USE_IMAGE(composited_image_id, REGULAR_2D, FRAGMENT_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(render_image, REGULAR_2D, COLOR_ATTACHMENT)
DAXA_DECL_TASK_USES_END()
#endif

struct PostprocessingRasterPush {
    u32vec2 final_size;
};

#if defined(__cplusplus)

struct CompositingComputeTaskState {
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    CompositingComputeTaskState(AsyncPipelineManager &pipeline_manager) {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"postprocessing.comp.glsl"},
                .compile_options = {.defines = {{"COMPOSITING_COMPUTE", "1"}}},
            },
            .name = "compositing",
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

    void record_commands(daxa::CommandList &cmd_list, u32vec2 render_size) {
        if (!pipeline_is_valid()) {
            return;
        }
        cmd_list.set_pipeline(*pipeline);
        // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
        cmd_list.dispatch((render_size.x + 7) / 8, (render_size.y + 7) / 8);
    }
};

struct PostprocessingRasterTaskState {
    std::shared_ptr<daxa::RasterPipeline> pipeline;
    daxa::Format render_color_format;

    auto get_color_format() -> daxa::Format {
        return render_color_format;
    }

    PostprocessingRasterTaskState(AsyncPipelineManager &pipeline_manager, daxa::Format a_render_color_format = daxa::Format::R32G32B32A32_SFLOAT)
        : render_color_format{a_render_color_format} {
        auto compile_result = pipeline_manager.add_raster_pipeline({
            .vertex_shader_info = daxa::ShaderCompileInfo{.source = daxa::ShaderFile{"FULL_SCREEN_TRIANGLE_VERTEX_SHADER"}, .compile_options = {.defines = {{"POSTPROCESSING_RASTER", "1"}}}},
            .fragment_shader_info = daxa::ShaderCompileInfo{.source = daxa::ShaderFile{"postprocessing.comp.glsl"}, .compile_options = {.defines = {{"POSTPROCESSING_RASTER", "1"}}}},
            .color_attachments = {{
                .format = get_color_format(),
            }},
            .push_constant_size = sizeof(PostprocessingRasterPush),
            .name = "postprocessing",
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

    void record_commands(daxa::CommandList &cmd_list, daxa::ImageId render_image, u32vec2 size) {
        if (!pipeline_is_valid()) {
            return;
        }
        cmd_list.begin_renderpass({
            .color_attachments = {{.image_view = render_image.default_view(), .load_op = daxa::AttachmentLoadOp::DONT_CARE, .clear_value = std::array<f32, 4>{0.0f, 0.0f, 0.0f, 0.0f}}},
            .render_area = {.x = 0, .y = 0, .width = size.x, .height = size.y},
        });
        cmd_list.set_pipeline(*pipeline);
        cmd_list.push_constant(PostprocessingRasterPush{
            .final_size = size,
        });
        cmd_list.draw({.vertex_count = 3});
        cmd_list.end_renderpass();
    }
};

struct CompositingComputeTask : CompositingComputeUses {
    CompositingComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        auto const &image_info = ti.get_device().info_image(uses.dst_image_id.image());
        state->record_commands(cmd_list, {image_info.size.x, image_info.size.y});
    }
};

struct PostprocessingRasterTask : PostprocessingRasterUses {
    PostprocessingRasterTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        auto const &image_info = ti.get_device().info_image(uses.render_image.image());
        state->record_commands(cmd_list, uses.render_image.image(), {image_info.size.x, image_info.size.y});
    }
};

struct Compositor {
    CompositingComputeTaskState compositing_compute_task_state;
    Compositor(AsyncPipelineManager &pipeline_manager)
        : compositing_compute_task_state{pipeline_manager} {
    }

    auto render(RecordContext &record_ctx, GbufferDepth &gbuffer_depth, daxa::TaskImageView sky_lut, daxa::TaskImageView ssao_image, daxa::TaskImageView shading_image) -> daxa::TaskImageView {
        auto output_image = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "composited_image",
        });

        record_ctx.task_graph.add_task(CompositingComputeTask{
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .g_buffer_image_id = gbuffer_depth.gbuffer,
                    .sky_lut = sky_lut,
                    .ssao_image_id = ssao_image,
                    .shading_image_id = shading_image,
                    .dst_image_id = output_image,
                },
            },
            &compositing_compute_task_state,
        });

        return output_image;
    }
};

#endif
