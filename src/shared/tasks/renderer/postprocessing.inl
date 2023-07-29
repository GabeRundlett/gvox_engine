#pragma once

#include <shared/core.inl>

DAXA_DECL_TASK_USES_BEGIN(PostprocessingRasterUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(settings, daxa_BufferPtr(GpuSettings), FRAGMENT_SHADER_READ)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), FRAGMENT_SHADER_READ)
DAXA_TASK_USE_IMAGE(g_buffer_image_id, REGULAR_2D, FRAGMENT_SHADER_STORAGE_READ_WRITE)
DAXA_TASK_USE_IMAGE(particles_image_id, REGULAR_2D, FRAGMENT_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(ssao_image_id, REGULAR_2D, FRAGMENT_SHADER_STORAGE_READ_WRITE)
DAXA_TASK_USE_IMAGE(indirect_diffuse_image_id, REGULAR_2D, FRAGMENT_SHADER_STORAGE_READ_WRITE)
DAXA_TASK_USE_IMAGE(reconstructed_shading_image_id, REGULAR_2D, FRAGMENT_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(render_image, REGULAR_2D, COLOR_ATTACHMENT)
DAXA_DECL_TASK_USES_END()

struct PostprocessingRasterPush {
    daxa_SamplerId final_sampler;
    u32vec2 final_size;
};

#if defined(__cplusplus)

struct PostprocessingRasterTaskState {
    daxa::PipelineManager &pipeline_manager;
    AppUi &ui;
    std::shared_ptr<daxa::RasterPipeline> pipeline;
    daxa::SamplerId &sampler;
    daxa::Format render_color_format;

    auto get_color_format() -> daxa::Format {
        return render_color_format;
    }

    void compile_pipeline() {
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
            ui.console.add_log(compile_result.message());
            return;
        }
        pipeline = compile_result.value();
        if (!compile_result.value()->is_valid()) {
            ui.console.add_log(compile_result.message());
        }
    }

    PostprocessingRasterTaskState(daxa::PipelineManager &a_pipeline_manager, AppUi &a_ui, daxa::SamplerId &a_sampler, daxa::Format a_render_color_format = daxa::Format::R32G32B32A32_SFLOAT) : pipeline_manager{a_pipeline_manager}, ui{a_ui}, sampler{a_sampler}, render_color_format{a_render_color_format} { compile_pipeline(); }
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
            .final_sampler = sampler,
            .final_size = size,
        });
        cmd_list.draw({.vertex_count = 3});
        cmd_list.end_renderpass();
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

#endif
