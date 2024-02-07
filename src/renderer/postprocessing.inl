#pragma once

#include <shared/core.inl>
#include <shared/input.inl>
#include <shared/globals.inl>
#include <renderer/core.inl>

DAXA_DECL_TASK_HEAD_BEGIN(PostprocessingRaster, 3)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(FRAGMENT_SHADER_SAMPLED, REGULAR_2D, composited_image_id)
DAXA_TH_IMAGE_INDEX(COLOR_ATTACHMENT, REGULAR_2D, render_image)
DAXA_DECL_TASK_HEAD_END
struct PostprocessingRasterPush {
    DAXA_TH_BLOB(PostprocessingRaster, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(DebugImageRaster, 4)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(FRAGMENT_SHADER_SAMPLED, REGULAR_2D, image_id)
DAXA_TH_IMAGE_INDEX(FRAGMENT_SHADER_SAMPLED, REGULAR_2D_ARRAY, cube_image_id)
DAXA_TH_IMAGE_INDEX(COLOR_ATTACHMENT, REGULAR_2D, render_image)
DAXA_DECL_TASK_HEAD_END
struct DebugImageRasterPush {
    daxa_u32 type;
    daxa_u32 cube_size;
    daxa_u32vec2 output_tex_size;
    DAXA_TH_BLOB(DebugImageRaster, uses)
};

#if defined(__cplusplus)

inline void tonemap_raster(RecordContext &record_ctx, daxa::TaskImageView antialiased_image, daxa::TaskImageView output_image, daxa::Format output_format) {
    record_ctx.add(RasterTask<PostprocessingRaster, PostprocessingRasterPush, NoTaskInfo>{
        .vert_source = daxa::ShaderFile{"FULL_SCREEN_TRIANGLE_VERTEX_SHADER"},
        .frag_source = daxa::ShaderFile{"postprocessing.raster.glsl"},
        .color_attachments = {{
            .format = output_format,
        }},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{PostprocessingRaster::gpu_input, record_ctx.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{PostprocessingRaster::composited_image_id, antialiased_image}},
            daxa::TaskViewVariant{std::pair{PostprocessingRaster::render_image, output_image}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, PostprocessingRasterPush &push, NoTaskInfo const &) {
            auto render_image = ti.get(PostprocessingRaster::render_image).ids[0];
            auto const image_info = ti.device.info_image(render_image).value();
            auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                .color_attachments = {{.image_view = render_image.default_view(), .load_op = daxa::AttachmentLoadOp::DONT_CARE, .clear_value = std::array<daxa_f32, 4>{0.0f, 0.0f, 0.0f, 0.0f}}},
                .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
            });
            renderpass_recorder.set_pipeline(pipeline);
            set_push_constant(ti, renderpass_recorder, push);
            renderpass_recorder.draw({.vertex_count = 3});
            ti.recorder = std::move(renderpass_recorder).end_renderpass();
        },
    });
}

inline void debug_pass(RecordContext &record_ctx, AppUi::Pass const &pass, daxa::TaskImageView output_image, daxa::Format output_format) {
    struct DebugImageRasterTaskInfo {
        daxa_u32 type;
    };

    record_ctx.add(RasterTask<DebugImageRaster, DebugImageRasterPush, DebugImageRasterTaskInfo>{
        .vert_source = daxa::ShaderFile{"FULL_SCREEN_TRIANGLE_VERTEX_SHADER"},
        .frag_source = daxa::ShaderFile{"postprocessing.raster.glsl"},
        .color_attachments = {{
            .format = output_format,
        }},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{DebugImageRaster::gpu_input, record_ctx.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{DebugImageRaster::image_id, pass.type == DEBUG_IMAGE_TYPE_CUBEMAP ? record_ctx.task_debug_texture : pass.task_image_id}},
            daxa::TaskViewVariant{std::pair{DebugImageRaster::cube_image_id, pass.type == DEBUG_IMAGE_TYPE_CUBEMAP ? pass.task_image_id : record_ctx.task_debug_texture}},
            daxa::TaskViewVariant{std::pair{DebugImageRaster::render_image, output_image}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, DebugImageRasterPush &push, DebugImageRasterTaskInfo const &info) {
            auto render_image = ti.get(DebugImageRaster::render_image).ids[0];
            auto const image_info = ti.device.info_image(render_image).value();
            auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                .color_attachments = {{.image_view = render_image.default_view(), .load_op = daxa::AttachmentLoadOp::DONT_CARE, .clear_value = std::array<daxa_f32, 4>{0.0f, 0.0f, 0.0f, 0.0f}}},
                .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
            });
            push.type = info.type;
            if (info.type == DEBUG_IMAGE_TYPE_CUBEMAP) {
                push.cube_size = ti.device.info_image(ti.get(DebugImageRaster::cube_image_id).ids[0]).value().size.x;
            }
            push.output_tex_size = {image_info.size.x, image_info.size.y};
            renderpass_recorder.set_pipeline(pipeline);
            set_push_constant(ti, renderpass_recorder, push);
            renderpass_recorder.draw({.vertex_count = 3});
            ti.recorder = std::move(renderpass_recorder).end_renderpass();
        },
        .info = {
            .type = pass.type,
        },
    });
}

#endif
