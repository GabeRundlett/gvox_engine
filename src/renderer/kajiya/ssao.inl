#pragma once

#include <core.inl>
#include <renderer/core.inl>

DAXA_DECL_TASK_HEAD_BEGIN(SsaoCompute, 5)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, ssao_image_id)
DAXA_DECL_TASK_HEAD_END
struct SsaoComputePush {
    DAXA_TH_BLOB(SsaoCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(SsaoSpatialFilterCompute, 5)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, src_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, dst_image_id)
DAXA_DECL_TASK_HEAD_END
struct SsaoSpatialFilterComputePush {
    DAXA_TH_BLOB(SsaoSpatialFilterCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(SsaoUpscaleCompute, 5)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, src_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, dst_image_id)
DAXA_DECL_TASK_HEAD_END
struct SsaoUpscaleComputePush {
    DAXA_TH_BLOB(SsaoUpscaleCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(SsaoTemporalFilterCompute, 5)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, history_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, src_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, dst_image_id)
DAXA_DECL_TASK_HEAD_END
struct SsaoTemporalFilterComputePush {
    DAXA_TH_BLOB(SsaoTemporalFilterCompute, uses)
};

#if defined(__cplusplus)

struct SsaoRenderer {
    PingPongImage ping_pong_ssao_image;

    void next_frame() {
        ping_pong_ssao_image.swap();
    }

    auto render(RecordContext &record_ctx, GbufferDepth &gbuffer_depth, daxa::TaskImageView reprojection_map) -> daxa::TaskImageView {
        auto scaled_depth_image = gbuffer_depth.get_downscaled_depth(record_ctx);
        auto scaled_view_normal_image = gbuffer_depth.get_downscaled_view_normal(record_ctx);
        ping_pong_ssao_image = PingPongImage{};
        auto [ssao_image, prev_ssao_image] = ping_pong_ssao_image.get(
            record_ctx.device,
            {
                .format = daxa::Format::R16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = "ssao_image",
            });

        clear_task_images(record_ctx.device, std::array{prev_ssao_image});

        record_ctx.task_graph.use_persistent_image(ssao_image);
        record_ctx.task_graph.use_persistent_image(prev_ssao_image);
        auto ssao_image0 = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16_SFLOAT,
            .size = {record_ctx.render_resolution.x / SHADING_SCL, record_ctx.render_resolution.y / SHADING_SCL, 1},
            .name = "ssao_image0",
        });
        auto ssao_image1 = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16_SFLOAT,
            .size = {record_ctx.render_resolution.x / SHADING_SCL, record_ctx.render_resolution.y / SHADING_SCL, 1},
            .name = "ssao_image1",
        });
        auto ssao_image2 = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "ssao_image2",
        });

        record_ctx.add(ComputeTask<SsaoCompute, SsaoComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"kajiya/ssao.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{SsaoCompute::gpu_input, record_ctx.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{SsaoCompute::globals, record_ctx.task_globals_buffer}},
                daxa::TaskViewVariant{std::pair{SsaoCompute::vs_normal_image_id, scaled_view_normal_image}},
                daxa::TaskViewVariant{std::pair{SsaoCompute::depth_image_id, scaled_depth_image}},
                daxa::TaskViewVariant{std::pair{SsaoCompute::ssao_image_id, ssao_image0}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SsaoComputePush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(SsaoCompute::ssao_image_id).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
                ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });
        record_ctx.add(ComputeTask<SsaoSpatialFilterCompute, SsaoSpatialFilterComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"kajiya/ssao.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{SsaoSpatialFilterCompute::gpu_input, record_ctx.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{SsaoSpatialFilterCompute::vs_normal_image_id, scaled_view_normal_image}},
                daxa::TaskViewVariant{std::pair{SsaoSpatialFilterCompute::depth_image_id, scaled_depth_image}},
                daxa::TaskViewVariant{std::pair{SsaoSpatialFilterCompute::src_image_id, ssao_image0}},
                daxa::TaskViewVariant{std::pair{SsaoSpatialFilterCompute::dst_image_id, ssao_image1}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SsaoSpatialFilterComputePush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(SsaoSpatialFilterCompute::dst_image_id).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
                ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });
        record_ctx.add(ComputeTask<SsaoUpscaleCompute, SsaoUpscaleComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"kajiya/ssao.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{SsaoUpscaleCompute::gpu_input, record_ctx.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{SsaoUpscaleCompute::g_buffer_image_id, gbuffer_depth.gbuffer}},
                daxa::TaskViewVariant{std::pair{SsaoUpscaleCompute::depth_image_id, gbuffer_depth.depth.current()}},
                daxa::TaskViewVariant{std::pair{SsaoUpscaleCompute::src_image_id, ssao_image1}},
                daxa::TaskViewVariant{std::pair{SsaoUpscaleCompute::dst_image_id, ssao_image2}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SsaoUpscaleComputePush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(SsaoUpscaleCompute::dst_image_id).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
                ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });
        record_ctx.add(ComputeTask<SsaoTemporalFilterCompute, SsaoTemporalFilterComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"kajiya/ssao.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{SsaoTemporalFilterCompute::gpu_input, record_ctx.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{SsaoTemporalFilterCompute::reprojection_image_id, reprojection_map}},
                daxa::TaskViewVariant{std::pair{SsaoTemporalFilterCompute::history_image_id, prev_ssao_image}},
                daxa::TaskViewVariant{std::pair{SsaoTemporalFilterCompute::src_image_id, ssao_image2}},
                daxa::TaskViewVariant{std::pair{SsaoTemporalFilterCompute::dst_image_id, ssao_image}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SsaoTemporalFilterComputePush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(SsaoTemporalFilterCompute::dst_image_id).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
                ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "ssao", .task_image_id = ssao_image0, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "ssao spatial", .task_image_id = ssao_image1, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "ssao upsample", .task_image_id = ssao_image2, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "ssao temporal", .task_image_id = ssao_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        return daxa::TaskImageView{ssao_image};
    }
};

#endif
