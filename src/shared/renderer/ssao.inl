#pragma once

#include <shared/core.inl>

#if SsaoComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(SsaoCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, ssao_image_id)
DAXA_DECL_TASK_HEAD_END
struct SsaoComputePush {
    SsaoCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(SsaoComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewId vs_normal_image_id = push.uses.vs_normal_image_id;
daxa_ImageViewId depth_image_id = push.uses.depth_image_id;
daxa_ImageViewId ssao_image_id = push.uses.ssao_image_id;
#endif
#endif

#if SsaoSpatialFilterComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(SsaoSpatialFilterCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, src_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, dst_image_id)
DAXA_DECL_TASK_HEAD_END
struct SsaoSpatialFilterComputePush {
    SsaoSpatialFilterCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(SsaoSpatialFilterComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewId vs_normal_image_id = push.uses.vs_normal_image_id;
daxa_ImageViewId depth_image_id = push.uses.depth_image_id;
daxa_ImageViewId src_image_id = push.uses.src_image_id;
daxa_ImageViewId dst_image_id = push.uses.dst_image_id;
#endif
#endif

#if SsaoUpscaleComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(SsaoUpscaleCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, src_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, dst_image_id)
DAXA_DECL_TASK_HEAD_END
struct SsaoUpscaleComputePush {
    SsaoUpscaleCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(SsaoUpscaleComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewId g_buffer_image_id = push.uses.g_buffer_image_id;
daxa_ImageViewId depth_image_id = push.uses.depth_image_id;
daxa_ImageViewId src_image_id = push.uses.src_image_id;
daxa_ImageViewId dst_image_id = push.uses.dst_image_id;
#endif
#endif

#if SsaoTemporalFilterComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(SsaoTemporalFilterCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, history_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, src_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, dst_image_id)
DAXA_DECL_TASK_HEAD_END
struct SsaoTemporalFilterComputePush {
    SsaoTemporalFilterCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(SsaoTemporalFilterComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewId reprojection_image_id = push.uses.reprojection_image_id;
daxa_ImageViewId history_image_id = push.uses.history_image_id;
daxa_ImageViewId src_image_id = push.uses.src_image_id;
daxa_ImageViewId dst_image_id = push.uses.dst_image_id;
#endif
#endif

#if defined(__cplusplus)


struct SsaoRenderer {
    PingPongImage ping_pong_ssao_image;

    void next_frame() {
        ping_pong_ssao_image.task_resources.output_resource.swap_images(ping_pong_ssao_image.task_resources.history_resource);
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
            .source = daxa::ShaderFile{"ssao.comp.glsl"},
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                .vs_normal_image_id = scaled_view_normal_image,
                .depth_image_id = scaled_depth_image,
                .ssao_image_id = ssao_image0,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SsaoCompute::Uses &uses, SsaoComputePush &push, NoTaskInfo const &) {
                auto const &image_info = ti.get_device().info_image(uses.ssao_image_id.image()).value();
                ti.get_recorder().set_pipeline(pipeline);
                ti.get_recorder().push_constant(push);
                // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
                ti.get_recorder().dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });
        record_ctx.add(ComputeTask<SsaoSpatialFilterCompute, SsaoSpatialFilterComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"ssao.comp.glsl"},
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .vs_normal_image_id = scaled_view_normal_image,
                .depth_image_id = scaled_depth_image,
                .src_image_id = ssao_image0,
                .dst_image_id = ssao_image1,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SsaoSpatialFilterCompute::Uses &uses, SsaoSpatialFilterComputePush &push, NoTaskInfo const &) {
                auto const &image_info = ti.get_device().info_image(uses.dst_image_id.image()).value();
                ti.get_recorder().set_pipeline(pipeline);
                ti.get_recorder().push_constant(push);
                // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
                ti.get_recorder().dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });
        record_ctx.add(ComputeTask<SsaoUpscaleCompute, SsaoUpscaleComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"ssao.comp.glsl"},
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .g_buffer_image_id = gbuffer_depth.gbuffer,
                .depth_image_id = gbuffer_depth.depth.task_resources.output_resource,
                .src_image_id = ssao_image1,
                .dst_image_id = ssao_image2,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SsaoUpscaleCompute::Uses &uses, SsaoUpscaleComputePush &push, NoTaskInfo const &) {
                auto const &image_info = ti.get_device().info_image(uses.dst_image_id.image()).value();
                ti.get_recorder().set_pipeline(pipeline);
                ti.get_recorder().push_constant(push);
                // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
                ti.get_recorder().dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });
        record_ctx.add(ComputeTask<SsaoTemporalFilterCompute, SsaoTemporalFilterComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"ssao.comp.glsl"},
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .reprojection_image_id = reprojection_map,
                .history_image_id = prev_ssao_image,
                .src_image_id = ssao_image2,
                .dst_image_id = ssao_image,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SsaoTemporalFilterCompute::Uses &uses, SsaoTemporalFilterComputePush &push, NoTaskInfo const &) {
                auto const &image_info = ti.get_device().info_image(uses.dst_image_id.image()).value();
                ti.get_recorder().set_pipeline(pipeline);
                ti.get_recorder().push_constant(push);
                // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
                ti.get_recorder().dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });

        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "ssao", .task_image_id = ssao_image0, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "ssao spatial", .task_image_id = ssao_image1, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "ssao upsample", .task_image_id = ssao_image2, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "ssao temporal", .task_image_id = ssao_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        return daxa::TaskImageView{ssao_image};
    }
};

#endif
