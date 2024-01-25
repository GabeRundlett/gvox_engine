#pragma once

#include <shared/core.inl>

#define TAA_WG_SIZE_X 16
#define TAA_WG_SIZE_Y 8

#if TaaReprojectComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(TaaReprojectCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, history_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_map)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, reprojected_history_img)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, closest_velocity_img)
DAXA_DECL_TASK_HEAD_END
struct TaaReprojectComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    TaaReprojectCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(TaaReprojectComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewId history_tex = push.uses.history_tex;
daxa_ImageViewId reprojection_map = push.uses.reprojection_map;
daxa_ImageViewId depth_image = push.uses.depth_image;
daxa_ImageViewId reprojected_history_img = push.uses.reprojected_history_img;
daxa_ImageViewId closest_velocity_img = push.uses.closest_velocity_img;
#endif
#endif
#if TaaFilterInputComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(TaaFilterInputCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_image)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, filtered_input_img)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, filtered_input_deviation_img)
DAXA_DECL_TASK_HEAD_END
struct TaaFilterInputComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    TaaFilterInputCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(TaaFilterInputComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewId input_image = push.uses.input_image;
daxa_ImageViewId depth_image = push.uses.depth_image;
daxa_ImageViewId filtered_input_img = push.uses.filtered_input_img;
daxa_ImageViewId filtered_input_deviation_img = push.uses.filtered_input_deviation_img;
#endif
#endif
#if TaaFilterHistoryComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(TaaFilterHistoryCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojected_history_img)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, filtered_history_img)
DAXA_DECL_TASK_HEAD_END
struct TaaFilterHistoryComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    TaaFilterHistoryCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(TaaFilterHistoryComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewId reprojected_history_img = push.uses.reprojected_history_img;
daxa_ImageViewId filtered_history_img = push.uses.filtered_history_img;
#endif
#endif
#if TaaInputProbComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(TaaInputProbCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_image)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, filtered_input_img)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, filtered_input_deviation_img)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojected_history_img)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, filtered_history_img)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_map)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, smooth_var_history_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, velocity_history_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, input_prob_img)
DAXA_DECL_TASK_HEAD_END
struct TaaInputProbComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    TaaInputProbCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(TaaInputProbComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewId input_image = push.uses.input_image;
daxa_ImageViewId filtered_input_img = push.uses.filtered_input_img;
daxa_ImageViewId filtered_input_deviation_img = push.uses.filtered_input_deviation_img;
daxa_ImageViewId reprojected_history_img = push.uses.reprojected_history_img;
daxa_ImageViewId filtered_history_img = push.uses.filtered_history_img;
daxa_ImageViewId reprojection_map = push.uses.reprojection_map;
daxa_ImageViewId depth_image = push.uses.depth_image;
daxa_ImageViewId smooth_var_history_tex = push.uses.smooth_var_history_tex;
daxa_ImageViewId velocity_history_tex = push.uses.velocity_history_tex;
daxa_ImageViewId input_prob_img = push.uses.input_prob_img;
#endif
#endif
#if TaaProbFilterComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(TaaProbFilterCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_prob_img)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, prob_filtered1_img)
DAXA_DECL_TASK_HEAD_END
struct TaaProbFilterComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    TaaProbFilterCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(TaaProbFilterComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewId input_prob_img = push.uses.input_prob_img;
daxa_ImageViewId prob_filtered1_img = push.uses.prob_filtered1_img;
#endif
#endif
#if TaaProbFilter2ComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(TaaProbFilter2Compute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, prob_filtered1_img)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, prob_filtered2_img)
DAXA_DECL_TASK_HEAD_END
struct TaaProbFilter2ComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    TaaProbFilter2Compute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(TaaProbFilter2ComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewId prob_filtered1_img = push.uses.prob_filtered1_img;
daxa_ImageViewId prob_filtered2_img = push.uses.prob_filtered2_img;
#endif
#endif
#if TaaComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(TaaCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_image)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojected_history_img)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_map)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, closest_velocity_img)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, velocity_history_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, smooth_var_history_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_prob_img)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, temporal_output_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, this_frame_output_img)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, smooth_var_output_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, temporal_velocity_output_tex)
DAXA_DECL_TASK_HEAD_END
struct TaaComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    TaaCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(TaaComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewId input_image = push.uses.input_image;
daxa_ImageViewId reprojected_history_img = push.uses.reprojected_history_img;
daxa_ImageViewId reprojection_map = push.uses.reprojection_map;
daxa_ImageViewId closest_velocity_img = push.uses.closest_velocity_img;
daxa_ImageViewId velocity_history_tex = push.uses.velocity_history_tex;
daxa_ImageViewId depth_image = push.uses.depth_image;
daxa_ImageViewId smooth_var_history_tex = push.uses.smooth_var_history_tex;
daxa_ImageViewId input_prob_img = push.uses.input_prob_img;
daxa_ImageViewId temporal_output_tex = push.uses.temporal_output_tex;
daxa_ImageViewId this_frame_output_img = push.uses.this_frame_output_img;
daxa_ImageViewId smooth_var_output_tex = push.uses.smooth_var_output_tex;
daxa_ImageViewId temporal_velocity_output_tex = push.uses.temporal_velocity_output_tex;
#endif
#endif

struct TaaPushCommon {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
};

#if defined(__cplusplus)

struct TaaRenderer {
    PingPongImage ping_pong_taa_col_image;
    PingPongImage ping_pong_taa_vel_image;
    PingPongImage ping_pong_smooth_var_image;

    void next_frame() {
        ping_pong_taa_col_image.task_resources.output_resource.swap_images(ping_pong_taa_col_image.task_resources.history_resource);
        ping_pong_taa_vel_image.task_resources.output_resource.swap_images(ping_pong_taa_vel_image.task_resources.history_resource);
        ping_pong_smooth_var_image.task_resources.output_resource.swap_images(ping_pong_smooth_var_image.task_resources.history_resource);
    }

    auto render(RecordContext &record_ctx, daxa::TaskImageView input_image, daxa::TaskImageView depth_image, daxa::TaskImageView reprojection_map) -> daxa::TaskImageView {
        ping_pong_taa_col_image = PingPongImage{};
        ping_pong_taa_vel_image = PingPongImage{};
        ping_pong_smooth_var_image = PingPongImage{};
        auto [temporal_output_tex, history_tex] = ping_pong_taa_col_image.get(
            record_ctx.device,
            {
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {record_ctx.output_resolution.x, record_ctx.output_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_SRC,
                .name = "taa_col",
            });
        auto [temporal_velocity_output_tex, velocity_history_tex] = ping_pong_taa_vel_image.get(
            record_ctx.device,
            {
                .format = daxa::Format::R16G16_SFLOAT,
                .size = {record_ctx.output_resolution.x, record_ctx.output_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_SRC,
                .name = "taa_vel",
            });
        auto [smooth_var_output_tex, smooth_var_history_tex] = ping_pong_smooth_var_image.get(
            record_ctx.device,
            {
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {record_ctx.output_resolution.x, record_ctx.output_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_SRC,
                .name = "smooth_var",
            });
        record_ctx.task_graph.use_persistent_image(temporal_output_tex);
        record_ctx.task_graph.use_persistent_image(history_tex);
        record_ctx.task_graph.use_persistent_image(temporal_velocity_output_tex);
        record_ctx.task_graph.use_persistent_image(velocity_history_tex);
        record_ctx.task_graph.use_persistent_image(smooth_var_output_tex);
        record_ctx.task_graph.use_persistent_image(smooth_var_history_tex);

        auto reprojected_history_img = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {record_ctx.output_resolution.x, record_ctx.output_resolution.y, 1},
            .name = "reprojected_history_img",
        });
        auto closest_velocity_img = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16_SFLOAT,
            .size = {record_ctx.output_resolution.x, record_ctx.output_resolution.y, 1},
            .name = "closest_velocity_img",
        });

        auto i_extent = daxa_f32vec2(static_cast<daxa_f32>(record_ctx.render_resolution.x), static_cast<daxa_f32>(record_ctx.render_resolution.y));
        auto o_extent = daxa_f32vec2(static_cast<daxa_f32>(record_ctx.output_resolution.x), static_cast<daxa_f32>(record_ctx.output_resolution.y));

        struct TaaTaskInfo {
            daxa_u32vec2 thread_count;
            daxa_f32vec2 input_tex_size;
            daxa_f32vec2 output_tex_size;
        };

        record_ctx.add(ComputeTask<TaaReprojectCompute, TaaReprojectComputePush, TaaTaskInfo>{
            .source = daxa::ShaderFile{"taa.comp.glsl"},
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,

                .history_tex = history_tex,
                .reprojection_map = reprojection_map,
                .depth_image = depth_image,

                .reprojected_history_img = reprojected_history_img,
                .closest_velocity_img = closest_velocity_img,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaReprojectCompute::Uses &, TaaReprojectComputePush &push, TaaTaskInfo const &info) {
                push.input_tex_size = info.input_tex_size;
                push.output_tex_size = info.output_tex_size;
                ti.get_recorder().set_pipeline(pipeline);
                ti.get_recorder().push_constant(push);
                ti.get_recorder().dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
            },
            .info = {
                .thread_count = record_ctx.output_resolution,
                .input_tex_size = i_extent,
                .output_tex_size = o_extent,
            },
        });

        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "taa reproject", .task_image_id = reprojected_history_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        auto filtered_input_img = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "filtered_input_img",
        });
        auto filtered_input_deviation_img = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "filtered_input_deviation_img",
        });

        record_ctx.add(ComputeTask<TaaFilterInputCompute, TaaFilterInputComputePush, TaaTaskInfo>{
            .source = daxa::ShaderFile{"taa.comp.glsl"},
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,

                .input_image = input_image,
                .depth_image = depth_image,

                .filtered_input_img = filtered_input_img,
                .filtered_input_deviation_img = filtered_input_deviation_img,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaFilterInputCompute::Uses &, TaaFilterInputComputePush &push, TaaTaskInfo const &info) {
                push.input_tex_size = info.input_tex_size;
                push.output_tex_size = info.output_tex_size;
                ti.get_recorder().set_pipeline(pipeline);
                ti.get_recorder().push_constant(push);
                ti.get_recorder().dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
            },
            .info = {
                .thread_count = record_ctx.render_resolution,
                .input_tex_size = i_extent,
                .output_tex_size = o_extent,
            },
        });

        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "taa filter input", .task_image_id = filtered_input_deviation_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        auto filtered_history_img = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "filtered_history_img",
        });

        record_ctx.add(ComputeTask<TaaFilterHistoryCompute, TaaFilterHistoryComputePush, TaaTaskInfo>{
            .source = daxa::ShaderFile{"taa.comp.glsl"},
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,

                .reprojected_history_img = reprojected_history_img,

                .filtered_history_img = filtered_history_img,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaFilterHistoryCompute::Uses &, TaaFilterHistoryComputePush &push, TaaTaskInfo const &info) {
                push.input_tex_size = info.input_tex_size;
                push.output_tex_size = info.output_tex_size;
                ti.get_recorder().set_pipeline(pipeline);
                ti.get_recorder().push_constant(push);
                ti.get_recorder().dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
            },
            .info = {
                .thread_count = record_ctx.render_resolution,
                .input_tex_size = i_extent,
                .output_tex_size = o_extent,
            },
        });

        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "taa filter history", .task_image_id = filtered_history_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        auto input_prob_img = [&]() {
            auto input_prob_img = record_ctx.task_graph.create_transient_image({
                .format = daxa::Format::R16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .name = "input_prob_img",
            });
            record_ctx.add(ComputeTask<TaaInputProbCompute, TaaInputProbComputePush, TaaTaskInfo>{
                .source = daxa::ShaderFile{"taa.comp.glsl"},
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .globals = record_ctx.task_globals_buffer,

                    .input_image = input_image,
                    .filtered_input_img = filtered_input_img,
                    .filtered_input_deviation_img = filtered_input_deviation_img,
                    .reprojected_history_img = reprojected_history_img,
                    .filtered_history_img = filtered_history_img,
                    .reprojection_map = reprojection_map,
                    .depth_image = depth_image,
                    .smooth_var_history_tex = smooth_var_history_tex,
                    .velocity_history_tex = velocity_history_tex,

                    .input_prob_img = input_prob_img,
                },
                .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaInputProbCompute::Uses &, TaaInputProbComputePush &push, TaaTaskInfo const &info) {
                    push.input_tex_size = info.input_tex_size;
                    push.output_tex_size = info.output_tex_size;
                    ti.get_recorder().set_pipeline(pipeline);
                    ti.get_recorder().push_constant(push);
                    ti.get_recorder().dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
                },
                .info = {
                    .thread_count = record_ctx.render_resolution,
                    .input_tex_size = i_extent,
                    .output_tex_size = o_extent,
                },
            });

            AppUi::DebugDisplay::s_instance->passes.push_back({.name = "taa input prob", .task_image_id = input_prob_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            auto prob_filtered1_img = record_ctx.task_graph.create_transient_image({
                .format = daxa::Format::R16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .name = "prob_filtered1_img",
            });

            record_ctx.add(ComputeTask<TaaProbFilterCompute, TaaProbFilterComputePush, TaaTaskInfo>{
                .source = daxa::ShaderFile{"taa.comp.glsl"},
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .globals = record_ctx.task_globals_buffer,

                    .input_prob_img = input_prob_img,

                    .prob_filtered1_img = prob_filtered1_img,
                },
                .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaProbFilterCompute::Uses &, TaaProbFilterComputePush &push, TaaTaskInfo const &info) {
                    push.input_tex_size = info.input_tex_size;
                    push.output_tex_size = info.output_tex_size;
                    ti.get_recorder().set_pipeline(pipeline);
                    ti.get_recorder().push_constant(push);
                    ti.get_recorder().dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
                },
                .info = {
                    .thread_count = record_ctx.render_resolution,
                    .input_tex_size = i_extent,
                    .output_tex_size = o_extent,
                },
            });

            AppUi::DebugDisplay::s_instance->passes.push_back({.name = "taa prob filter 1", .task_image_id = prob_filtered1_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            auto prob_filtered2_img = record_ctx.task_graph.create_transient_image({
                .format = daxa::Format::R16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .name = "prob_filtered2_img",
            });

            record_ctx.add(ComputeTask<TaaProbFilter2Compute, TaaProbFilter2ComputePush, TaaTaskInfo>{
                .source = daxa::ShaderFile{"taa.comp.glsl"},
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .globals = record_ctx.task_globals_buffer,

                    .prob_filtered1_img = prob_filtered1_img,

                    .prob_filtered2_img = prob_filtered2_img,
                },
                .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaProbFilter2Compute::Uses &, TaaProbFilter2ComputePush &push, TaaTaskInfo const &info) {
                    push.input_tex_size = info.input_tex_size;
                    push.output_tex_size = info.output_tex_size;
                    ti.get_recorder().set_pipeline(pipeline);
                    ti.get_recorder().push_constant(push);
                    ti.get_recorder().dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
                },
                .info = {
                    .thread_count = record_ctx.render_resolution,
                    .input_tex_size = i_extent,
                    .output_tex_size = o_extent,
                },
            });

            AppUi::DebugDisplay::s_instance->passes.push_back({.name = "taa prob filter 2", .task_image_id = prob_filtered2_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            return prob_filtered2_img;
        }();

        auto this_frame_output_img = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {record_ctx.output_resolution.x, record_ctx.output_resolution.y, 1},
            .name = "this_frame_output_img",
        });

        record_ctx.add(ComputeTask<TaaCompute, TaaComputePush, TaaTaskInfo>{
            .source = daxa::ShaderFile{"taa.comp.glsl"},
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,

                .input_image = input_image,
                .reprojected_history_img = reprojected_history_img,
                .reprojection_map = reprojection_map,
                .closest_velocity_img = closest_velocity_img,
                .velocity_history_tex = velocity_history_tex,
                .depth_image = depth_image,
                .smooth_var_history_tex = smooth_var_history_tex,
                .input_prob_img = input_prob_img,

                .temporal_output_tex = temporal_output_tex,
                .this_frame_output_img = this_frame_output_img,
                .smooth_var_output_tex = smooth_var_output_tex,
                .temporal_velocity_output_tex = temporal_velocity_output_tex,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaCompute::Uses &, TaaComputePush &push, TaaTaskInfo const &info) {
                push.input_tex_size = info.input_tex_size;
                push.output_tex_size = info.output_tex_size;
                ti.get_recorder().set_pipeline(pipeline);
                ti.get_recorder().push_constant(push);
                ti.get_recorder().dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
            },
            .info = {
                .thread_count = record_ctx.output_resolution,
                .input_tex_size = i_extent,
                .output_tex_size = o_extent,
            },
        });

        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "taa", .task_image_id = this_frame_output_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        return daxa::TaskImageView{this_frame_output_img};
    }
};

#endif
