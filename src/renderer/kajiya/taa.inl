#pragma once

#include <core.inl>
#include <application/input.inl>
#include <application/globals.inl>

#define TAA_WG_SIZE_X 16
#define TAA_WG_SIZE_Y 8

DAXA_DECL_TASK_HEAD_BEGIN(TaaReprojectCompute, 7)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_map)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, reprojected_history_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, closest_velocity_img)
DAXA_DECL_TASK_HEAD_END
struct TaaReprojectComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    DAXA_TH_BLOB(TaaReprojectCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(TaaFilterInputCompute, 6)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_image)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, filtered_input_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, filtered_input_deviation_img)
DAXA_DECL_TASK_HEAD_END
struct TaaFilterInputComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    DAXA_TH_BLOB(TaaFilterInputCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(TaaFilterHistoryCompute, 4)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojected_history_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, filtered_history_img)
DAXA_DECL_TASK_HEAD_END
struct TaaFilterHistoryComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    DAXA_TH_BLOB(TaaFilterHistoryCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(TaaInputProbCompute, 12)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_image)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, filtered_input_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, filtered_input_deviation_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojected_history_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, filtered_history_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_map)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, smooth_var_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, velocity_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, input_prob_img)
DAXA_DECL_TASK_HEAD_END
struct TaaInputProbComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    DAXA_TH_BLOB(TaaInputProbCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(TaaProbFilterCompute, 4)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_prob_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, prob_filtered1_img)
DAXA_DECL_TASK_HEAD_END
struct TaaProbFilterComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    DAXA_TH_BLOB(TaaProbFilterCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(TaaProbFilter2Compute, 4)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, prob_filtered1_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, prob_filtered2_img)
DAXA_DECL_TASK_HEAD_END
struct TaaProbFilter2ComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    DAXA_TH_BLOB(TaaProbFilter2Compute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(TaaCompute, 14)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_image)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojected_history_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_map)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, closest_velocity_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, velocity_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, smooth_var_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_prob_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, temporal_output_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, this_frame_output_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, smooth_var_output_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, temporal_velocity_output_tex)
DAXA_DECL_TASK_HEAD_END
struct TaaComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    DAXA_TH_BLOB(TaaCompute, uses)
};

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
        ping_pong_taa_col_image.swap();
        ping_pong_taa_vel_image.swap();
        ping_pong_smooth_var_image.swap();
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
            .source = daxa::ShaderFile{"kajiya/taa.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TaaReprojectCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
                daxa::TaskViewVariant{std::pair{TaaReprojectCompute::globals, record_ctx.gpu_context->task_globals_buffer}},

                daxa::TaskViewVariant{std::pair{TaaReprojectCompute::history_tex, history_tex}},
                daxa::TaskViewVariant{std::pair{TaaReprojectCompute::reprojection_map, reprojection_map}},
                daxa::TaskViewVariant{std::pair{TaaReprojectCompute::depth_image, depth_image}},

                daxa::TaskViewVariant{std::pair{TaaReprojectCompute::reprojected_history_img, reprojected_history_img}},
                daxa::TaskViewVariant{std::pair{TaaReprojectCompute::closest_velocity_img, closest_velocity_img}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaReprojectComputePush &push, TaaTaskInfo const &info) {
                push.input_tex_size = info.input_tex_size;
                push.output_tex_size = info.output_tex_size;
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
            },
            .info = {
                .thread_count = record_ctx.output_resolution,
                .input_tex_size = i_extent,
                .output_tex_size = o_extent,
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "taa reproject", .task_image_id = reprojected_history_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

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
            .source = daxa::ShaderFile{"kajiya/taa.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TaaFilterInputCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
                daxa::TaskViewVariant{std::pair{TaaFilterInputCompute::globals, record_ctx.gpu_context->task_globals_buffer}},

                daxa::TaskViewVariant{std::pair{TaaFilterInputCompute::input_image, input_image}},
                daxa::TaskViewVariant{std::pair{TaaFilterInputCompute::depth_image, depth_image}},

                daxa::TaskViewVariant{std::pair{TaaFilterInputCompute::filtered_input_img, filtered_input_img}},
                daxa::TaskViewVariant{std::pair{TaaFilterInputCompute::filtered_input_deviation_img, filtered_input_deviation_img}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaFilterInputComputePush &push, TaaTaskInfo const &info) {
                push.input_tex_size = info.input_tex_size;
                push.output_tex_size = info.output_tex_size;
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
            },
            .info = {
                .thread_count = record_ctx.render_resolution,
                .input_tex_size = i_extent,
                .output_tex_size = o_extent,
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "taa filter input", .task_image_id = filtered_input_deviation_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        auto filtered_history_img = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "filtered_history_img",
        });

        record_ctx.add(ComputeTask<TaaFilterHistoryCompute, TaaFilterHistoryComputePush, TaaTaskInfo>{
            .source = daxa::ShaderFile{"kajiya/taa.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TaaFilterHistoryCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
                daxa::TaskViewVariant{std::pair{TaaFilterHistoryCompute::globals, record_ctx.gpu_context->task_globals_buffer}},

                daxa::TaskViewVariant{std::pair{TaaFilterHistoryCompute::reprojected_history_img, reprojected_history_img}},

                daxa::TaskViewVariant{std::pair{TaaFilterHistoryCompute::filtered_history_img, filtered_history_img}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaFilterHistoryComputePush &push, TaaTaskInfo const &info) {
                push.input_tex_size = info.input_tex_size;
                push.output_tex_size = info.output_tex_size;
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
            },
            .info = {
                .thread_count = record_ctx.render_resolution,
                .input_tex_size = i_extent,
                .output_tex_size = o_extent,
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "taa filter history", .task_image_id = filtered_history_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        auto input_prob_img = [&]() {
            auto input_prob_img = record_ctx.task_graph.create_transient_image({
                .format = daxa::Format::R16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .name = "input_prob_img",
            });
            record_ctx.add(ComputeTask<TaaInputProbCompute, TaaInputProbComputePush, TaaTaskInfo>{
                .source = daxa::ShaderFile{"kajiya/taa.comp.glsl"},
                .views = std::array{
                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::globals, record_ctx.gpu_context->task_globals_buffer}},

                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::input_image, input_image}},
                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::filtered_input_img, filtered_input_img}},
                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::filtered_input_deviation_img, filtered_input_deviation_img}},
                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::reprojected_history_img, reprojected_history_img}},
                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::filtered_history_img, filtered_history_img}},
                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::reprojection_map, reprojection_map}},
                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::depth_image, depth_image}},
                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::smooth_var_history_tex, smooth_var_history_tex}},
                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::velocity_history_tex, velocity_history_tex}},

                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::input_prob_img, input_prob_img}},
                },
                .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaInputProbComputePush &push, TaaTaskInfo const &info) {
                    push.input_tex_size = info.input_tex_size;
                    push.output_tex_size = info.output_tex_size;
                    ti.recorder.set_pipeline(pipeline);
                    set_push_constant(ti, push);
                    ti.recorder.dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
                },
                .info = {
                    .thread_count = record_ctx.render_resolution,
                    .input_tex_size = i_extent,
                    .output_tex_size = o_extent,
                },
            });

            debug_utils::DebugDisplay::add_pass({.name = "taa input prob", .task_image_id = input_prob_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            auto prob_filtered1_img = record_ctx.task_graph.create_transient_image({
                .format = daxa::Format::R16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .name = "prob_filtered1_img",
            });

            record_ctx.add(ComputeTask<TaaProbFilterCompute, TaaProbFilterComputePush, TaaTaskInfo>{
                .source = daxa::ShaderFile{"kajiya/taa.comp.glsl"},
                .views = std::array{
                    daxa::TaskViewVariant{std::pair{TaaProbFilterCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
                    daxa::TaskViewVariant{std::pair{TaaProbFilterCompute::globals, record_ctx.gpu_context->task_globals_buffer}},

                    daxa::TaskViewVariant{std::pair{TaaProbFilterCompute::input_prob_img, input_prob_img}},

                    daxa::TaskViewVariant{std::pair{TaaProbFilterCompute::prob_filtered1_img, prob_filtered1_img}},
                },
                .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaProbFilterComputePush &push, TaaTaskInfo const &info) {
                    push.input_tex_size = info.input_tex_size;
                    push.output_tex_size = info.output_tex_size;
                    ti.recorder.set_pipeline(pipeline);
                    set_push_constant(ti, push);
                    ti.recorder.dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
                },
                .info = {
                    .thread_count = record_ctx.render_resolution,
                    .input_tex_size = i_extent,
                    .output_tex_size = o_extent,
                },
            });

            debug_utils::DebugDisplay::add_pass({.name = "taa prob filter 1", .task_image_id = prob_filtered1_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            auto prob_filtered2_img = record_ctx.task_graph.create_transient_image({
                .format = daxa::Format::R16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .name = "prob_filtered2_img",
            });

            record_ctx.add(ComputeTask<TaaProbFilter2Compute, TaaProbFilter2ComputePush, TaaTaskInfo>{
                .source = daxa::ShaderFile{"kajiya/taa.comp.glsl"},
                .views = std::array{
                    daxa::TaskViewVariant{std::pair{TaaProbFilter2Compute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
                    daxa::TaskViewVariant{std::pair{TaaProbFilter2Compute::globals, record_ctx.gpu_context->task_globals_buffer}},

                    daxa::TaskViewVariant{std::pair{TaaProbFilter2Compute::prob_filtered1_img, prob_filtered1_img}},

                    daxa::TaskViewVariant{std::pair{TaaProbFilter2Compute::prob_filtered2_img, prob_filtered2_img}},
                },
                .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaProbFilter2ComputePush &push, TaaTaskInfo const &info) {
                    push.input_tex_size = info.input_tex_size;
                    push.output_tex_size = info.output_tex_size;
                    ti.recorder.set_pipeline(pipeline);
                    set_push_constant(ti, push);
                    ti.recorder.dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
                },
                .info = {
                    .thread_count = record_ctx.render_resolution,
                    .input_tex_size = i_extent,
                    .output_tex_size = o_extent,
                },
            });

            debug_utils::DebugDisplay::add_pass({.name = "taa prob filter 2", .task_image_id = prob_filtered2_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            return prob_filtered2_img;
        }();

        auto this_frame_output_img = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {record_ctx.output_resolution.x, record_ctx.output_resolution.y, 1},
            .name = "this_frame_output_img",
        });

        record_ctx.add(ComputeTask<TaaCompute, TaaComputePush, TaaTaskInfo>{
            .source = daxa::ShaderFile{"kajiya/taa.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TaaCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
                daxa::TaskViewVariant{std::pair{TaaCompute::globals, record_ctx.gpu_context->task_globals_buffer}},

                daxa::TaskViewVariant{std::pair{TaaCompute::input_image, input_image}},
                daxa::TaskViewVariant{std::pair{TaaCompute::reprojected_history_img, reprojected_history_img}},
                daxa::TaskViewVariant{std::pair{TaaCompute::reprojection_map, reprojection_map}},
                daxa::TaskViewVariant{std::pair{TaaCompute::closest_velocity_img, closest_velocity_img}},
                daxa::TaskViewVariant{std::pair{TaaCompute::velocity_history_tex, velocity_history_tex}},
                daxa::TaskViewVariant{std::pair{TaaCompute::depth_image, depth_image}},
                daxa::TaskViewVariant{std::pair{TaaCompute::smooth_var_history_tex, smooth_var_history_tex}},
                daxa::TaskViewVariant{std::pair{TaaCompute::input_prob_img, input_prob_img}},

                daxa::TaskViewVariant{std::pair{TaaCompute::temporal_output_tex, temporal_output_tex}},
                daxa::TaskViewVariant{std::pair{TaaCompute::this_frame_output_img, this_frame_output_img}},
                daxa::TaskViewVariant{std::pair{TaaCompute::smooth_var_output_tex, smooth_var_output_tex}},
                daxa::TaskViewVariant{std::pair{TaaCompute::temporal_velocity_output_tex, temporal_velocity_output_tex}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaComputePush &push, TaaTaskInfo const &info) {
                push.input_tex_size = info.input_tex_size;
                push.output_tex_size = info.output_tex_size;
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
            },
            .info = {
                .thread_count = record_ctx.output_resolution,
                .input_tex_size = i_extent,
                .output_tex_size = o_extent,
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "taa", .task_image_id = this_frame_output_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        return daxa::TaskImageView{this_frame_output_img};
    }
};

#endif
