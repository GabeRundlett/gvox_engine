#ifndef RENDERER_KAJIYA_TAA_INL
#define RENDERER_KAJIYA_TAA_INL

#include <core.inl>
#include <application/input.inl>

#define TAA_WG_SIZE_X 16
#define TAA_WG_SIZE_Y 8

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(TaaReprojectCompute)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, history_tex)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, reprojection_map)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_INDEX(WRITE, REGULAR_2D, reprojected_history_img)
DAXA_TH_IMAGE_INDEX(WRITE, REGULAR_2D, closest_velocity_img)
DAXA_DECL_TASK_HEAD_END
struct TaaReprojectComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    DAXA_TH_BLOB(TaaReprojectCompute, uses)
};
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(TaaFilterInputCompute)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, input_image)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_INDEX(WRITE, REGULAR_2D, filtered_input_img)
DAXA_TH_IMAGE_INDEX(WRITE, REGULAR_2D, filtered_input_deviation_img)
DAXA_DECL_TASK_HEAD_END
struct TaaFilterInputComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    DAXA_TH_BLOB(TaaFilterInputCompute, uses)
};
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(TaaFilterHistoryCompute)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, reprojected_history_img)
DAXA_TH_IMAGE_INDEX(WRITE, REGULAR_2D, filtered_history_img)
DAXA_DECL_TASK_HEAD_END
struct TaaFilterHistoryComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    DAXA_TH_BLOB(TaaFilterHistoryCompute, uses)
};
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(TaaInputProbCompute)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, input_image)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, filtered_input_img)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, filtered_input_deviation_img)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, reprojected_history_img)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, filtered_history_img)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, reprojection_map)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, smooth_var_history_tex)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, velocity_history_tex)
DAXA_TH_IMAGE_INDEX(WRITE, REGULAR_2D, input_prob_img)
DAXA_DECL_TASK_HEAD_END
struct TaaInputProbComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    DAXA_TH_BLOB(TaaInputProbCompute, uses)
};
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(TaaProbFilterCompute)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, input_prob_img)
DAXA_TH_IMAGE_INDEX(WRITE, REGULAR_2D, prob_filtered1_img)
DAXA_DECL_TASK_HEAD_END
struct TaaProbFilterComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    DAXA_TH_BLOB(TaaProbFilterCompute, uses)
};
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(TaaProbFilter2Compute)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, prob_filtered1_img)
DAXA_TH_IMAGE_INDEX(WRITE, REGULAR_2D, prob_filtered2_img)
DAXA_DECL_TASK_HEAD_END
struct TaaProbFilter2ComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    DAXA_TH_BLOB(TaaProbFilter2Compute, uses)
};
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(TaaCompute)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, input_image)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, reprojected_history_img)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, reprojection_map)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, closest_velocity_img)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, velocity_history_tex)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, smooth_var_history_tex)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, input_prob_img)
DAXA_TH_IMAGE_INDEX(WRITE, REGULAR_2D, temporal_output_tex)
DAXA_TH_IMAGE_INDEX(WRITE, REGULAR_2D, this_frame_output_img)
DAXA_TH_IMAGE_INDEX(WRITE, REGULAR_2D, smooth_var_output_tex)
DAXA_TH_IMAGE_INDEX(WRITE, REGULAR_2D, temporal_velocity_output_tex)
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

    auto render(GpuContext &gpu_context, daxa::TaskImageView input_image, daxa::TaskImageView depth_image, daxa::TaskImageView reprojection_map) -> daxa::TaskImageView {
        ping_pong_taa_col_image = PingPongImage{};
        ping_pong_taa_vel_image = PingPongImage{};
        ping_pong_smooth_var_image = PingPongImage{};
        auto [temporal_output_tex, history_tex] = ping_pong_taa_col_image.get(
            gpu_context,
            {
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {gpu_context.output_resolution.x, gpu_context.output_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_SRC,
                .name = "taa_col",
            });
        auto [temporal_velocity_output_tex, velocity_history_tex] = ping_pong_taa_vel_image.get(
            gpu_context,
            {
                .format = daxa::Format::R16G16_SFLOAT,
                .size = {gpu_context.output_resolution.x, gpu_context.output_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_SRC,
                .name = "taa_vel",
            });
        auto [smooth_var_output_tex, smooth_var_history_tex] = ping_pong_smooth_var_image.get(
            gpu_context,
            {
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {gpu_context.output_resolution.x, gpu_context.output_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_SRC,
                .name = "smooth_var",
            });
        gpu_context.frame_task_graph.register_image(temporal_output_tex);
        gpu_context.frame_task_graph.register_image(history_tex);
        gpu_context.frame_task_graph.register_image(temporal_velocity_output_tex);
        gpu_context.frame_task_graph.register_image(velocity_history_tex);
        gpu_context.frame_task_graph.register_image(smooth_var_output_tex);
        gpu_context.frame_task_graph.register_image(smooth_var_history_tex);

        auto reprojected_history_img = gpu_context.frame_task_graph.create_task_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {gpu_context.output_resolution.x, gpu_context.output_resolution.y, 1},
            .name = "reprojected_history_img",
        });
        auto closest_velocity_img = gpu_context.frame_task_graph.create_task_image({
            .format = daxa::Format::R16G16_SFLOAT,
            .size = {gpu_context.output_resolution.x, gpu_context.output_resolution.y, 1},
            .name = "closest_velocity_img",
        });

        auto i_extent = daxa_f32vec2(static_cast<daxa_f32>(gpu_context.render_resolution.x), static_cast<daxa_f32>(gpu_context.render_resolution.y));
        auto o_extent = daxa_f32vec2(static_cast<daxa_f32>(gpu_context.output_resolution.x), static_cast<daxa_f32>(gpu_context.output_resolution.y));

        struct TaaTaskInfo {
            daxa_u32vec2 thread_count;
            daxa_f32vec2 input_tex_size;
            daxa_f32vec2 output_tex_size;
        };

        gpu_context.add(ComputeTask<TaaReprojectCompute::Info, TaaReprojectComputePush, TaaTaskInfo>{
            .source = "kajiya/taa/reproject_history.comp.glsl",
            .views = TaaReprojectCompute::Views{
                .gpu_input = gpu_context.task_input_buffer.view(),

                .history_tex = history_tex.view(),
                .reprojection_map = reprojection_map,
                .depth_image = depth_image,

                .reprojected_history_img = reprojected_history_img,
                .closest_velocity_img = closest_velocity_img,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaReprojectComputePush &push, TaaTaskInfo const &info) {
                push.input_tex_size = info.input_tex_size;
                push.output_tex_size = info.output_tex_size;
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
            },
            .info = {
                .thread_count = gpu_context.output_resolution,
                .input_tex_size = i_extent,
                .output_tex_size = o_extent,
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "taa reproject", .task_image_id = reprojected_history_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        auto filtered_input_img = gpu_context.frame_task_graph.create_task_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "filtered_input_img",
        });
        auto filtered_input_deviation_img = gpu_context.frame_task_graph.create_task_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "filtered_input_deviation_img",
        });

        gpu_context.add(ComputeTask<TaaFilterInputCompute::Info, TaaFilterInputComputePush, TaaTaskInfo>{
            .source = "kajiya/taa/filter_input.comp.glsl",
            .views = TaaFilterInputCompute::Views{
                .gpu_input = gpu_context.task_input_buffer.view(),

                .input_image = input_image,
                .depth_image = depth_image,

                .filtered_input_img = filtered_input_img,
                .filtered_input_deviation_img = filtered_input_deviation_img,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaFilterInputComputePush &push, TaaTaskInfo const &info) {
                push.input_tex_size = info.input_tex_size;
                push.output_tex_size = info.output_tex_size;
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
            },
            .info = {
                .thread_count = gpu_context.render_resolution,
                .input_tex_size = i_extent,
                .output_tex_size = o_extent,
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "taa filter input", .task_image_id = filtered_input_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "taa filter input deviation", .task_image_id = filtered_input_deviation_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        auto filtered_history_img = gpu_context.frame_task_graph.create_task_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "filtered_history_img",
        });

        gpu_context.add(ComputeTask<TaaFilterHistoryCompute::Info, TaaFilterHistoryComputePush, TaaTaskInfo>{
            .source = "kajiya/taa/filter_history.comp.glsl",
            .views = TaaFilterHistoryCompute::Views{
                .gpu_input = gpu_context.task_input_buffer.view(),

                .reprojected_history_img = reprojected_history_img,

                .filtered_history_img = filtered_history_img,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaFilterHistoryComputePush &push, TaaTaskInfo const &info) {
                push.input_tex_size = info.input_tex_size;
                push.output_tex_size = info.output_tex_size;
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
            },
            .info = {
                .thread_count = gpu_context.render_resolution,
                .input_tex_size = o_extent,
                .output_tex_size = i_extent,
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "taa filter history", .task_image_id = filtered_history_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        auto input_prob_img = [&]() {
            auto input_prob_img = gpu_context.frame_task_graph.create_task_image({
                .format = daxa::Format::R16_SFLOAT,
                .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
                .name = "input_prob_img",
            });
            gpu_context.add(ComputeTask<TaaInputProbCompute::Info, TaaInputProbComputePush, TaaTaskInfo>{
                .source = "kajiya/taa/input_prob.comp.glsl",
                .views = TaaInputProbCompute::Views{
                    .gpu_input = gpu_context.task_input_buffer.view(),

                    .input_image = input_image,
                    .filtered_input_img = filtered_input_img,
                    .filtered_input_deviation_img = filtered_input_deviation_img,
                    .reprojected_history_img = reprojected_history_img,
                    .filtered_history_img = filtered_history_img,
                    .reprojection_map = reprojection_map,
                    .depth_image = depth_image,
                    .smooth_var_history_tex = smooth_var_history_tex.view(),
                    .velocity_history_tex = velocity_history_tex.view(),

                    .input_prob_img = input_prob_img,
                },
                .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaInputProbComputePush &push, TaaTaskInfo const &info) {
                    push.input_tex_size = info.input_tex_size;
                    push.output_tex_size = info.output_tex_size;
                    ti.recorder.set_pipeline(pipeline);
                    set_push_constant(ti, push);
                    ti.recorder.dispatch({(info.thread_count.x + 15) / 16, (info.thread_count.y + 15) / 16});
                },
                .info = {
                    .thread_count = gpu_context.render_resolution,
                    .input_tex_size = i_extent,
                    .output_tex_size = o_extent,
                },
            });

            debug_utils::DebugDisplay::add_pass({.name = "taa input prob", .task_image_id = input_prob_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            auto prob_filtered1_img = gpu_context.frame_task_graph.create_task_image({
                .format = daxa::Format::R16_SFLOAT,
                .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
                .name = "prob_filtered1_img",
            });

            gpu_context.add(ComputeTask<TaaProbFilterCompute::Info, TaaProbFilterComputePush, TaaTaskInfo>{
                .source = "kajiya/taa/filter_prob.comp.glsl",
                .views = TaaProbFilterCompute::Views{
                    .gpu_input = gpu_context.task_input_buffer.view(),

                    .input_prob_img = input_prob_img,

                    .prob_filtered1_img = prob_filtered1_img,
                },
                .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaProbFilterComputePush &push, TaaTaskInfo const &info) {
                    push.input_tex_size = info.input_tex_size;
                    push.output_tex_size = info.output_tex_size;
                    ti.recorder.set_pipeline(pipeline);
                    set_push_constant(ti, push);
                    ti.recorder.dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
                },
                .info = {
                    .thread_count = gpu_context.render_resolution,
                    .input_tex_size = i_extent,
                    .output_tex_size = o_extent,
                },
            });

            debug_utils::DebugDisplay::add_pass({.name = "taa prob filter 1", .task_image_id = prob_filtered1_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            auto prob_filtered2_img = gpu_context.frame_task_graph.create_task_image({
                .format = daxa::Format::R16_SFLOAT,
                .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
                .name = "prob_filtered2_img",
            });

            gpu_context.add(ComputeTask<TaaProbFilter2Compute::Info, TaaProbFilter2ComputePush, TaaTaskInfo>{
                .source = "kajiya/taa/filter_prob2.comp.glsl",
                .views = TaaProbFilter2Compute::Views{
                    .gpu_input = gpu_context.task_input_buffer.view(),

                    .prob_filtered1_img = prob_filtered1_img,

                    .prob_filtered2_img = prob_filtered2_img,
                },
                .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaProbFilter2ComputePush &push, TaaTaskInfo const &info) {
                    push.input_tex_size = info.input_tex_size;
                    push.output_tex_size = info.output_tex_size;
                    ti.recorder.set_pipeline(pipeline);
                    set_push_constant(ti, push);
                    ti.recorder.dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
                },
                .info = {
                    .thread_count = gpu_context.render_resolution,
                    .input_tex_size = i_extent,
                    .output_tex_size = o_extent,
                },
            });

            debug_utils::DebugDisplay::add_pass({.name = "taa prob filter 2", .task_image_id = prob_filtered2_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            return prob_filtered2_img;
        }();

        auto this_frame_output_img = gpu_context.frame_task_graph.create_task_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {gpu_context.output_resolution.x, gpu_context.output_resolution.y, 1},
            .name = "this_frame_output_img",
        });

        gpu_context.add(ComputeTask<TaaCompute::Info, TaaComputePush, TaaTaskInfo>{
            .source = "kajiya/taa/taa.comp.glsl",
            .views = TaaCompute::Views{
                .gpu_input = gpu_context.task_input_buffer.view(),

                .input_image = input_image,
                .reprojected_history_img = reprojected_history_img,
                .reprojection_map = reprojection_map,
                .closest_velocity_img = closest_velocity_img,
                .velocity_history_tex = velocity_history_tex.view(),
                .depth_image = depth_image,
                .smooth_var_history_tex = smooth_var_history_tex.view(),
                .input_prob_img = input_prob_img,

                .temporal_output_tex = temporal_output_tex.view(),
                .this_frame_output_img = this_frame_output_img,
                .smooth_var_output_tex = smooth_var_output_tex.view(),
                .temporal_velocity_output_tex = temporal_velocity_output_tex.view(),
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaComputePush &push, TaaTaskInfo const &info) {
                push.input_tex_size = info.input_tex_size;
                push.output_tex_size = info.output_tex_size;
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
            },
            .info = {
                .thread_count = gpu_context.output_resolution,
                .input_tex_size = i_extent,
                .output_tex_size = o_extent,
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "taa", .task_image_id = this_frame_output_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        return daxa::TaskImageView{this_frame_output_img};
    }
};

#endif

#endif // RENDERER_KAJIYA_TAA_INL
