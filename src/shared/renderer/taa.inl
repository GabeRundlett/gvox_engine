#pragma once

#include <shared/core.inl>

#define TAA_WG_SIZE_X 16
#define TAA_WG_SIZE_Y 8

#if TAA_REPROJECT_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(TaaReprojectComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(history_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reprojection_map, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(depth_image, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reprojected_history_img, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(closest_velocity_img, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif
#if TAA_FILTER_INPUT_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(TaaFilterInputComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(input_image, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(depth_image, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(filtered_input_img, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(filtered_input_deviation_img, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif
#if TAA_FILTER_HISTORY_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(TaaFilterHistoryComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(reprojected_history_img, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(filtered_history_img, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif
#if TAA_INPUT_PROB_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(TaaInputProbComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(input_image, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(filtered_input_img, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(filtered_input_deviation_img, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reprojected_history_img, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(filtered_history_img, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reprojection_map, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(depth_image, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(smooth_var_history_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(velocity_history_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(input_prob_img, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif
#if TAA_PROB_FILTER_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(TaaProbFilterComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(input_prob_img, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(prob_filtered1_img, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif
#if TAA_PROB_FILTER2_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(TaaProbFilter2ComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(prob_filtered1_img, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(prob_filtered2_img, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif
#if TAA_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(TaaComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(input_image, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reprojected_history_img, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reprojection_map, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(closest_velocity_img, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(velocity_history_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(depth_image, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(smooth_var_history_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(input_prob_img, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(temporal_output_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(this_frame_output_img, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(smooth_var_output_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(temporal_velocity_output_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif

struct TaaPush {
    f32vec4 input_tex_size;
    f32vec4 output_tex_size;
};

#if defined(__cplusplus)

inline void taa_compile_compute_pipeline(daxa::PipelineManager &pipeline_manager, char const *const name, std::shared_ptr<daxa::ComputePipeline> &pipeline) {
    auto compile_result = pipeline_manager.add_compute_pipeline({
        .shader_info = {
            .source = daxa::ShaderFile{"taa.comp.glsl"},
            .compile_options = {.defines = {{name, "1"}}},
        },
        .push_constant_size = sizeof(TaaPush),
        .name = std::string("taa_") + name,
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

#define TAA_DECL_TASK_STATE(Name, NAME)                                                                                                                 \
    struct Name##ComputeTaskState {                                                                                                                     \
        std::shared_ptr<daxa::ComputePipeline> pipeline;                                                                                                \
        Name##ComputeTaskState(daxa::PipelineManager &pipeline_manager) { taa_compile_compute_pipeline(pipeline_manager, #NAME "_COMPUTE", pipeline); } \
        auto pipeline_is_valid() -> bool { return pipeline && pipeline->is_valid(); }                                                                   \
        void record_commands(daxa::CommandList &cmd_list, u32vec2 thread_count, TaaPush const &push) {                                                  \
            if (!pipeline_is_valid())                                                                                                                   \
                return;                                                                                                                                 \
            cmd_list.set_pipeline(*pipeline);                                                                                                           \
            cmd_list.push_constant(push);                                                                                                               \
            cmd_list.dispatch((thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y);          \
        }                                                                                                                                               \
    };                                                                                                                                                  \
    struct Name##ComputeTask : Name##ComputeUses {                                                                                                      \
        Name##ComputeTaskState *state;                                                                                                                  \
        u32vec2 thread_count;                                                                                                                           \
        TaaPush push;                                                                                                                                   \
        void callback(daxa::TaskInterface const &ti) {                                                                                                  \
            auto cmd_list = ti.get_command_list();                                                                                                      \
            cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());                                                                             \
            state->record_commands(cmd_list, thread_count, push);                                                                                       \
        }                                                                                                                                               \
    }

TAA_DECL_TASK_STATE(TaaReproject, TAA_REPROJECT);
TAA_DECL_TASK_STATE(TaaFilterInput, TAA_FILTER_INPUT);
TAA_DECL_TASK_STATE(TaaFilterHistory, TAA_FILTER_HISTORY);
TAA_DECL_TASK_STATE(TaaInputProb, TAA_INPUT_PROB);
TAA_DECL_TASK_STATE(TaaProbFilter, TAA_PROB_FILTER);
TAA_DECL_TASK_STATE(TaaProbFilter2, TAA_PROB_FILTER2);
TAA_DECL_TASK_STATE(Taa, TAA);

struct TaaRenderer {
    PingPongImage ping_pong_taa_col_image;
    PingPongImage ping_pong_taa_vel_image;
    PingPongImage ping_pong_smooth_var_image;

    TaaReprojectComputeTaskState taa_reproject_task_state;
    TaaFilterInputComputeTaskState taa_filter_input_task_state;
    TaaFilterHistoryComputeTaskState taa_filter_history_task_state;
    TaaInputProbComputeTaskState taa_input_prob_task_state;
    TaaProbFilterComputeTaskState taa_prob_filter_task_state;
    TaaProbFilter2ComputeTaskState taa_prob_filter2_task_state;
    TaaComputeTaskState taa_task_state;

    TaaRenderer(daxa::PipelineManager &pipeline_manager)
        : taa_reproject_task_state{pipeline_manager},
          taa_filter_input_task_state{pipeline_manager},
          taa_filter_history_task_state{pipeline_manager},
          taa_input_prob_task_state{pipeline_manager},
          taa_prob_filter_task_state{pipeline_manager},
          taa_prob_filter2_task_state{pipeline_manager},
          taa_task_state{pipeline_manager} {
    }

    void next_frame() {
        ping_pong_taa_col_image.task_resources.output_image.swap_images(ping_pong_taa_col_image.task_resources.history_image);
        ping_pong_taa_vel_image.task_resources.output_image.swap_images(ping_pong_taa_vel_image.task_resources.history_image);
        ping_pong_smooth_var_image.task_resources.output_image.swap_images(ping_pong_smooth_var_image.task_resources.history_image);
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

        auto i_extent_inv_extent = f32vec4(static_cast<f32>(record_ctx.render_resolution.x), static_cast<f32>(record_ctx.render_resolution.y), 0.0f, 0.0f);
        i_extent_inv_extent.z = 1.0f / i_extent_inv_extent.x;
        i_extent_inv_extent.w = 1.0f / i_extent_inv_extent.y;

        auto o_extent_inv_extent = f32vec4(static_cast<f32>(record_ctx.output_resolution.x), static_cast<f32>(record_ctx.output_resolution.y), 0.0f, 0.0f);
        o_extent_inv_extent.z = 1.0f / o_extent_inv_extent.x;
        o_extent_inv_extent.w = 1.0f / o_extent_inv_extent.y;

        record_ctx.task_graph.add_task(TaaReprojectComputeTask{
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .globals = record_ctx.task_globals_buffer,

                    .history_tex = history_tex,
                    .reprojection_map = reprojection_map,
                    .depth_image = depth_image,

                    .reprojected_history_img = reprojected_history_img,
                    .closest_velocity_img = closest_velocity_img,
                },
            },
            &taa_reproject_task_state,
            record_ctx.output_resolution,
            TaaPush{
                i_extent_inv_extent,
                o_extent_inv_extent,
            },
        });

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

        record_ctx.task_graph.add_task(TaaFilterInputComputeTask{
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .globals = record_ctx.task_globals_buffer,

                    .input_image = input_image,
                    .depth_image = depth_image,

                    .filtered_input_img = filtered_input_img,
                    .filtered_input_deviation_img = filtered_input_deviation_img,
                },
            },
            &taa_filter_input_task_state,
            record_ctx.render_resolution,
            TaaPush{
                i_extent_inv_extent,
                o_extent_inv_extent,
            },
        });

        auto filtered_history_img = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "filtered_history_img",
        });

        record_ctx.task_graph.add_task(TaaFilterHistoryComputeTask{
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .globals = record_ctx.task_globals_buffer,

                    .reprojected_history_img = reprojected_history_img,

                    .filtered_history_img = filtered_history_img,
                },
            },
            &taa_filter_history_task_state,
            record_ctx.render_resolution,
            TaaPush{
                o_extent_inv_extent,
                i_extent_inv_extent,
            },
        });

        auto input_prob_img = [&]() {
            auto input_prob_img = record_ctx.task_graph.create_transient_image({
                .format = daxa::Format::R16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .name = "input_prob_img",
            });
            record_ctx.task_graph.add_task(TaaInputProbComputeTask{
                {
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
                },
                &taa_input_prob_task_state,
                record_ctx.render_resolution,
                TaaPush{
                    i_extent_inv_extent,
                    o_extent_inv_extent,
                },
            });

            auto prob_filtered1_img = record_ctx.task_graph.create_transient_image({
                .format = daxa::Format::R16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .name = "prob_filtered1_img",
            });
            record_ctx.task_graph.add_task(TaaProbFilterComputeTask{
                {
                    .uses = {
                        .gpu_input = record_ctx.task_input_buffer,
                        .globals = record_ctx.task_globals_buffer,

                        .input_prob_img = input_prob_img,

                        .prob_filtered1_img = prob_filtered1_img,
                    },
                },
                &taa_prob_filter_task_state,
                record_ctx.render_resolution,
                TaaPush{
                    i_extent_inv_extent,
                    o_extent_inv_extent,
                },
            });

            auto prob_filtered2_img = record_ctx.task_graph.create_transient_image({
                .format = daxa::Format::R16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .name = "prob_filtered2_img",
            });
            record_ctx.task_graph.add_task(TaaProbFilter2ComputeTask{
                {
                    .uses = {
                        .gpu_input = record_ctx.task_input_buffer,
                        .globals = record_ctx.task_globals_buffer,

                        .prob_filtered1_img = prob_filtered1_img,

                        .prob_filtered2_img = prob_filtered2_img,
                    },
                },
                &taa_prob_filter2_task_state,
                record_ctx.render_resolution,
                TaaPush{
                    i_extent_inv_extent,
                    o_extent_inv_extent,
                },
            });

            return prob_filtered2_img;
        }();

        auto this_frame_output_img = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {record_ctx.output_resolution.x, record_ctx.output_resolution.y, 1},
            .name = "this_frame_output_img",
        });

        record_ctx.task_graph.add_task(TaaComputeTask{
            {
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
            },
            &taa_task_state,
            record_ctx.output_resolution,
            TaaPush{
                i_extent_inv_extent,
                o_extent_inv_extent,
            },
        });

        return daxa::TaskImageView{this_frame_output_img};
    }
};

#endif
