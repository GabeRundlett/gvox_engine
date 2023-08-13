#pragma once

#include <shared/core.inl>

#if RTDGI_TEMPORAL_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(RtdgiTemporalComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(input_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(history_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(variance_history_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reprojection_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(rt_history_invalidity_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(output_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(history_output_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(variance_history_output_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif
#if RTDGI_SPATIAL_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(RtdgiSpatialComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(input_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(depth_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(ssao_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(geometric_normal_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(output_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif
#if RTDGI_REPROJECT_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(RtdgiReprojectComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(input_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reprojection_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(output_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif
#if RTDGI_VALIDATE_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(RtdgiValidateComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(half_view_normal_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(depth_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reprojected_gi_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reservoir_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(reservoir_ray_history_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reprojection_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
// DAXA_TASK_USE_IMAGE(input_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
// DAXA_TASK_USE_IMAGE(output_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(irradiance_history_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(ray_orig_history_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(rt_history_invalidity_out_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif
#if RTDGI_TRACE_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(RtdgiTraceComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(blue_noise_vec2, REGULAR_3D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(half_view_normal_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(depth_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reprojected_gi_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reprojection_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
// DAXA_TASK_USE_IMAGE(input_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
// DAXA_TASK_USE_IMAGE(output_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(ray_orig_history_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(candidate_irradiance_out_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(candidate_normal_out_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(candidate_hit_out_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(rt_history_invalidity_in_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(rt_history_invalidity_out_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif
#if RTDGI_VALIDITY_INTEGRATE_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(RtdgiValidityIntegrateComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(input_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(history_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reprojection_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(half_view_normal_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(half_depth_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(output_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif
#if RTDGI_RESTIR_TEMPORAL_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(RtdgiRestirTemporalComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(half_view_normal_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(depth_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(candidate_radiance_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(candidate_normal_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(candidate_hit_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(radiance_history_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(ray_orig_history_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(ray_history_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reservoir_history_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reprojection_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(hit_normal_history_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(candidate_history_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(rt_invalidity_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(radiance_out_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(ray_orig_output_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(ray_output_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(hit_normal_output_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(reservoir_out_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(candidate_out_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(temporal_reservoir_packed_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif
#if RTDGI_RESTIR_SPATIAL_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(RtdgiRestirSpatialComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(reservoir_input_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(bounced_radiance_input_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(half_view_normal_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(half_depth_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(depth_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(half_ssao_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(temporal_reservoir_packed_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reprojected_gi_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reservoir_output_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(bounced_radiance_output_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif
#if RTDGI_RESTIR_RESOLVE_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(RtdgiRestirResolveComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(blue_noise_vec2, REGULAR_3D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(radiance_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reservoir_input_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(gbuffer_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(depth_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(half_view_normal_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(half_depth_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(ssao_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(candidate_radiance_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(candidate_hit_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(temporal_reservoir_packed_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(bounced_radiance_input_tex, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(irradiance_output_tex, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif

struct RtdgiPush {
    f32vec4 output_tex_size;
};
struct RtdgiTemporalPush {
    f32vec4 output_tex_size;
    f32vec4 gbuffer_tex_size;
};
struct RtdgiRestirTemporalPush {
    f32vec4 gbuffer_tex_size;
};
struct RtdgiRestirSpatialPush {
    f32vec4 gbuffer_tex_size;
    f32vec4 output_tex_size;
    u32 spatial_reuse_pass_idx;
    // Only done in the last spatial resampling pass
    u32 perform_occlusion_raymarch;
    u32 occlusion_raymarch_importance_only;
};
struct RtdgiRestirResolvePush {
    f32vec4 gbuffer_tex_size;
    f32vec4 output_tex_size;
};

#define ENABLE_RESTIR 0

#if defined(__cplusplus)
#include <shared/renderer/downscale.inl>

template <typename PushT>
inline void rtdgi_compile_compute_pipeline(daxa::PipelineManager &pipeline_manager, char const *const name, std::shared_ptr<daxa::ComputePipeline> &pipeline) {
    auto compile_result = pipeline_manager.add_compute_pipeline({
        .shader_info = {
            .source = daxa::ShaderFile{"diffuse_gi.comp.glsl"},
            .compile_options = {.defines = {{name, "1"}}},
        },
        .push_constant_size = sizeof(PushT),
        .name = std::string("rtdgi_") + name,
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

#define RTDGI_DECL_TASK_STATE(Name, NAME, PushType)                                                                                                                 \
    struct Name##ComputeTaskState {                                                                                                                                 \
        std::shared_ptr<daxa::ComputePipeline> pipeline;                                                                                                            \
        Name##ComputeTaskState(daxa::PipelineManager &pipeline_manager) { rtdgi_compile_compute_pipeline<PushType>(pipeline_manager, #NAME "_COMPUTE", pipeline); } \
        auto pipeline_is_valid() -> bool { return pipeline && pipeline->is_valid(); }                                                                               \
        void record_commands(daxa::CommandList &cmd_list, u32vec2 thread_count, PushType const &push) {                                                             \
            if (!pipeline_is_valid())                                                                                                                               \
                return;                                                                                                                                             \
            cmd_list.set_pipeline(*pipeline);                                                                                                                       \
            cmd_list.push_constant(push);                                                                                                                           \
            cmd_list.dispatch((thread_count.x + 7) / 8, (thread_count.y + 7) / 8);                                                                                  \
        }                                                                                                                                                           \
    };                                                                                                                                                              \
    struct Name##ComputeTask : Name##ComputeUses {                                                                                                                  \
        Name##ComputeTaskState *state;                                                                                                                              \
        u32vec2 thread_count;                                                                                                                                       \
        PushType push;                                                                                                                                              \
        void callback(daxa::TaskInterface const &ti) {                                                                                                              \
            auto cmd_list = ti.get_command_list();                                                                                                                  \
            cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());                                                                                         \
            state->record_commands(cmd_list, thread_count, push);                                                                                                   \
        }                                                                                                                                                           \
    }

RTDGI_DECL_TASK_STATE(RtdgiTemporal, RTDGI_TEMPORAL, RtdgiTemporalPush);
RTDGI_DECL_TASK_STATE(RtdgiSpatial, RTDGI_SPATIAL, RtdgiPush);
RTDGI_DECL_TASK_STATE(RtdgiReproject, RTDGI_REPROJECT, RtdgiPush);
RTDGI_DECL_TASK_STATE(RtdgiValidate, RTDGI_VALIDATE, RtdgiRestirTemporalPush);
RTDGI_DECL_TASK_STATE(RtdgiTrace, RTDGI_TRACE, RtdgiRestirTemporalPush);
RTDGI_DECL_TASK_STATE(RtdgiValidityIntegrate, RTDGI_VALIDITY_INTEGRATE, RtdgiRestirResolvePush);
RTDGI_DECL_TASK_STATE(RtdgiRestirTemporal, RTDGI_RESTIR_TEMPORAL, RtdgiRestirTemporalPush);
RTDGI_DECL_TASK_STATE(RtdgiRestirSpatial, RTDGI_RESTIR_SPATIAL, RtdgiRestirSpatialPush);
RTDGI_DECL_TASK_STATE(RtdgiRestirResolve, RTDGI_RESTIR_RESOLVE, RtdgiRestirResolvePush);

static inline constexpr u32 SPATIAL_REUSE_PASS_COUNT = 2;

struct DiffuseGiRenderer {
    PingPongImage temporal_radiance_tex;
    PingPongImage temporal_ray_orig_tex;
    PingPongImage temporal_ray_tex;
    PingPongImage temporal_reservoir_tex;
    PingPongImage temporal_candidate_tex;
    PingPongImage temporal_invalidity_tex;
    PingPongImage temporal2_tex;
    PingPongImage temporal2_variance_tex;
    PingPongImage temporal_hit_normal_tex;

    DownscaleComputeTaskState downscale_ssao_task_state;
    RtdgiTemporalComputeTaskState rtdgi_temporal_task_state;
    RtdgiSpatialComputeTaskState rtdgi_spatial_task_state;
    RtdgiReprojectComputeTaskState rtdgi_reproject_task_state;
    RtdgiValidateComputeTaskState rtdgi_validate_task_state;
    RtdgiTraceComputeTaskState rtdgi_trace_task_state;
    RtdgiValidityIntegrateComputeTaskState rtdgi_validity_integrate_task_state;
    RtdgiRestirTemporalComputeTaskState rtdgi_restir_temporal_task_state;
    RtdgiRestirSpatialComputeTaskState rtdgi_restir_spatial_task_state;
    RtdgiRestirResolveComputeTaskState rtdgi_restir_resolve_task_state;

    f32vec4 scaled_extent_inv_extent;
    f32vec4 extent_inv_extent;
    u32vec2 shading_resolution;

    DiffuseGiRenderer(daxa::PipelineManager &pipeline_manager)
        : downscale_ssao_task_state{pipeline_manager, {{"DOWNSCALE_SSAO", "1"}}},
          rtdgi_temporal_task_state{pipeline_manager},
          rtdgi_spatial_task_state{pipeline_manager},
          rtdgi_reproject_task_state{pipeline_manager},
          rtdgi_validate_task_state{pipeline_manager},
          rtdgi_trace_task_state{pipeline_manager},
          rtdgi_validity_integrate_task_state{pipeline_manager},
          rtdgi_restir_temporal_task_state{pipeline_manager},
          rtdgi_restir_spatial_task_state{pipeline_manager},
          rtdgi_restir_resolve_task_state{pipeline_manager} {
    }

    void next_frame() {
        temporal_radiance_tex.task_resources.output_image.swap_images(temporal_radiance_tex.task_resources.history_image);
        temporal_ray_orig_tex.task_resources.output_image.swap_images(temporal_ray_orig_tex.task_resources.history_image);
        temporal_ray_tex.task_resources.output_image.swap_images(temporal_ray_tex.task_resources.history_image);
        temporal_reservoir_tex.task_resources.output_image.swap_images(temporal_reservoir_tex.task_resources.history_image);
        temporal_candidate_tex.task_resources.output_image.swap_images(temporal_candidate_tex.task_resources.history_image);
        temporal_invalidity_tex.task_resources.output_image.swap_images(temporal_invalidity_tex.task_resources.history_image);
        temporal2_tex.task_resources.output_image.swap_images(temporal2_tex.task_resources.history_image);
#if ENABLE_RESTIR
        temporal2_variance_tex.task_resources.output_image.swap_images(temporal2_variance_tex.task_resources.history_image);
#endif
        temporal_hit_normal_tex.task_resources.output_image.swap_images(temporal_hit_normal_tex.task_resources.history_image);
    }

    auto temporal(
        RecordContext &record_ctx,
        daxa::TaskImageView input_color,
        daxa::TaskImageView reprojection_map,
        daxa::TaskImageView reprojected_history_tex,
        daxa::TaskImageView rt_history_invalidity_tex,
        daxa::TaskImageView temporal_output_tex)
        -> daxa::TaskImageView {
        auto [temporal_variance_output_tex, variance_history_tex] = temporal2_variance_tex.get(
            record_ctx.device,
            {
                .format = daxa::Format::R16G16_SFLOAT,
                .size = {shading_resolution.x, shading_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                .name = "temporal2_variance",
            });
        record_ctx.task_graph.use_persistent_image(temporal_variance_output_tex);
        record_ctx.task_graph.use_persistent_image(variance_history_tex);

        auto temporal_filtered_tex = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "temporal_filtered_tex",
        });

        record_ctx.task_graph.add_task(RtdgiTemporalComputeTask{
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .globals = record_ctx.task_globals_buffer,

                    .input_tex = input_color,
                    .history_tex = reprojected_history_tex,
                    .variance_history_tex = variance_history_tex,
                    .reprojection_tex = reprojection_map,
                    .rt_history_invalidity_tex = rt_history_invalidity_tex,

                    .output_tex = temporal_filtered_tex,
                    .history_output_tex = temporal_output_tex,
                    .variance_history_output_tex = temporal_variance_output_tex,
                },
            },
            &rtdgi_temporal_task_state,
            record_ctx.render_resolution,
            RtdgiTemporalPush{
                extent_inv_extent,
                extent_inv_extent,
            },
        });

        return temporal_filtered_tex;
    }

    auto spatial(
        RecordContext &record_ctx,
        daxa::TaskImageView input_color,
        GbufferDepth &gbuffer_depth,
        daxa::TaskImageView ssao_tex)
        -> daxa::TaskImageView {
        auto spatial_filtered_tex = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "spatial_filtered_tex",
        });

        record_ctx.task_graph.add_task(RtdgiSpatialComputeTask{
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .globals = record_ctx.task_globals_buffer,

                    .input_tex = input_color,
                    .depth_tex = gbuffer_depth.depth.task_resources.output_image,
                    .ssao_tex = ssao_tex,
                    .geometric_normal_tex = gbuffer_depth.geometric_normal,

                    .output_tex = spatial_filtered_tex,
                },
            },
            &rtdgi_spatial_task_state,
            record_ctx.render_resolution,
            RtdgiPush{
                extent_inv_extent,
            },
        });

        return spatial_filtered_tex;
    }

    struct ReprojectedRtdgi {
        daxa::TaskImageView history_tex;
        daxa::TaskImageView temporal_output_tex;
    };

    auto reproject(
        RecordContext &record_ctx,
        daxa::TaskImageView reprojection_map)
        -> ReprojectedRtdgi {
        shading_resolution = u32vec2{record_ctx.render_resolution.x / SHADING_SCL, record_ctx.render_resolution.y / SHADING_SCL};

        extent_inv_extent = f32vec4(static_cast<f32>(record_ctx.render_resolution.x), static_cast<f32>(record_ctx.render_resolution.y), 0.0f, 0.0f);
        extent_inv_extent.z = 1.0f / extent_inv_extent.x;
        extent_inv_extent.w = 1.0f / extent_inv_extent.y;

        scaled_extent_inv_extent = f32vec4(static_cast<f32>(shading_resolution.x), static_cast<f32>(shading_resolution.y), 0.0f, 0.0f);
        scaled_extent_inv_extent.z = 1.0f / scaled_extent_inv_extent.x;
        scaled_extent_inv_extent.w = 1.0f / scaled_extent_inv_extent.y;

        temporal2_tex = PingPongImage{};
        auto [temporal_output_tex, history_tex] = temporal2_tex.get(
            record_ctx.device,
            {
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                .name = "temporal2",
            });
        record_ctx.task_graph.use_persistent_image(temporal_output_tex);
        record_ctx.task_graph.use_persistent_image(history_tex);

        auto reprojected_history_tex = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "reprojected_history_tex",
        });

        record_ctx.task_graph.add_task(RtdgiReprojectComputeTask{
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .globals = record_ctx.task_globals_buffer,

                    .input_tex = history_tex,
                    .reprojection_tex = reprojection_map,

                    .output_tex = reprojected_history_tex,
                },
            },
            &rtdgi_reproject_task_state,
            record_ctx.render_resolution,
            RtdgiPush{
                extent_inv_extent,
            },
        });

        return {
            reprojected_history_tex,
            temporal_output_tex,
        };
    }

    auto render(
        RecordContext &record_ctx,
        GbufferDepth &gbuffer_depth,
        ReprojectedRtdgi reprojected_rtdgi,
        daxa::TaskImageView reprojection_map,
        VoxelWorld::Buffers &voxel_buffers,
        daxa::TaskImageView ssao_tex)
        -> daxa::TaskImageView {
        temporal_radiance_tex = PingPongImage{};
        temporal_ray_orig_tex = PingPongImage{};
        temporal_ray_tex = PingPongImage{};
        temporal_reservoir_tex = PingPongImage{};
        temporal_candidate_tex = PingPongImage{};
        temporal_invalidity_tex = PingPongImage{};
        temporal2_variance_tex = PingPongImage{};
        temporal_hit_normal_tex = PingPongImage{};

        auto half_ssao_tex = extract_downscaled_ssao(record_ctx, downscale_ssao_task_state, ssao_tex);

        auto [hit_normal_output_tex, hit_normal_history_tex] = temporal_hit_normal_tex.get(
            record_ctx.device,
            {
                .format = daxa::Format::R8G8B8A8_UNORM,
                .size = {shading_resolution.x, shading_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                .name = "temporal_hit_normal",
            });
        auto [candidate_output_tex, candidate_history_tex] = temporal_candidate_tex.get(
            record_ctx.device,
            {
                .format = daxa::Format::R8G8B8A8_UNORM,
                .size = {shading_resolution.x, shading_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                .name = "temporal_candidate",
            });
        auto [invalidity_output_tex, invalidity_history_tex] = temporal_invalidity_tex.get(
            record_ctx.device,
            {
                .format = daxa::Format::R16G16_SFLOAT,
                .size = {shading_resolution.x, shading_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                .name = "temporal_invalidity",
            });
        record_ctx.task_graph.use_persistent_image(hit_normal_output_tex);
        record_ctx.task_graph.use_persistent_image(hit_normal_history_tex);
        record_ctx.task_graph.use_persistent_image(candidate_output_tex);
        record_ctx.task_graph.use_persistent_image(candidate_history_tex);
        record_ctx.task_graph.use_persistent_image(invalidity_output_tex);
        record_ctx.task_graph.use_persistent_image(invalidity_history_tex);

        auto candidate_radiance_tex = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {shading_resolution.x, shading_resolution.y, 1},
            .name = "candidate_radiance_tex",
        });
        auto candidate_normal_tex = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R8G8B8A8_SNORM,
            .size = {shading_resolution.x, shading_resolution.y, 1},
            .name = "candidate_normal_tex",
        });
        auto candidate_hit_tex = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {shading_resolution.x, shading_resolution.y, 1},
            .name = "candidate_hit_tex",
        });
        auto temporal_reservoir_packed_tex = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R32G32B32A32_UINT,
            .size = {shading_resolution.x, shading_resolution.y, 1},
            .name = "temporal_reservoir_packed_tex",
        });
        auto half_depth_tex = gbuffer_depth.get_downscaled_depth(record_ctx);

        auto debug_tex = daxa::TaskImageView{};

        auto [radiance_tex, local_temporal_reservoir_tex] = [&]() -> std::array<daxa::TaskImageView, 2> {
            auto [radiance_output_tex, radiance_history_tex] = temporal_radiance_tex.get(
                record_ctx.device,
                {
                    .format = daxa::Format::R16G16B16A16_SFLOAT,
                    .size = {shading_resolution.x, shading_resolution.y, 1},
                    .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                    .name = "temporal_radiance",
                });
            auto [ray_orig_output_tex, ray_orig_history_tex] = temporal_ray_orig_tex.get(
                record_ctx.device,
                {
                    .format = daxa::Format::R32G32B32A32_SFLOAT,
                    .size = {shading_resolution.x, shading_resolution.y, 1},
                    .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                    .name = "temporal_ray_orig",
                });
            auto [ray_output_tex, ray_history_tex] = temporal_ray_tex.get(
                record_ctx.device,
                {
                    .format = daxa::Format::R16G16B16A16_SFLOAT,
                    .size = {shading_resolution.x, shading_resolution.y, 1},
                    .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                    .name = "temporal_ray",
                });
            auto [reservoir_output_tex, reservoir_history_tex] = temporal_reservoir_tex.get(
                record_ctx.device,
                {
                    .format = daxa::Format::R32G32_UINT,
                    .size = {shading_resolution.x, shading_resolution.y, 1},
                    .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                    .name = "temporal_reservoir",
                });

            record_ctx.task_graph.use_persistent_image(radiance_output_tex);
            record_ctx.task_graph.use_persistent_image(radiance_history_tex);
            record_ctx.task_graph.use_persistent_image(ray_orig_output_tex);
            record_ctx.task_graph.use_persistent_image(ray_orig_history_tex);
            record_ctx.task_graph.use_persistent_image(ray_output_tex);
            record_ctx.task_graph.use_persistent_image(ray_history_tex);
            record_ctx.task_graph.use_persistent_image(reservoir_output_tex);
            record_ctx.task_graph.use_persistent_image(reservoir_history_tex);

            auto half_view_normal_tex = gbuffer_depth.get_downscaled_view_normal(record_ctx);

            auto rt_history_validity_pre_input_tex = record_ctx.task_graph.create_transient_image({
                .format = daxa::Format::R8_UNORM,
                .size = {shading_resolution.x, shading_resolution.y, 1},
                .name = "rt_history_validity_pre_input_tex",
            });

#if ENABLE_RESTIR
            record_ctx.task_graph.add_task(RtdgiValidateComputeTask{
                {
                    .uses = {
                        .gpu_input = record_ctx.task_input_buffer,
                        .globals = record_ctx.task_globals_buffer,
                        VOXELS_BUFFER_USES_ASSIGN(voxel_buffers),

                        .half_view_normal_tex = half_view_normal_tex,
                        .depth_tex = gbuffer_depth.depth.task_resources.output_image,
                        .reprojected_gi_tex = reprojected_rtdgi.history_tex,
                        .reservoir_tex = reservoir_history_tex,
                        .reservoir_ray_history_tex = ray_history_tex,
                        .reprojection_tex = reprojection_map,

                        // .irradiance_cache = irradiance_cache,
                        // .wrc = wrc,
                        // .sky_cube = sky_cube,

                        .irradiance_history_tex = radiance_history_tex,
                        .ray_orig_history_tex = ray_orig_history_tex,
                        .rt_history_invalidity_out_tex = rt_history_validity_pre_input_tex,
                    },
                },
                &rtdgi_validate_task_state,
                shading_resolution,
                RtdgiRestirTemporalPush{
                    extent_inv_extent,
                },
            });
#endif

            auto rt_history_validity_input_tex = record_ctx.task_graph.create_transient_image({
                .format = daxa::Format::R8_UNORM,
                .size = {shading_resolution.x, shading_resolution.y, 1},
                .name = "rt_history_validity_input_tex",
            });

            record_ctx.task_graph.add_task(RtdgiTraceComputeTask{
                {
                    .uses = {
                        .gpu_input = record_ctx.task_input_buffer,
                        .globals = record_ctx.task_globals_buffer,
                        VOXELS_BUFFER_USES_ASSIGN(voxel_buffers),
                        .blue_noise_vec2 = record_ctx.task_blue_noise_vec2_image,

                        .half_view_normal_tex = half_view_normal_tex,
                        .depth_tex = gbuffer_depth.depth.task_resources.output_image,
                        .reprojected_gi_tex = reprojected_rtdgi.history_tex,
                        .reprojection_tex = reprojection_map,

                        // .irradiance_cache = irradiance_cache,
                        // .wrc = wrc,
                        // .sky_cube_tex = sky_cube,

                        .ray_orig_history_tex = ray_orig_history_tex,
                        .candidate_irradiance_out_tex = candidate_radiance_tex,
                        .candidate_normal_out_tex = candidate_normal_tex,
                        .candidate_hit_out_tex = candidate_hit_tex,
                        .rt_history_invalidity_in_tex = rt_history_validity_pre_input_tex,
                        .rt_history_invalidity_out_tex = rt_history_validity_input_tex,
                    },
                },
                &rtdgi_trace_task_state,
                shading_resolution,
                RtdgiRestirTemporalPush{
                    extent_inv_extent,
                },
            });

            debug_tex = candidate_radiance_tex;

#if ENABLE_RESTIR
            record_ctx.task_graph.add_task(RtdgiValidityIntegrateComputeTask{
                {
                    .uses = {
                        .gpu_input = record_ctx.task_input_buffer,
                        .globals = record_ctx.task_globals_buffer,

                        .input_tex = rt_history_validity_input_tex,
                        .history_tex = invalidity_history_tex,
                        .reprojection_tex = reprojection_map,
                        .half_view_normal_tex = half_view_normal_tex,
                        .half_depth_tex = half_depth_tex,

                        .output_tex = invalidity_output_tex,
                    },
                },
                &rtdgi_validity_integrate_task_state,
                shading_resolution,
                RtdgiRestirResolvePush{
                    extent_inv_extent,
                    scaled_extent_inv_extent,
                },
            });

            record_ctx.task_graph.add_task(RtdgiRestirTemporalComputeTask{
                {
                    .uses = {
                        .gpu_input = record_ctx.task_input_buffer,
                        .globals = record_ctx.task_globals_buffer,

                        .half_view_normal_tex = half_view_normal_tex,
                        .depth_tex = gbuffer_depth.depth.task_resources.output_image,
                        .candidate_radiance_tex = candidate_radiance_tex,
                        .candidate_normal_tex = candidate_normal_tex,
                        .candidate_hit_tex = candidate_hit_tex,
                        .radiance_history_tex = radiance_history_tex,
                        .ray_orig_history_tex = ray_orig_history_tex,
                        .ray_history_tex = ray_history_tex,
                        .reservoir_history_tex = reservoir_history_tex,
                        .reprojection_tex = reprojection_map,
                        .hit_normal_history_tex = hit_normal_history_tex,
                        .candidate_history_tex = candidate_history_tex,
                        .rt_invalidity_tex = invalidity_output_tex,

                        .radiance_out_tex = radiance_output_tex,
                        .ray_orig_output_tex = ray_orig_output_tex,
                        .ray_output_tex = ray_output_tex,
                        .hit_normal_output_tex = hit_normal_output_tex,
                        .reservoir_out_tex = reservoir_output_tex,
                        .candidate_out_tex = candidate_output_tex,
                        .temporal_reservoir_packed_tex = temporal_reservoir_packed_tex,
                    },
                },
                &rtdgi_restir_temporal_task_state,
                shading_resolution,
                RtdgiRestirTemporalPush{
                    extent_inv_extent,
                },
            });
#endif

            return {radiance_output_tex, reservoir_output_tex};
        }();

#if ENABLE_RESTIR
        auto irradiance_tex = [&]() -> daxa::TaskImageView {
            auto half_view_normal_tex = gbuffer_depth.get_downscaled_view_normal(record_ctx);
            auto reservoir_output_tex0 = record_ctx.task_graph.create_transient_image({
                .format = daxa::Format::R32G32_UINT,
                .size = {shading_resolution.x, shading_resolution.y, 1},
                .name = "reservoir_output_tex0",
            });
            auto reservoir_output_tex1 = record_ctx.task_graph.create_transient_image({
                .format = daxa::Format::R32G32_UINT,
                .size = {shading_resolution.x, shading_resolution.y, 1},
                .name = "reservoir_output_tex1",
            });
            auto bounced_radiance_output_tex0 = record_ctx.task_graph.create_transient_image({
                .format = daxa::Format::B10G11R11_UFLOAT_PACK32,
                .size = {shading_resolution.x, shading_resolution.y, 1},
                .name = "bounced_radiance_output_tex0",
            });
            auto bounced_radiance_output_tex1 = record_ctx.task_graph.create_transient_image({
                .format = daxa::Format::B10G11R11_UFLOAT_PACK32,
                .size = {shading_resolution.x, shading_resolution.y, 1},
                .name = "bounced_radiance_output_tex1",
            });

            auto reservoir_input_tex = local_temporal_reservoir_tex;
            auto bounced_radiance_input_tex = radiance_tex;

            for (u32 spatial_reuse_pass_idx = 0; spatial_reuse_pass_idx < SPATIAL_REUSE_PASS_COUNT; ++spatial_reuse_pass_idx) {
                bool perform_occlusion_raymarch = (spatial_reuse_pass_idx + 1 == SPATIAL_REUSE_PASS_COUNT);
                bool occlusion_raymarch_importance_only = false; // self.use_raytraced_reservoir_visibility

                record_ctx.task_graph.add_task(RtdgiRestirSpatialComputeTask{
                    {
                        .uses = {
                            .gpu_input = record_ctx.task_input_buffer,
                            .globals = record_ctx.task_globals_buffer,

                            .reservoir_input_tex = reservoir_input_tex,
                            .bounced_radiance_input_tex = bounced_radiance_input_tex,
                            .half_view_normal_tex = half_view_normal_tex,
                            .half_depth_tex = half_depth_tex,
                            .depth_tex = gbuffer_depth.depth.task_resources.output_image,
                            .half_ssao_tex = half_ssao_tex,
                            .temporal_reservoir_packed_tex = temporal_reservoir_packed_tex,
                            .reprojected_gi_tex = reprojected_rtdgi.history_tex,

                            .reservoir_output_tex = reservoir_output_tex0,
                            .bounced_radiance_output_tex = bounced_radiance_output_tex0,
                        },
                    },
                    &rtdgi_restir_spatial_task_state,
                    shading_resolution,
                    RtdgiRestirSpatialPush{
                        extent_inv_extent,
                        scaled_extent_inv_extent,
                        spatial_reuse_pass_idx,
                        perform_occlusion_raymarch,
                        occlusion_raymarch_importance_only,
                    },
                });

                std::swap(reservoir_output_tex0, reservoir_output_tex1);
                std::swap(bounced_radiance_output_tex0, bounced_radiance_output_tex1);

                reservoir_input_tex = reservoir_output_tex1;
                bounced_radiance_input_tex = bounced_radiance_output_tex1;
            }

            if (false) { // self.use_raytraced_reservoir_visibility
                // ...
            }

            auto irradiance_output_tex = record_ctx.task_graph.create_transient_image({
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .name = "irradiance_output_tex",
            });

            record_ctx.task_graph.add_task(RtdgiRestirResolveComputeTask{
                {
                    .uses = {
                        .gpu_input = record_ctx.task_input_buffer,
                        .globals = record_ctx.task_globals_buffer,
                        .blue_noise_vec2 = record_ctx.task_blue_noise_vec2_image,

                        .radiance_tex = radiance_tex,
                        .reservoir_input_tex = reservoir_input_tex,
                        .gbuffer_tex = gbuffer_depth.gbuffer,
                        .depth_tex = gbuffer_depth.depth.task_resources.output_image,
                        .half_view_normal_tex = half_view_normal_tex,
                        .half_depth_tex = half_depth_tex,
                        .ssao_tex = ssao_tex,
                        .candidate_radiance_tex = candidate_radiance_tex,
                        .candidate_hit_tex = candidate_hit_tex,
                        .temporal_reservoir_packed_tex = temporal_reservoir_packed_tex,
                        .bounced_radiance_input_tex = bounced_radiance_input_tex,

                        .irradiance_output_tex = irradiance_output_tex,
                    },
                },
                &rtdgi_restir_resolve_task_state,
                record_ctx.render_resolution,
                RtdgiRestirResolvePush{
                    extent_inv_extent,
                    extent_inv_extent,
                },
            });

            return irradiance_output_tex;
        }();

        auto filtered_tex = temporal(
            record_ctx,
            irradiance_tex,
            reprojection_map,
            reprojected_rtdgi.history_tex,
            invalidity_output_tex,
            reprojected_rtdgi.temporal_output_tex);

        filtered_tex = spatial(
            record_ctx,
            filtered_tex,
            gbuffer_depth,
            ssao_tex);

        return filtered_tex;
#else
        return candidate_radiance_tex;
#endif
    }
};

#endif
