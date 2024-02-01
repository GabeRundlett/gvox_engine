#pragma once

#include <shared/core.inl>
#include <shared/renderer/ircache.inl>

#if RtdgiFullresReprojectComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(RtdgiFullresReprojectCompute, 4)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, output_tex)
DAXA_DECL_TASK_HEAD_END
struct RtdgiFullresReprojectComputePush {
    daxa_f32vec4 output_tex_size;
    DAXA_TH_BLOB(RtdgiFullresReprojectCompute, uses)
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(RtdgiFullresReprojectComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewIndex input_tex = push.uses.input_tex;
daxa_ImageViewIndex reprojection_tex = push.uses.reprojection_tex;
daxa_ImageViewIndex output_tex = push.uses.output_tex;
#endif
#endif

#if RtdgiValidateComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(RtdgiValidateCompute, 12 + VOXEL_BUFFER_USE_N + IRCACHE_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, half_view_normal_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojected_gi_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, reservoir_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reservoir_ray_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_3D, blue_noise_vec2)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
IRCACHE_USE_BUFFERS()
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, CUBE, sky_cube_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, irradiance_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, ray_orig_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, rt_history_invalidity_out_tex)
DAXA_DECL_TASK_HEAD_END
struct RtdgiValidateComputePush {
    daxa_f32vec4 gbuffer_tex_size;
    DAXA_TH_BLOB(RtdgiValidateCompute, uses)
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(RtdgiValidateComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewIndex half_view_normal_tex = push.uses.half_view_normal_tex;
daxa_ImageViewIndex depth_tex = push.uses.depth_tex;
daxa_ImageViewIndex reprojected_gi_tex = push.uses.reprojected_gi_tex;
daxa_ImageViewIndex reservoir_tex = push.uses.reservoir_tex;
daxa_ImageViewIndex reservoir_ray_history_tex = push.uses.reservoir_ray_history_tex;
daxa_ImageViewIndex blue_noise_vec2 = push.uses.blue_noise_vec2;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_BufferPtr)
IRCACHE_USE_BUFFERS_PUSH_USES()
daxa_ImageViewIndex sky_cube_tex = push.uses.sky_cube_tex;
daxa_ImageViewIndex irradiance_history_tex = push.uses.irradiance_history_tex;
daxa_ImageViewIndex ray_orig_history_tex = push.uses.ray_orig_history_tex;
daxa_ImageViewIndex rt_history_invalidity_out_tex = push.uses.rt_history_invalidity_out_tex;
#endif
#endif

#if RtdgiTraceComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(RtdgiTraceCompute, 14 + VOXEL_BUFFER_USE_N + IRCACHE_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, half_view_normal_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojected_gi_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_3D, blue_noise_vec2)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, CUBE, sky_cube_tex)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
IRCACHE_USE_BUFFERS()
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, ray_orig_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, candidate_irradiance_out_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, candidate_normal_out_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, candidate_hit_out_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, rt_history_invalidity_in_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, rt_history_invalidity_out_tex)
DAXA_DECL_TASK_HEAD_END
struct RtdgiTraceComputePush {
    daxa_f32vec4 gbuffer_tex_size;
    DAXA_TH_BLOB(RtdgiTraceCompute, uses)
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(RtdgiTraceComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewIndex half_view_normal_tex = push.uses.half_view_normal_tex;
daxa_ImageViewIndex depth_tex = push.uses.depth_tex;
daxa_ImageViewIndex reprojected_gi_tex = push.uses.reprojected_gi_tex;
daxa_ImageViewIndex reprojection_tex = push.uses.reprojection_tex;
daxa_ImageViewIndex blue_noise_vec2 = push.uses.blue_noise_vec2;
daxa_ImageViewIndex sky_cube_tex = push.uses.sky_cube_tex;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_BufferPtr)
IRCACHE_USE_BUFFERS_PUSH_USES()
daxa_ImageViewIndex ray_orig_history_tex = push.uses.ray_orig_history_tex;
daxa_ImageViewIndex candidate_irradiance_out_tex = push.uses.candidate_irradiance_out_tex;
daxa_ImageViewIndex candidate_normal_out_tex = push.uses.candidate_normal_out_tex;
daxa_ImageViewIndex candidate_hit_out_tex = push.uses.candidate_hit_out_tex;
daxa_ImageViewIndex rt_history_invalidity_in_tex = push.uses.rt_history_invalidity_in_tex;
daxa_ImageViewIndex rt_history_invalidity_out_tex = push.uses.rt_history_invalidity_out_tex;
#endif
#endif

#if defined(__cplusplus)

struct ReprojectedRtdgi {
    daxa::TaskImageView reprojected_history_tex;
    daxa::TaskImageView temporal_output_tex;
};

struct RtdgiCandidates {
    daxa::TaskImageView candidate_radiance_tex;
    daxa::TaskImageView candidate_normal_tex;
    daxa::TaskImageView candidate_hit_tex;
};
struct RtdgiOutput {
    daxa::TaskImageView screen_irradiance_tex;
    RtdgiCandidates candidates;
};

struct RtdgiRenderer {
    PingPongImage temporal_radiance_tex;
    PingPongImage temporal_ray_orig_tex;
    PingPongImage temporal_ray_tex;
    PingPongImage pp_temporal_reservoir_tex;
    PingPongImage temporal_candidate_tex;

    PingPongImage temporal_invalidity_tex;

    PingPongImage temporal2_tex;
    PingPongImage temporal2_variance_tex;
    PingPongImage temporal_hit_normal_tex;

    daxa_u32 spatial_reuse_pass_count;
    bool use_raytraced_reservoir_visibility;

    void next_frame() {
        temporal2_tex.swap();
        temporal_hit_normal_tex.swap();
        temporal_candidate_tex.swap();
        temporal_invalidity_tex.swap();
        temporal_radiance_tex.swap();
        temporal_ray_orig_tex.swap();
        temporal_ray_tex.swap();
        pp_temporal_reservoir_tex.swap();
    }

    auto reproject(RecordContext &record_ctx, daxa::TaskImageView reprojection_map) -> ReprojectedRtdgi {
        auto [temporal_output_tex, history_tex] = temporal2_tex.get(
            record_ctx.device,
            {
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                .name = "temporal2_tex",
            });
        record_ctx.task_graph.use_persistent_image(temporal_output_tex);
        record_ctx.task_graph.use_persistent_image(history_tex);

        auto reprojected_history_tex = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "reprojected_history_tex",
        });

        record_ctx.add(ComputeTask<RtdgiFullresReprojectCompute, RtdgiFullresReprojectComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"rtdgi/fullres_reproject.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{RtdgiFullresReprojectCompute::gpu_input, record_ctx.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{RtdgiFullresReprojectCompute::input_tex, history_tex}},
                daxa::TaskViewVariant{std::pair{RtdgiFullresReprojectCompute::reprojection_tex, reprojection_map}},
                daxa::TaskViewVariant{std::pair{RtdgiFullresReprojectCompute::output_tex, reprojected_history_tex}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, RtdgiFullresReprojectComputePush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(RtdgiFullresReprojectCompute::output_tex).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                push.output_tex_size = extent_inv_extent_2d(image_info);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });
        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "rtdgi reprojected history", .task_image_id = reprojected_history_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        return ReprojectedRtdgi{
            reprojected_history_tex,
            temporal_output_tex,
        };
    }

    auto render(
        RecordContext &record_ctx,
        ReprojectedRtdgi reprojected_rtdgi,
        GbufferDepth &gbuffer_depth,
        daxa::TaskImageView reprojection_map,
        daxa::TaskImageView sky_cube,
        IrcacheRenderState &ircache,
        VoxelWorld::Buffers &voxel_buffers,
        daxa::TaskImageView ssao_tex) -> RtdgiOutput {
        auto [reprojected_history_tex, temporal_output_tex] = reprojected_rtdgi;
        auto half_ssao_tex = extract_downscaled_ssao(record_ctx, ssao_tex);
        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "rtdgi downscaled ssao", .task_image_id = half_ssao_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        auto gbuffer_half_res = daxa_u32vec2{(record_ctx.render_resolution.x + 1) / 2, (record_ctx.render_resolution.y + 1) / 2};

        auto [hit_normal_output_tex, hit_normal_history_tex] = temporal_hit_normal_tex.get(
            record_ctx.device,
            {
                .format = daxa::Format::R8G8B8A8_UNORM,
                .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                .name = "temporal_hit_normal_tex",
            });
        record_ctx.task_graph.use_persistent_image(hit_normal_output_tex);
        record_ctx.task_graph.use_persistent_image(hit_normal_history_tex);

        auto [candidate_output_tex, candidate_history_tex] = temporal_candidate_tex.get(
            record_ctx.device,
            {
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                .name = "temporal_candidate_tex",
            });
        record_ctx.task_graph.use_persistent_image(candidate_output_tex);
        record_ctx.task_graph.use_persistent_image(candidate_history_tex);

        auto candidate_radiance_tex = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
            .name = "candidate_radiance_tex",
        });
        auto candidate_normal_tex = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R8G8B8A8_SNORM,
            .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
            .name = "candidate_normal_tex",
        });
        auto candidate_hit_tex = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
            .name = "candidate_hit_tex",
        });
        auto temporal_reservoir_packed_tex = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R32G32B32A32_UINT,
            .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
            .name = "temporal_reservoir_packed_tex",
        });
        auto half_depth_tex = gbuffer_depth.get_downscaled_depth(record_ctx);

        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "rtdgi downscaled depth", .task_image_id = half_depth_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        auto [invalidity_output_tex, invalidity_history_tex] = temporal_invalidity_tex.get(
            record_ctx.device,
            {
                .format = daxa::Format::R16G16_SFLOAT,
                .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                .name = "temporal_invalidity_tex",
            });
        record_ctx.task_graph.use_persistent_image(invalidity_output_tex);
        record_ctx.task_graph.use_persistent_image(invalidity_history_tex);

        // auto [radiance_tex, temporal_reservoir_tex] =
        auto radiance_tex = daxa::TaskImageView{};
        auto temporal_reservoir_tex = daxa::TaskImageView{};
        {
            auto [radiance_output_tex, radiance_history_tex] = temporal_radiance_tex.get(
                record_ctx.device,
                {
                    .format = daxa::Format::R32G32B32A32_UINT,
                    .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                    .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                    .name = "temporal_radiance_tex",
                });
            record_ctx.task_graph.use_persistent_image(radiance_output_tex);
            record_ctx.task_graph.use_persistent_image(radiance_history_tex);

            auto [ray_orig_output_tex, ray_orig_history_tex] = temporal_ray_orig_tex.get(
                record_ctx.device,
                {
                    .format = daxa::Format::R32G32B32A32_SFLOAT,
                    .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                    .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                    .name = "temporal_ray_orig_tex",
                });
            record_ctx.task_graph.use_persistent_image(ray_orig_output_tex);
            record_ctx.task_graph.use_persistent_image(ray_orig_history_tex);

            auto [ray_output_tex, ray_history_tex] = temporal_ray_tex.get(
                record_ctx.device,
                {
                    .format = daxa::Format::R16G16B16A16_SFLOAT,
                    .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                    .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                    .name = "temporal_ray_tex",
                });
            record_ctx.task_graph.use_persistent_image(ray_output_tex);
            record_ctx.task_graph.use_persistent_image(ray_history_tex);

            auto half_view_normal_tex = gbuffer_depth.get_downscaled_view_normal(record_ctx);

            auto rt_history_validity_pre_input_tex = record_ctx.task_graph.create_transient_image({
                .format = daxa::Format::R8_UNORM,
                .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                .name = "rt_history_validity_pre_input_tex",
            });

            auto [reservoir_output_tex, reservoir_history_tex] = pp_temporal_reservoir_tex.get(
                record_ctx.device,
                {
                    .format = daxa::Format::R32G32_UINT,
                    .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                    .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                    .name = "temporal_reservoir_tex",
                });
            record_ctx.task_graph.use_persistent_image(reservoir_output_tex);
            record_ctx.task_graph.use_persistent_image(reservoir_history_tex);

            record_ctx.add(ComputeTask<RtdgiValidateCompute, RtdgiValidateComputePush, NoTaskInfo>{
                .source = daxa::ShaderFile{"rtdgi/diffuse_validate.comp.glsl"},
                .views = std::array{
                    daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::gpu_input, record_ctx.task_input_buffer}},
                    daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::globals, record_ctx.task_globals_buffer}},
                    daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::half_view_normal_tex, half_view_normal_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::depth_tex, gbuffer_depth.depth.current()}},
                    daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::reprojected_gi_tex, reprojected_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::reservoir_tex, reservoir_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::reservoir_ray_history_tex, ray_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::blue_noise_vec2, record_ctx.task_blue_noise_vec2_image}},
                    // daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::reprojection_tex, reprojection_map}},
                    VOXELS_BUFFER_USES_ASSIGN(RtdgiValidateCompute, voxel_buffers),
                    IRCACHE_BUFFER_USES_ASSIGN(RtdgiValidateCompute, ircache),
                    daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::sky_cube_tex, sky_cube}},
                    daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::irradiance_history_tex, radiance_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::ray_orig_history_tex, ray_orig_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::rt_history_invalidity_out_tex, rt_history_validity_pre_input_tex}},
                },
                .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, RtdgiValidateComputePush &push, NoTaskInfo const &) {
                    auto const image_info = ti.device.info_image(ti.get(RtdgiValidateCompute::depth_tex).ids[0]).value();
                    auto const candidate_image_info = ti.device.info_image(ti.get(RtdgiValidateCompute::reservoir_tex).ids[0]).value();
                    ti.recorder.set_pipeline(pipeline);
                    push.gbuffer_tex_size = extent_inv_extent_2d(image_info);
                    set_push_constant(ti, push);
                    ti.recorder.dispatch({(candidate_image_info.size.x + 7) / 8, (candidate_image_info.size.y + 7) / 8});
                },
            });

            AppUi::DebugDisplay::s_instance->passes.push_back({.name = "rtdgi validate", .task_image_id = half_depth_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            auto rt_history_validity_input_tex = record_ctx.task_graph.create_transient_image({
                .format = daxa::Format::R8_UNORM,
                .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                .name = "rt_history_validity_input_tex",
            });

            record_ctx.add(ComputeTask<RtdgiTraceCompute, RtdgiTraceComputePush, NoTaskInfo>{
                .source = daxa::ShaderFile{"rtdgi/trace_diffuse.comp.glsl"},
                .views = std::array{
                    daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::gpu_input, record_ctx.task_input_buffer}},
                    daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::globals, record_ctx.task_globals_buffer}},
                    daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::half_view_normal_tex, half_view_normal_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::depth_tex, gbuffer_depth.depth.current()}},
                    daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::reprojected_gi_tex, reprojected_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::reprojection_tex, reprojection_map}},
                    daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::blue_noise_vec2, record_ctx.task_blue_noise_vec2_image}},
                    daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::sky_cube_tex, sky_cube}},
                    VOXELS_BUFFER_USES_ASSIGN(RtdgiTraceCompute, voxel_buffers),
                    IRCACHE_BUFFER_USES_ASSIGN(RtdgiTraceCompute, ircache),
                    daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::ray_orig_history_tex, ray_orig_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::candidate_irradiance_out_tex, candidate_radiance_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::candidate_normal_out_tex, candidate_normal_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::candidate_hit_out_tex, candidate_hit_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::rt_history_invalidity_in_tex, rt_history_validity_pre_input_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::rt_history_invalidity_out_tex, rt_history_validity_input_tex}},
                },
                .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, RtdgiTraceComputePush &push, NoTaskInfo const &) {
                    auto const image_info = ti.device.info_image(ti.get(RtdgiTraceCompute::depth_tex).ids[0]).value();
                    auto const candidate_image_info = ti.device.info_image(ti.get(RtdgiTraceCompute::candidate_hit_out_tex).ids[0]).value();
                    ti.recorder.set_pipeline(pipeline);
                    push.gbuffer_tex_size = extent_inv_extent_2d(image_info);
                    set_push_constant(ti, push);
                    ti.recorder.dispatch({(candidate_image_info.size.x + 7) / 8, (candidate_image_info.size.y + 7) / 8});
                },
            });

            AppUi::DebugDisplay::s_instance->passes.push_back({.name = "rtdgi trace a", .task_image_id = candidate_radiance_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});
            AppUi::DebugDisplay::s_instance->passes.push_back({.name = "rtdgi trace b", .task_image_id = candidate_normal_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});
            AppUi::DebugDisplay::s_instance->passes.push_back({.name = "rtdgi trace c", .task_image_id = candidate_hit_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});
            AppUi::DebugDisplay::s_instance->passes.push_back({.name = "rtdgi trace d", .task_image_id = rt_history_validity_input_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            // return std::pair{radiance_output_tex, reservoir_output_tex};
            radiance_tex = radiance_output_tex;
            temporal_reservoir_tex = reservoir_output_tex;
        }

        return RtdgiOutput{};
    }
};

#endif
