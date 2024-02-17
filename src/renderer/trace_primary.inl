#pragma once

#include <core.inl>
#include <renderer/core.inl>

DAXA_DECL_TASK_HEAD_BEGIN(TraceDepthPrepassCompute, 3 + VOXEL_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, render_depth_prepass_image)
DAXA_DECL_TASK_HEAD_END
struct TraceDepthPrepassComputePush {
    DAXA_TH_BLOB(TraceDepthPrepassCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(TracePrimaryCompute, 7 + VOXEL_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_3D, blue_noise_vec2)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, debug_texture)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, render_depth_prepass_image)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, velocity_image_id)
DAXA_DECL_TASK_HEAD_END
struct TracePrimaryComputePush {
    DAXA_TH_BLOB(TracePrimaryCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(CompositeParticlesCompute, 9)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, depth_image_id)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_BufferPtr(SimulatedVoxelParticle), simulated_voxel_particles)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, particles_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, particles_depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct CompositeParticlesComputePush {
    DAXA_TH_BLOB(CompositeParticlesCompute, uses)
};

#if defined(__cplusplus)

struct GbufferRenderer {
    GbufferDepth gbuffer_depth;

    void next_frame() {
        gbuffer_depth.next_frame();
    }

    auto render(RecordContext &record_ctx, VoxelWorldBuffers &voxel_buffers, daxa::TaskBufferView simulated_voxel_particles_buffer, daxa::TaskImageView particles_image, daxa::TaskImageView particles_depth_image)
        -> std::pair<GbufferDepth &, daxa::TaskImageView> {
        gbuffer_depth.gbuffer = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R32G32B32A32_UINT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "gbuffer",
        });
        gbuffer_depth.geometric_normal = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::A2B10G10R10_UNORM_PACK32,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "normal",
        });

        gbuffer_depth.downscaled_view_normal = std::nullopt;
        gbuffer_depth.downscaled_depth = std::nullopt;

        gbuffer_depth.depth = PingPongImage{};
        auto [depth_image, prev_depth_image] = gbuffer_depth.depth.get(
            record_ctx.device,
            {
                .format = daxa::Format::R32_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_SRC,
                .name = "depth_image",
            });

        record_ctx.task_graph.use_persistent_image(depth_image);
        record_ctx.task_graph.use_persistent_image(prev_depth_image);

        auto velocity_image = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "velocity_image",
        });

        auto depth_prepass_image = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R32_SFLOAT,
            .size = {record_ctx.render_resolution.x / PREPASS_SCL, record_ctx.render_resolution.y / PREPASS_SCL, 1},
            .name = "depth_prepass_image",
        });

        record_ctx.add(ComputeTask<TraceDepthPrepassCompute, TraceDepthPrepassComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"trace_primary.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TraceDepthPrepassCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
                daxa::TaskViewVariant{std::pair{TraceDepthPrepassCompute::globals, record_ctx.gpu_context->task_globals_buffer}},
                VOXELS_BUFFER_USES_ASSIGN(TraceDepthPrepassCompute, voxel_buffers),
                daxa::TaskViewVariant{std::pair{TraceDepthPrepassCompute::render_depth_prepass_image, depth_prepass_image}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TraceDepthPrepassComputePush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(TraceDepthPrepassCompute::render_depth_prepass_image).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
                ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });

        record_ctx.add(ComputeTask<TracePrimaryCompute, TracePrimaryComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"trace_primary.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TracePrimaryCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
                daxa::TaskViewVariant{std::pair{TracePrimaryCompute::globals, record_ctx.gpu_context->task_globals_buffer}},
                VOXELS_BUFFER_USES_ASSIGN(TracePrimaryCompute, voxel_buffers),
                daxa::TaskViewVariant{std::pair{TracePrimaryCompute::blue_noise_vec2, record_ctx.gpu_context->task_blue_noise_vec2_image}},
                daxa::TaskViewVariant{std::pair{TracePrimaryCompute::debug_texture, record_ctx.gpu_context->task_debug_texture}},
                daxa::TaskViewVariant{std::pair{TracePrimaryCompute::render_depth_prepass_image, depth_prepass_image}},
                daxa::TaskViewVariant{std::pair{TracePrimaryCompute::g_buffer_image_id, gbuffer_depth.gbuffer}},
                daxa::TaskViewVariant{std::pair{TracePrimaryCompute::velocity_image_id, velocity_image}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TracePrimaryComputePush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(TracePrimaryCompute::g_buffer_image_id).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                // debug_utils::Console::add_log(fmt::format("0 {}, {}", image_info.size.x, image_info.size.y));
                set_push_constant(ti, push);
                // debug_utils::Console::add_log(fmt::format("1 {}, {}", image_info.size.x, image_info.size.y));
                // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
                ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });

        record_ctx.add(ComputeTask<CompositeParticlesCompute, CompositeParticlesComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"trace_primary.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{CompositeParticlesCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
                daxa::TaskViewVariant{std::pair{CompositeParticlesCompute::globals, record_ctx.gpu_context->task_globals_buffer}},

                daxa::TaskViewVariant{std::pair{CompositeParticlesCompute::g_buffer_image_id, gbuffer_depth.gbuffer}},
                daxa::TaskViewVariant{std::pair{CompositeParticlesCompute::velocity_image_id, velocity_image}},
                daxa::TaskViewVariant{std::pair{CompositeParticlesCompute::vs_normal_image_id, gbuffer_depth.geometric_normal}},
                daxa::TaskViewVariant{std::pair{CompositeParticlesCompute::depth_image_id, depth_image}},

                daxa::TaskViewVariant{std::pair{CompositeParticlesCompute::simulated_voxel_particles, simulated_voxel_particles_buffer}},
                daxa::TaskViewVariant{std::pair{CompositeParticlesCompute::particles_image_id, particles_image}},
                daxa::TaskViewVariant{std::pair{CompositeParticlesCompute::particles_depth_image_id, particles_depth_image}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, CompositeParticlesComputePush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(CompositeParticlesCompute::g_buffer_image_id).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
                ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "depth_prepass", .task_image_id = depth_prepass_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "gbuffer", .task_image_id = gbuffer_depth.gbuffer, .type = DEBUG_IMAGE_TYPE_GBUFFER});
        debug_utils::DebugDisplay::add_pass({.name = "geometric_normal", .task_image_id = gbuffer_depth.geometric_normal, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "velocity", .task_image_id = velocity_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        return {gbuffer_depth, velocity_image};
    }
};

#endif
