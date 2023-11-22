#pragma once

#include <shared/core.inl>

#if TRACE_SECONDARY_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(TraceSecondaryComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(shadow_image_buffer, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_READ_WRITE)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(blue_noise_vec2, REGULAR_3D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(g_buffer_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(depth_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_DECL_TASK_USES_END()
#endif

#if UPSCALE_RECONSTRUCT_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(UpscaleReconstructComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(depth_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reprojection_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_BUFFER(scaled_shading_image, daxa_BufferPtr(daxa_u32), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(src_image_id, daxa_BufferPtr(daxa_u32), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(dst_image_id, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_WRITE)
DAXA_DECL_TASK_USES_END()
#endif

#if defined(__cplusplus)

struct TraceSecondaryComputeTaskState {
    AsyncManagedComputePipeline pipeline;

    TraceSecondaryComputeTaskState(AsyncPipelineManager &pipeline_manager) {
        pipeline = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"trace_secondary.comp.glsl"},
                .compile_options = {.defines = {{"TRACE_SECONDARY_COMPUTE", "1"}}},
            },
            .name = "trace_secondary",
        });
    }

    void record_commands(daxa::CommandRecorder &recorder, daxa_u32vec2 render_size) {
        if (!pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(pipeline.get());
        // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
        recorder.dispatch({(render_size.x + 7) / 8, (render_size.y + 7) / 8});
    }
};

struct UpscaleReconstructComputeTaskState {
    AsyncManagedComputePipeline pipeline;

    UpscaleReconstructComputeTaskState(AsyncPipelineManager &pipeline_manager) {
        pipeline = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"trace_secondary.comp.glsl"},
                .compile_options = {.defines = {{"UPSCALE_RECONSTRUCT_COMPUTE", "1"}}},
            },
            .name = "upscale_reconstruct",
        });
    }

    void record_commands(daxa::CommandRecorder &recorder, daxa_u32vec2 render_size) {
        if (!pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(pipeline.get());
        // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
        recorder.dispatch({(render_size.x + 7) / 8, (render_size.y + 7) / 8});
    }
};

struct TraceSecondaryComputeTask : TraceSecondaryComputeUses {
    TraceSecondaryComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        recorder.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        auto const &image_info = ti.get_device().info_image(uses.g_buffer_image_id.image()).value();
        state->record_commands(recorder, {(image_info.size.x + (SHADING_SCL - 1)) / SHADING_SCL, (image_info.size.y + (SHADING_SCL - 1)) / SHADING_SCL});
    }
};

struct UpscaleReconstructComputeTask : UpscaleReconstructComputeUses {
    UpscaleReconstructComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        recorder.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        auto const &image_info = ti.get_device().info_image(uses.reprojection_image_id.image()).value();
        state->record_commands(recorder, {image_info.size.x, image_info.size.y});
    }
};

struct ShadowRenderer {
    PingPongBuffer ping_pong_shading_image;
    TraceSecondaryComputeTaskState trace_secondary_task_state;
    UpscaleReconstructComputeTaskState upscale_reconstruct_task_state;

    ShadowRenderer(AsyncPipelineManager &pipeline_manager)
        : trace_secondary_task_state{pipeline_manager},
          upscale_reconstruct_task_state{pipeline_manager} {
    }

    void next_frame() {
        ping_pong_shading_image.task_resources.output_resource.swap_buffers(ping_pong_shading_image.task_resources.history_resource);
    }

    auto render(RecordContext &record_ctx, GbufferDepth &gbuffer_depth, daxa::TaskImageView reprojection_map, VoxelWorld::Buffers &voxel_buffers)
        -> daxa::TaskBufferView {
        auto scaled_depth_image = gbuffer_depth.get_downscaled_depth(record_ctx);
        ping_pong_shading_image = PingPongBuffer{};
        auto base_size = (record_ctx.render_resolution.x + 31) / 32 * record_ctx.render_resolution.y * static_cast<uint32_t>(sizeof(uint32_t));
        auto [shadow_bitmap, prev_shadow_bitmap] = ping_pong_shading_image.get(
            record_ctx.device,
            {
                .size = base_size,
                .name = "shading_image",
            });
        auto scaled_size = ((record_ctx.render_resolution.x / SHADING_SCL + 31) / 32) * record_ctx.render_resolution.y / SHADING_SCL * static_cast<uint32_t>(sizeof(uint32_t));
        auto scaled_shading_image = record_ctx.task_graph.create_transient_buffer({
            .size = scaled_size,
            .name = "scaled_shading_image",
        });

        record_ctx.task_graph.use_persistent_buffer(shadow_bitmap);
        record_ctx.task_graph.use_persistent_buffer(prev_shadow_bitmap);
        record_ctx.task_graph.add_task({
            .uses = {
                daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{scaled_shading_image},
                daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{shadow_bitmap},
            },
            .task = [scaled_shading_image, scaled_size, shadow_bitmap, base_size](daxa::TaskInterface ti) {
                auto &recorder = ti.get_recorder();
                recorder.clear_buffer({
                    .buffer = ti.uses[scaled_shading_image].buffer(),
                    .offset = 0,
                    .size = scaled_size,
                    .clear_value = 0,
                });
                recorder.clear_buffer({
                    .buffer = ti.uses[shadow_bitmap].buffer(),
                    .offset = 0,
                    .size = base_size,
                    .clear_value = 0,
                });
            },
            .name = "Clear Shadow image",
        });

        record_ctx.task_graph.add_task(TraceSecondaryComputeTask{
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .globals = record_ctx.task_globals_buffer,
                    .shadow_image_buffer = scaled_shading_image,
                    VOXELS_BUFFER_USES_ASSIGN(voxel_buffers),
                    .blue_noise_vec2 = record_ctx.task_blue_noise_vec2_image,
                    .g_buffer_image_id = gbuffer_depth.gbuffer,
                    .depth_image_id = scaled_depth_image,
                },
            },
            &trace_secondary_task_state,
        });

        record_ctx.task_graph.add_task(UpscaleReconstructComputeTask{
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .globals = record_ctx.task_globals_buffer,
                    .depth_image_id = gbuffer_depth.depth.task_resources.output_resource,
                    .reprojection_image_id = reprojection_map,
                    .scaled_shading_image = scaled_shading_image,
                    .src_image_id = prev_shadow_bitmap,
                    .dst_image_id = shadow_bitmap,
                },
            },
            &upscale_reconstruct_task_state,
        });

        return daxa::TaskBufferView{shadow_bitmap};
    }
};

#endif
