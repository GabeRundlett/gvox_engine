#pragma once

#include <shared/core.inl>

#if TRACE_SECONDARY_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(TraceSecondaryCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), shadow_image_buffer)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_3D, blue_noise_vec2)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct TraceSecondaryComputePush {
    TraceSecondaryCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(TraceSecondaryComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_RWBufferPtr(daxa_u32) shadow_image_buffer = push.uses.shadow_image_buffer;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_BufferPtr)
daxa_ImageViewId blue_noise_vec2 = push.uses.blue_noise_vec2;
daxa_ImageViewId g_buffer_image_id = push.uses.g_buffer_image_id;
daxa_ImageViewId depth_image_id = push.uses.depth_image_id;
#endif
#endif

#if UPSCALE_RECONSTRUCT_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(UpscaleReconstructCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_image_id)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), scaled_shading_image)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), src_image_id)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32), dst_image_id)
DAXA_DECL_TASK_HEAD_END
struct UpscaleReconstructComputePush {
    UpscaleReconstructCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(UpscaleReconstructComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewId depth_image_id = push.uses.depth_image_id;
daxa_ImageViewId reprojection_image_id = push.uses.reprojection_image_id;
daxa_BufferPtr(daxa_u32) scaled_shading_image = push.uses.scaled_shading_image;
daxa_BufferPtr(daxa_u32) src_image_id = push.uses.src_image_id;
daxa_RWBufferPtr(daxa_u32) dst_image_id = push.uses.dst_image_id;
#endif
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
            .push_constant_size = sizeof(TraceSecondaryComputePush),
            .name = "trace_secondary",
        });
    }

    void record_commands(TraceSecondaryComputePush const &push, daxa::CommandRecorder &recorder, daxa_u32vec2 render_size) {
        if (!pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(pipeline.get());
        recorder.push_constant(push);
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
            .push_constant_size = sizeof(UpscaleReconstructComputePush),
            .name = "upscale_reconstruct",
        });
    }

    void record_commands(UpscaleReconstructComputePush const &push, daxa::CommandRecorder &recorder, daxa_u32vec2 render_size) {
        if (!pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(pipeline.get());
        recorder.push_constant(push);
        // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
        recorder.dispatch({(render_size.x + 7) / 8, (render_size.y + 7) / 8});
    }
};

struct TraceSecondaryComputeTask {
    TraceSecondaryCompute::Uses uses;
    TraceSecondaryComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto const &image_info = ti.get_device().info_image(uses.g_buffer_image_id.image()).value();
        auto push = TraceSecondaryComputePush{};
        ti.copy_task_head_to(&push.uses);
        state->record_commands(push, recorder, {(image_info.size.x + (SHADING_SCL - 1)) / SHADING_SCL, (image_info.size.y + (SHADING_SCL - 1)) / SHADING_SCL});
    }
};

struct UpscaleReconstructComputeTask {
    UpscaleReconstructCompute::Uses uses;
    UpscaleReconstructComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto const &image_info = ti.get_device().info_image(uses.reprojection_image_id.image()).value();
        auto push = UpscaleReconstructComputePush{};
        ti.copy_task_head_to(&push.uses);
        state->record_commands(push, recorder, {image_info.size.x, image_info.size.y});
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
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                .shadow_image_buffer = scaled_shading_image,
                VOXELS_BUFFER_USES_ASSIGN(voxel_buffers),
                .blue_noise_vec2 = record_ctx.task_blue_noise_vec2_image,
                .g_buffer_image_id = gbuffer_depth.gbuffer,
                .depth_image_id = scaled_depth_image,
            },
            .state = &trace_secondary_task_state,
        });

        record_ctx.task_graph.add_task(UpscaleReconstructComputeTask{
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                .depth_image_id = gbuffer_depth.depth.task_resources.output_resource,
                .reprojection_image_id = reprojection_map,
                .scaled_shading_image = scaled_shading_image,
                .src_image_id = prev_shadow_bitmap,
                .dst_image_id = shadow_bitmap,
            },
            .state = &upscale_reconstruct_task_state,
        });

        return daxa::TaskBufferView{shadow_bitmap};
    }
};

#endif
