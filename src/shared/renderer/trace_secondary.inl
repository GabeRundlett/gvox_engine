#pragma once

#include <shared/core.inl>

#if TRACE_SECONDARY_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(TraceSecondaryComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(voxel_malloc_page_allocator, daxa_RWBufferPtr(VoxelMallocPageAllocator), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(voxel_chunks, daxa_BufferPtr(VoxelLeafChunk), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(blue_noise_vec2, REGULAR_3D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(g_buffer_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(depth_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(indirect_diffuse_image_id, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif

#if UPSCALE_RECONSTRUCT_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(UpscaleReconstructComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(depth_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(reprojection_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(scaled_shading_image, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(src_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(dst_image_id, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif

#if defined(__cplusplus)

struct TraceSecondaryComputeTaskState {
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    TraceSecondaryComputeTaskState(daxa::PipelineManager &pipeline_manager) {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"trace_secondary.comp.glsl"},
                .compile_options = {.defines = {{"TRACE_SECONDARY_COMPUTE", "1"}}},
            },
            .name = "trace_secondary",
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
    auto pipeline_is_valid() -> bool { return pipeline && pipeline->is_valid(); }

    void record_commands(daxa::CommandList &cmd_list, u32vec2 render_size) {
        if (!pipeline_is_valid()) {
            return;
        }
        cmd_list.set_pipeline(*pipeline);
        // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
        cmd_list.dispatch((render_size.x + 7) / 8, (render_size.y + 7) / 8);
    }
};

struct UpscaleReconstructComputeTaskState {
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    UpscaleReconstructComputeTaskState(daxa::PipelineManager &pipeline_manager) {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"trace_secondary.comp.glsl"},
                .compile_options = {.defines = {{"UPSCALE_RECONSTRUCT_COMPUTE", "1"}}},
            },
            .name = "upscale_reconstruct",
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
    auto pipeline_is_valid() -> bool { return pipeline && pipeline->is_valid(); }

    void record_commands(daxa::CommandList &cmd_list, u32vec2 render_size) {
        if (!pipeline_is_valid()) {
            return;
        }
        cmd_list.set_pipeline(*pipeline);
        // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
        cmd_list.dispatch((render_size.x + 7) / 8, (render_size.y + 7) / 8);
    }
};

struct TraceSecondaryComputeTask : TraceSecondaryComputeUses {
    TraceSecondaryComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        auto const &image_info = ti.get_device().info_image(uses.indirect_diffuse_image_id.image());
        state->record_commands(cmd_list, {image_info.size.x, image_info.size.y});
    }
};

struct UpscaleReconstructComputeTask : UpscaleReconstructComputeUses {
    UpscaleReconstructComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        auto const &image_info = ti.get_device().info_image(uses.dst_image_id.image());
        state->record_commands(cmd_list, {image_info.size.x, image_info.size.y});
    }
};

struct ShadowRenderer {
    PingPongImage ping_pong_shading_image;
    TraceSecondaryComputeTaskState trace_secondary_task_state;
    UpscaleReconstructComputeTaskState upscale_reconstruct_task_state;

    ShadowRenderer(daxa::PipelineManager &pipeline_manager)
        : trace_secondary_task_state{pipeline_manager},
          upscale_reconstruct_task_state{pipeline_manager} {
    }

    void next_frame() {
        ping_pong_shading_image.task_resources.output_image.swap_images(ping_pong_shading_image.task_resources.history_image);
    }

    auto render(RecordContext &record_ctx, GbufferDepth &gbuffer_depth, daxa::TaskImageView reprojection_map, daxa::TaskBufferView voxel_malloc_task_allocator_buffer, daxa::TaskBufferView task_voxel_chunks_buffer)
        -> daxa::TaskImageView {
        auto scaled_depth_image = gbuffer_depth.get_downscaled_depth(record_ctx);
        ping_pong_shading_image = PingPongImage{};
        auto [shading_image, prev_shading_image] = ping_pong_shading_image.get(
            record_ctx.device,
            {
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                .name = "shading_image",
            });
        auto scaled_shading_image = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {record_ctx.render_resolution.x / SHADING_SCL, record_ctx.render_resolution.y / SHADING_SCL, 1},
            .name = "scaled_shading_image",
        });
        record_ctx.task_graph.use_persistent_image(shading_image);
        record_ctx.task_graph.use_persistent_image(prev_shading_image);
        record_ctx.task_graph.add_task(TraceSecondaryComputeTask{
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .globals = record_ctx.task_globals_buffer,
                    .voxel_malloc_page_allocator = voxel_malloc_task_allocator_buffer,
                    .voxel_chunks = task_voxel_chunks_buffer,
                    .blue_noise_vec2 = record_ctx.task_blue_noise_vec2_image,
                    .g_buffer_image_id = gbuffer_depth.gbuffer,
                    .depth_image_id = scaled_depth_image,
                    .indirect_diffuse_image_id = scaled_shading_image,
                },
            },
            &trace_secondary_task_state,
        });

        record_ctx.task_graph.add_task(UpscaleReconstructComputeTask{
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .globals = record_ctx.task_globals_buffer,
                    .depth_image_id = gbuffer_depth.depth.task_resources.output_image,
                    .reprojection_image_id = reprojection_map,
                    .scaled_shading_image = scaled_shading_image,
                    .src_image_id = prev_shading_image,
                    .dst_image_id = shading_image,
                },
            },
            &upscale_reconstruct_task_state,
        });

        return daxa::TaskImageView{shading_image};
    }
};

#endif
