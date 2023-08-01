#pragma once

#include <shared/core.inl>

#if TRACE_DEPTH_PREPASS_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(TraceDepthPrepassComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(voxel_malloc_page_allocator, daxa_RWBufferPtr(VoxelMallocPageAllocator), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(voxel_chunks, daxa_BufferPtr(VoxelLeafChunk), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(render_depth_prepass_image, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif

#if TRACE_PRIMARY_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(TracePrimaryComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(voxel_malloc_page_allocator, daxa_RWBufferPtr(VoxelMallocPageAllocator), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(voxel_chunks, daxa_BufferPtr(VoxelLeafChunk), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(blue_noise_vec2, REGULAR_3D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(render_depth_prepass_image, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(g_buffer_image_id, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(vs_normal_image_id, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(velocity_image_id, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_TASK_USE_IMAGE(depth_image_id, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif

#if defined(__cplusplus)

struct TraceDepthPrepassComputeTaskState {
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    TraceDepthPrepassComputeTaskState(daxa::PipelineManager &pipeline_manager) {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"trace_depth_prepass.comp.glsl"},
                .compile_options = {.defines = {{"TRACE_DEPTH_PREPASS_COMPUTE", "1"}}},
            },
            .name = "trace_depth_prepass",
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

struct TracePrimaryComputeTaskState {
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    TracePrimaryComputeTaskState(daxa::PipelineManager &pipeline_manager) {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"trace_primary.comp.glsl"},
                .compile_options = {.defines = {{"TRACE_PRIMARY_COMPUTE", "1"}}},
            },
            .name = "trace_primary",
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

struct TraceDepthPrepassComputeTask : TraceDepthPrepassComputeUses {
    TraceDepthPrepassComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        auto const &image_info = ti.get_device().info_image(uses.render_depth_prepass_image.image());
        state->record_commands(cmd_list, {image_info.size.x, image_info.size.y});
    }
};

struct TracePrimaryComputeTask : TracePrimaryComputeUses {
    TracePrimaryComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        auto const &image_info = ti.get_device().info_image(uses.g_buffer_image_id.image());
        state->record_commands(cmd_list, {image_info.size.x, image_info.size.y});
    }
};

struct GbufferRenderer {
    GbufferDepth gbuffer_depth;
    TraceDepthPrepassComputeTaskState trace_depth_prepass_task_state;
    TracePrimaryComputeTaskState trace_primary_task_state;

    GbufferRenderer(daxa::PipelineManager &pipeline_manager)
        : gbuffer_depth{pipeline_manager},
          trace_depth_prepass_task_state{pipeline_manager},
          trace_primary_task_state{pipeline_manager} {
    }

    void next_frame() {
        gbuffer_depth.next_frame();
    }

    auto render(RecordContext &record_ctx, daxa::TaskBufferView voxel_malloc_task_allocator_buffer, daxa::TaskBufferView task_voxel_chunks_buffer)
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

        record_ctx.task_graph.add_task(TraceDepthPrepassComputeTask{
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .globals = record_ctx.task_globals_buffer,
                    .voxel_malloc_page_allocator = voxel_malloc_task_allocator_buffer,
                    .voxel_chunks = task_voxel_chunks_buffer,
                    .render_depth_prepass_image = depth_prepass_image,
                },
            },
            &trace_depth_prepass_task_state,
        });

        record_ctx.task_graph.add_task(TracePrimaryComputeTask{
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .globals = record_ctx.task_globals_buffer,
                    .voxel_malloc_page_allocator = voxel_malloc_task_allocator_buffer,
                    .voxel_chunks = task_voxel_chunks_buffer,
                    .blue_noise_vec2 = record_ctx.task_blue_noise_vec2_image,
                    .render_depth_prepass_image = depth_prepass_image,
                    .g_buffer_image_id = gbuffer_depth.gbuffer,
                    .vs_normal_image_id = gbuffer_depth.geometric_normal,
                    .velocity_image_id = velocity_image,
                    .depth_image_id = depth_image,
                },
            },
            &trace_primary_task_state,
        });

        return {gbuffer_depth, velocity_image};
    }
};

#endif
