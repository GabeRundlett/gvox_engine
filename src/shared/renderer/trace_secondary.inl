#pragma once

#include <shared/core.inl>

#if TRACE_SECONDARY_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(TraceSecondaryCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, shadow_bitmap)
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
daxa_ImageViewId shadow_bitmap = push.uses.shadow_bitmap;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_BufferPtr)
daxa_ImageViewId blue_noise_vec2 = push.uses.blue_noise_vec2;
daxa_ImageViewId g_buffer_image_id = push.uses.g_buffer_image_id;
daxa_ImageViewId depth_image_id = push.uses.depth_image_id;
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

struct TraceSecondaryComputeTask {
    TraceSecondaryCompute::Uses uses;
    std::string name = "TraceSecondaryCompute";
    TraceSecondaryComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto const &image_info = ti.get_device().info_image(uses.g_buffer_image_id.image()).value();
        auto push = TraceSecondaryComputePush{};
        ti.copy_task_head_to(&push.uses);
        state->record_commands(push, recorder, {image_info.size.x, image_info.size.y});
    }
};

struct ShadowRenderer {
    TraceSecondaryComputeTaskState trace_secondary_task_state;

    ShadowRenderer(AsyncPipelineManager &pipeline_manager)
        : trace_secondary_task_state{pipeline_manager} {
    }

    auto render(RecordContext &record_ctx, GbufferDepth &gbuffer_depth, VoxelWorld::Buffers &voxel_buffers)
        -> daxa::TaskImageView {

        auto shadow_bitmap = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R8_UNORM,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "shadow_bitmap",
        });

        record_ctx.task_graph.add_task(TraceSecondaryComputeTask{
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                .shadow_bitmap = shadow_bitmap,
                VOXELS_BUFFER_USES_ASSIGN(voxel_buffers),
                .blue_noise_vec2 = record_ctx.task_blue_noise_vec2_image,
                .g_buffer_image_id = gbuffer_depth.gbuffer,
                .depth_image_id = gbuffer_depth.depth.task_resources.output_resource,
            },
            .state = &trace_secondary_task_state,
        });
        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "trace shadow bitmap", .task_image_id = shadow_bitmap, .type = DEBUG_IMAGE_TYPE_DEFAULT_UINT});

        return daxa::TaskImageView{shadow_bitmap};
    }
};

#endif
