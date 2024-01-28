#pragma once

#include <shared/core.inl>

#if TraceSecondaryComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(TraceSecondaryCompute, 6 + VOXEL_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, shadow_mask)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_3D, blue_noise_vec2)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct TraceSecondaryComputePush {
    DAXA_TH_BLOB(TraceSecondaryCompute, uses)
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(TraceSecondaryComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewId shadow_mask = push.uses.shadow_mask;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_BufferPtr)
daxa_ImageViewId blue_noise_vec2 = push.uses.blue_noise_vec2;
daxa_ImageViewId g_buffer_image_id = push.uses.g_buffer_image_id;
daxa_ImageViewId depth_image_id = push.uses.depth_image_id;
#endif
#endif

#if defined(__cplusplus)

inline auto trace_shadows(RecordContext &record_ctx, GbufferDepth &gbuffer_depth, VoxelWorld::Buffers &voxel_buffers) -> daxa::TaskImageView {
    auto shadow_mask = record_ctx.task_graph.create_transient_image({
        .format = daxa::Format::R8_UNORM,
        .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
        .name = "shadow_mask",
    });

    record_ctx.add(ComputeTask<TraceSecondaryCompute, TraceSecondaryComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"trace_secondary.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{TraceSecondaryCompute::gpu_input, record_ctx.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{TraceSecondaryCompute::globals, record_ctx.task_globals_buffer}},
            daxa::TaskViewVariant{std::pair{TraceSecondaryCompute::shadow_mask, shadow_mask}},
            VOXELS_BUFFER_USES_ASSIGN(TraceSecondaryCompute, voxel_buffers),
            daxa::TaskViewVariant{std::pair{TraceSecondaryCompute::blue_noise_vec2, record_ctx.task_blue_noise_vec2_image}},
            daxa::TaskViewVariant{std::pair{TraceSecondaryCompute::g_buffer_image_id, gbuffer_depth.gbuffer}},
            daxa::TaskViewVariant{std::pair{TraceSecondaryCompute::depth_image_id, gbuffer_depth.depth.task_resources.output_resource}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TraceSecondaryComputePush &push, NoTaskInfo const &) {
            auto const image_info = ti.device.info_image(ti.get(TraceSecondaryCompute::g_buffer_image_id).ids[0]).value();
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
            ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
        },
    });

    AppUi::DebugDisplay::s_instance->passes.push_back({.name = "trace shadow bitmap", .task_image_id = shadow_mask, .type = DEBUG_IMAGE_TYPE_DEFAULT_UINT});

    return daxa::TaskImageView{shadow_mask};
}

#endif
