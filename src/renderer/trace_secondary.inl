#pragma once

#include <core.inl>
#include <renderer/core.inl>

DAXA_DECL_TASK_HEAD_BEGIN(TraceSecondaryCompute, 6 + VOXEL_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, shadow_mask)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_3D, blue_noise_vec2)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, particles_shadow_depth_tex)
DAXA_DECL_TASK_HEAD_END
struct TraceSecondaryComputePush {
    DAXA_TH_BLOB(TraceSecondaryCompute, uses)
};

#if defined(__cplusplus)

#include <application/settings.hpp>

inline auto trace_shadows(GpuContext &gpu_context, GbufferDepth &gbuffer_depth, VoxelWorldBuffers &voxel_buffers, daxa::TaskImageView particles_shadow_depth_image) -> daxa::TaskImageView {
    auto shadow_mask = gpu_context.frame_task_graph.create_transient_image({
        .format = daxa::Format::R8_UNORM,
        .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
        .name = "shadow_mask",
    });

    AppSettings::add<settings::Checkbox>({"Graphics", "Render Shadows", {.value = true}, {.task_graph_depends = true}});

    auto render_shadows = AppSettings::get<settings::Checkbox>("Graphics", "Render Shadows").value;

    if (render_shadows) {
        gpu_context.add(ComputeTask<TraceSecondaryCompute, TraceSecondaryComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"trace_secondary.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TraceSecondaryCompute::gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{TraceSecondaryCompute::shadow_mask, shadow_mask}},
                VOXELS_BUFFER_USES_ASSIGN(TraceSecondaryCompute, voxel_buffers),
                daxa::TaskViewVariant{std::pair{TraceSecondaryCompute::blue_noise_vec2, gpu_context.task_blue_noise_vec2_image}},
                daxa::TaskViewVariant{std::pair{TraceSecondaryCompute::g_buffer_image_id, gbuffer_depth.gbuffer}},
                daxa::TaskViewVariant{std::pair{TraceSecondaryCompute::depth_image_id, gbuffer_depth.depth.current()}},
                daxa::TaskViewVariant{std::pair{TraceSecondaryCompute::particles_shadow_depth_tex, particles_shadow_depth_image}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TraceSecondaryComputePush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(TraceSecondaryCompute::g_buffer_image_id).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
                ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });
    } else {
        clear_task_images(gpu_context.frame_task_graph, std::array<daxa::TaskImageView, 1>{shadow_mask}, std::array<daxa::ClearValue, 1>{std::array<float, 4>{1.0f, 1.0f, 1.0f, 1.0f}});
    }

    debug_utils::DebugDisplay::add_pass({.name = "trace shadow bitmap", .task_image_id = shadow_mask, .type = DEBUG_IMAGE_TYPE_DEFAULT_UINT});

    return daxa::TaskImageView{shadow_mask};
}

#endif
