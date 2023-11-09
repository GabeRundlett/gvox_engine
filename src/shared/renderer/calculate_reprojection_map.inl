#pragma once

#include <shared/core.inl>

#if CALCULATE_REPROJECTION_MAP_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(CalculateReprojectionMapComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(vs_normal_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(depth_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(prev_depth_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(velocity_image_id, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(dst_image_id, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif

#if defined(__cplusplus)

struct CalculateReprojectionMapComputeTaskState {
    AsyncManagedComputePipeline pipeline;

    CalculateReprojectionMapComputeTaskState(AsyncPipelineManager &pipeline_manager) {
        pipeline = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"calculate_reprojection_map.comp.glsl"},
                .compile_options = {.defines = {{"CALCULATE_REPROJECTION_MAP_COMPUTE", "1"}}, .enable_debug_info = true},
            },
            .name = "calculate_reprojection_map",
        });
    }

    void record_commands(daxa::CommandList &cmd_list, u32vec2 render_size) {
        if (!pipeline.is_valid()) {
            return;
        }
        cmd_list.set_pipeline(pipeline.get());
        // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
        cmd_list.dispatch((render_size.x + 7) / 8, (render_size.y + 7) / 8);
    }
};

struct CalculateReprojectionMapComputeTask : CalculateReprojectionMapComputeUses {
    CalculateReprojectionMapComputeTaskState *state;
    u32vec2 render_size;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        auto const &image_info = ti.get_device().info_image(uses.dst_image_id.image());
        state->record_commands(cmd_list, {image_info.size.x, image_info.size.y});
    }
};

struct ReprojectionRenderer {
    CalculateReprojectionMapComputeTaskState calculate_reprojection_map_task_state;

    ReprojectionRenderer(AsyncPipelineManager &pipeline_manager)
        : calculate_reprojection_map_task_state{pipeline_manager} {
    }

    auto calculate_reprojection_map(RecordContext &record_ctx, GbufferDepth const &gbuffer_depth, daxa::TaskImageView velocity_image) -> daxa::TaskImageView {
        auto reprojection_map = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "reprojection_image",
        });
        record_ctx.task_graph.add_task(CalculateReprojectionMapComputeTask{
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .globals = record_ctx.task_globals_buffer,
                    .vs_normal_image_id = gbuffer_depth.geometric_normal,
                    .depth_image_id = gbuffer_depth.depth.task_resources.output_resource,
                    .prev_depth_image_id = gbuffer_depth.depth.task_resources.history_resource,
                    .velocity_image_id = velocity_image,
                    .dst_image_id = reprojection_map,
                },
            },
            &calculate_reprojection_map_task_state,
        });
        return reprojection_map;
    }
};

#endif
