#pragma once

#include <shared/core.inl>

DAXA_DECL_TASK_USES_BEGIN(VoxelParticleRasterUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(simulated_voxel_particles, daxa_BufferPtr(SimulatedVoxelParticle), SHADER_READ)
DAXA_TASK_USE_BUFFER(rendered_voxel_particles, daxa_BufferPtr(daxa_u32), SHADER_READ)
DAXA_TASK_USE_IMAGE(render_image, REGULAR_2D, COLOR_ATTACHMENT)
DAXA_TASK_USE_IMAGE(depth_image, REGULAR_2D, DEPTH_ATTACHMENT)
DAXA_DECL_TASK_USES_END()

#if defined(__cplusplus)

struct VoxelParticleRasterTaskState {
    daxa::PipelineManager &pipeline_manager;
    AppUi &ui;
    std::shared_ptr<daxa::RasterPipeline> pipeline;

    void compile_pipeline() {
        auto compile_result = pipeline_manager.add_raster_pipeline({
            .vertex_shader_info = daxa::ShaderCompileInfo{.source = daxa::ShaderFile{"voxel_particle.raster.glsl"}, .compile_options = {.defines = {{"VOXEL_PARTICLE_RASTER", "1"}}}},
            .fragment_shader_info = daxa::ShaderCompileInfo{.source = daxa::ShaderFile{"voxel_particle.raster.glsl"}, .compile_options = {.defines = {{"VOXEL_PARTICLE_RASTER", "1"}}}},
            .color_attachments = {{
                .format = daxa::Format::R32G32B32A32_SFLOAT,
            }},
            .depth_test = {
                .depth_attachment_format = daxa::Format::D32_SFLOAT,
                .enable_depth_test = true,
                .enable_depth_write = true,
                .depth_test_compare_op = daxa::CompareOp::GREATER,
            },
            .raster = {
#if USE_POINTS
                .polygon_mode = daxa::PolygonMode::POINT,
#endif
                .face_culling = daxa::FaceCullFlagBits::BACK_BIT,
            },
            .name = "voxel_particle_sim",
        });
        if (compile_result.is_err()) {
            ui.console.add_log(compile_result.message());
            return;
        }
        pipeline = compile_result.value();
        if (!compile_result.value()->is_valid()) {
            ui.console.add_log(compile_result.message());
        }
    }

    VoxelParticleRasterTaskState(daxa::PipelineManager &a_pipeline_manager, AppUi &a_ui) : pipeline_manager{a_pipeline_manager}, ui{a_ui} { compile_pipeline(); }
    auto pipeline_is_valid() -> bool { return pipeline && pipeline->is_valid(); }

    void record_commands(daxa::CommandList &cmd_list, daxa::BufferId globals_buffer_id, daxa::ImageId render_image, daxa::ImageId depth_image, u32vec2 size) {
        if (!pipeline_is_valid()) {
            return;
        }
        cmd_list.begin_renderpass({
            .color_attachments = {{.image_view = render_image.default_view(), .load_op = daxa::AttachmentLoadOp::CLEAR, .clear_value = std::array<f32, 4>{0.0f, 0.0f, 0.0f, 0.0f}}},
            .depth_attachment = {{.image_view = depth_image.default_view(), .load_op = daxa::AttachmentLoadOp::CLEAR, .clear_value = daxa::DepthValue{0.0f, 0}}},
            .render_area = {.x = 0, .y = 0, .width = size.x, .height = size.y},
        });
        cmd_list.set_pipeline(*pipeline);
        cmd_list.draw_indirect({
            .draw_command_buffer = globals_buffer_id,
            .draw_command_buffer_read_offset = offsetof(GpuGlobals, voxel_particles_state) + offsetof(VoxelParticlesState, draw_params),
            .is_indexed = false,
        });
        cmd_list.end_renderpass();
    }
};

struct VoxelParticleRasterTask : VoxelParticleRasterUses {
    VoxelParticleRasterTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        auto const &image_info = ti.get_device().info_image(uses.render_image.image());
        state->record_commands(cmd_list, uses.globals.buffer(), uses.render_image.image(), uses.depth_image.image(), {image_info.size.x, image_info.size.y});
    }
};

#endif
