#pragma once

#include <shared/core.inl>

#if VOXEL_PARTICLE_RASTER || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(VoxelParticleRasterUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), FRAGMENT_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), FRAGMENT_SHADER_READ)
DAXA_TASK_USE_BUFFER(simulated_voxel_particles, daxa_BufferPtr(SimulatedVoxelParticle), FRAGMENT_SHADER_READ)
DAXA_TASK_USE_BUFFER(rendered_voxel_particles, daxa_BufferPtr(daxa_u32), FRAGMENT_SHADER_READ)
DAXA_TASK_USE_IMAGE(render_image, REGULAR_2D, COLOR_ATTACHMENT)
DAXA_TASK_USE_IMAGE(depth_image_id, REGULAR_2D, DEPTH_ATTACHMENT)
DAXA_DECL_TASK_USES_END()
#endif

#if defined(__cplusplus)

struct VoxelParticleRasterTaskState {
    AsyncManagedRasterPipeline pipeline;

    VoxelParticleRasterTaskState(AsyncPipelineManager &pipeline_manager) {
        pipeline = pipeline_manager.add_raster_pipeline({
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
                .face_culling = daxa::FaceCullFlagBits::BACK_BIT,
            },
            .name = "voxel_particle_sim",
        });
    }

    void record_commands(daxa::CommandList &cmd_list, daxa::BufferId globals_buffer_id, daxa::ImageId render_image, daxa::ImageId depth_image_id, u32vec2 size) {
        if (!pipeline.is_valid()) {
            return;
        }
        cmd_list.begin_renderpass({
            .color_attachments = {{.image_view = render_image.default_view(), .load_op = daxa::AttachmentLoadOp::CLEAR, .clear_value = std::array<f32, 4>{0.0f, 0.0f, 0.0f, 0.0f}}},
            .depth_attachment = {{.image_view = depth_image_id.default_view(), .load_op = daxa::AttachmentLoadOp::CLEAR, .clear_value = daxa::DepthValue{0.0f, 0}}},
            .render_area = {.x = 0, .y = 0, .width = size.x, .height = size.y},
        });
        cmd_list.set_pipeline(pipeline.get());
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
        state->record_commands(cmd_list, uses.globals.buffer(), uses.render_image.image(), uses.depth_image_id.image(), {image_info.size.x, image_info.size.y});
    }
};

#endif
