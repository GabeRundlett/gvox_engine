#pragma once

#include <shared/core.inl>

#if VOXEL_PARTICLE_RASTER || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(VoxelParticleRaster)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_BufferPtr(SimulatedVoxelParticle), simulated_voxel_particles)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_BufferPtr(daxa_u32), rendered_voxel_particles)
DAXA_TH_IMAGE_ID(COLOR_ATTACHMENT, REGULAR_2D, render_image)
DAXA_TH_IMAGE_ID(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct VoxelParticleRasterPush {
    VoxelParticleRaster uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(VoxelParticleRasterPush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_BufferPtr(SimulatedVoxelParticle) simulated_voxel_particles = push.uses.simulated_voxel_particles;
daxa_BufferPtr(daxa_u32) rendered_voxel_particles = push.uses.rendered_voxel_particles;
daxa_ImageViewId render_image = push.uses.render_image;
daxa_ImageViewId depth_image_id = push.uses.depth_image_id;
#endif
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
            .depth_test = daxa::DepthTestInfo{
                .depth_attachment_format = daxa::Format::D32_SFLOAT,
                // .enable_depth_test = true,
                .enable_depth_write = true,
                .depth_test_compare_op = daxa::CompareOp::GREATER,
            },
            .raster = {
                .face_culling = daxa::FaceCullFlagBits::BACK_BIT,
            },
            .push_constant_size = sizeof(VoxelParticleRasterPush),
            .name = "voxel_particle_sim",
        });
    }

    void record_commands(VoxelParticleRasterPush const &push, daxa::CommandRecorder &recorder, daxa::BufferId globals_buffer_id, daxa::ImageId render_image, daxa::ImageId depth_image_id, daxa_u32vec2 size) {
        if (!pipeline.is_valid()) {
            return;
        }
        auto renderpass_recorder = std::move(recorder).begin_renderpass({
            .color_attachments = {{.image_view = render_image.default_view(), .load_op = daxa::AttachmentLoadOp::CLEAR, .clear_value = std::array<daxa_f32, 4>{0.0f, 0.0f, 0.0f, 0.0f}}},
            .depth_attachment = {{.image_view = depth_image_id.default_view(), .load_op = daxa::AttachmentLoadOp::CLEAR, .clear_value = daxa::DepthValue{0.0f, 0}}},
            .render_area = {.x = 0, .y = 0, .width = size.x, .height = size.y},
        });
        renderpass_recorder.set_pipeline(pipeline.get());
        renderpass_recorder.push_constant(push);
        renderpass_recorder.draw_indirect({
            .draw_command_buffer = globals_buffer_id,
            .indirect_buffer_offset = offsetof(GpuGlobals, voxel_particles_state) + offsetof(VoxelParticlesState, draw_params),
            .is_indexed = false,
        });
        recorder = std::move(renderpass_recorder).end_renderpass();
    }
};

struct VoxelParticleRasterTask {
    VoxelParticleRaster::Uses uses;
    std::string name = "VoxelParticleRaster";
    VoxelParticleRasterTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto const &image_info = ti.get_device().info_image(uses.render_image.image()).value();
        auto push = VoxelParticleRasterPush{};
        ti.copy_task_head_to(&push.uses);
        state->record_commands(push, recorder, uses.globals.buffer(), uses.render_image.image(), uses.depth_image_id.image(), {image_info.size.x, image_info.size.y});
    }
};

#endif
