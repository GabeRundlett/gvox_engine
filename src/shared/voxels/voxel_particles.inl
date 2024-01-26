#pragma once

#include <shared/core.inl>

#if VoxelParticleSimComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(VoxelParticleSimCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(GpuGlobals), globals)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(SimulatedVoxelParticle), simulated_voxel_particles)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), rendered_voxel_particles)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), placed_voxel_particles)
DAXA_DECL_TASK_HEAD_END
struct VoxelParticleSimComputePush {
    VoxelParticleSimCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(VoxelParticleSimComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_BufferPtr)
daxa_RWBufferPtr(SimulatedVoxelParticle) simulated_voxel_particles = push.uses.simulated_voxel_particles;
daxa_RWBufferPtr(daxa_u32) rendered_voxel_particles = push.uses.rendered_voxel_particles;
daxa_RWBufferPtr(daxa_u32) placed_voxel_particles = push.uses.placed_voxel_particles;
#endif
#endif

#if VoxelParticleRasterShader || defined(__cplusplus)
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

struct VoxelParticles {
    daxa::BufferId simulated_voxel_particles_buffer;
    daxa::BufferId rendered_voxel_particles_buffer;
    daxa::BufferId placed_voxel_particles_buffer;
    daxa::TaskBuffer task_simulated_voxel_particles_buffer{{.name = "task_simulated_voxel_particles_buffer"}};
    daxa::TaskBuffer task_rendered_voxel_particles_buffer{{.name = "task_rendered_voxel_particles_buffer"}};
    daxa::TaskBuffer task_placed_voxel_particles_buffer{{.name = "task_placed_voxel_particles_buffer"}};

    void create(daxa::Device &device) {
        simulated_voxel_particles_buffer = device.create_buffer({
            .size = sizeof(SimulatedVoxelParticle) * std::max<daxa_u32>(MAX_SIMULATED_VOXEL_PARTICLES, 1),
            .name = "simulated_voxel_particles_buffer",
        });
        rendered_voxel_particles_buffer = device.create_buffer({
            .size = sizeof(daxa_u32) * std::max<daxa_u32>(MAX_RENDERED_VOXEL_PARTICLES, 1),
            .name = "rendered_voxel_particles_buffer",
        });
        placed_voxel_particles_buffer = device.create_buffer({
            .size = sizeof(daxa_u32) * std::max<daxa_u32>(MAX_SIMULATED_VOXEL_PARTICLES, 1),
            .name = "placed_voxel_particles_buffer",
        });
        task_simulated_voxel_particles_buffer.set_buffers({.buffers = std::array{simulated_voxel_particles_buffer}});
        task_rendered_voxel_particles_buffer.set_buffers({.buffers = std::array{rendered_voxel_particles_buffer}});
        task_placed_voxel_particles_buffer.set_buffers({.buffers = std::array{placed_voxel_particles_buffer}});
    }

    void destroy(daxa::Device &device) {
        device.destroy_buffer(simulated_voxel_particles_buffer);
        device.destroy_buffer(rendered_voxel_particles_buffer);
        device.destroy_buffer(placed_voxel_particles_buffer);
    }

    void use_buffers(RecordContext &record_ctx) {
        record_ctx.task_graph.use_persistent_buffer(task_simulated_voxel_particles_buffer);
        record_ctx.task_graph.use_persistent_buffer(task_rendered_voxel_particles_buffer);
        record_ctx.task_graph.use_persistent_buffer(task_placed_voxel_particles_buffer);
    }

    void simulate(RecordContext &record_ctx, VoxelWorld::Buffers &voxel_world_buffers) {
        if constexpr (MAX_RENDERED_VOXEL_PARTICLES == 0) {
            return;
        }
        record_ctx.add(ComputeTask<VoxelParticleSimCompute, VoxelParticleSimComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"voxels/voxel_particle_sim.comp.glsl"},
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                VOXELS_BUFFER_USES_ASSIGN(voxel_world_buffers),
                .simulated_voxel_particles = task_simulated_voxel_particles_buffer,
                .rendered_voxel_particles = task_rendered_voxel_particles_buffer,
                .placed_voxel_particles = task_placed_voxel_particles_buffer,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, VoxelParticleSimCompute::Uses &uses, VoxelParticleSimComputePush &push, NoTaskInfo const &) {
                ti.get_recorder().set_pipeline(pipeline);
                ti.get_recorder().push_constant(push);
                ti.get_recorder().dispatch_indirect({
                    .indirect_buffer = uses.globals.buffer(),
                    .offset = offsetof(GpuGlobals, voxel_particles_state) + offsetof(VoxelParticlesState, simulation_dispatch),
                });
            },
        });
    }

    auto render(RecordContext &record_ctx) -> std::pair<daxa::TaskImageView, daxa::TaskImageView> {
        auto format = daxa::Format::R32_UINT;
        auto raster_color_image = record_ctx.task_graph.create_transient_image({
            .format = format,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "raster_color_image",
        });
        auto raster_depth_image = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::D32_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "raster_depth_image",
        });

        if constexpr (MAX_RENDERED_VOXEL_PARTICLES == 0) {
            return {raster_color_image, raster_depth_image};
        }

        record_ctx.add(RasterTask<VoxelParticleRaster, VoxelParticleRasterPush, NoTaskInfo>{
            .vert_source = daxa::ShaderFile{"voxel_particle.raster.glsl"},
            .frag_source = daxa::ShaderFile{"voxel_particle.raster.glsl"},
            .color_attachments = {{
                .format = format,
            }},
            .depth_test = daxa::DepthTestInfo{
                .depth_attachment_format = daxa::Format::D32_SFLOAT,
                .enable_depth_write = true,
                .depth_test_compare_op = daxa::CompareOp::GREATER,
            },
            .raster = {
                .face_culling = daxa::FaceCullFlagBits::BACK_BIT,
            },
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                .simulated_voxel_particles = task_simulated_voxel_particles_buffer,
                .rendered_voxel_particles = task_rendered_voxel_particles_buffer,
                .render_image = raster_color_image,
                .depth_image_id = raster_depth_image,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, VoxelParticleRaster::Uses &uses, VoxelParticleRasterPush &push, NoTaskInfo const &) {
                auto const &image_info = ti.get_device().info_image(uses.render_image.image()).value();
                auto renderpass_recorder = std::move(ti.get_recorder()).begin_renderpass({
                    .color_attachments = {{.image_view = uses.render_image.image().default_view(), .load_op = daxa::AttachmentLoadOp::CLEAR, .clear_value = std::array<daxa_f32, 4>{0.0f, 0.0f, 0.0f, 0.0f}}},
                    .depth_attachment = {{.image_view = uses.depth_image_id.image().default_view(), .load_op = daxa::AttachmentLoadOp::CLEAR, .clear_value = daxa::DepthValue{0.0f, 0}}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                renderpass_recorder.push_constant(push);
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = uses.globals.buffer(),
                    .indirect_buffer_offset = offsetof(GpuGlobals, voxel_particles_state) + offsetof(VoxelParticlesState, draw_params),
                    .is_indexed = false,
                });
                ti.get_recorder() = std::move(renderpass_recorder).end_renderpass();
            },
        });

        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "voxel particles", .task_image_id = raster_color_image, .type = DEBUG_IMAGE_TYPE_DEFAULT_UINT});
        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "voxel particles depth", .task_image_id = raster_depth_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        return {raster_color_image, raster_depth_image};
    }
};

#endif
