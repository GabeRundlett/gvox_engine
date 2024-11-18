#pragma once

#include "../common.inl"

#define MAX_SIMULATED_VOXEL_PARTICLES (1 << 16)

struct SimulatedVoxelParticle {
    daxa_f32vec3 pos;
    daxa_f32 duration_alive;
    daxa_f32vec3 vel;
    PackedVoxel packed_voxel;
    daxa_u32 flags;
};
DAXA_DECL_BUFFER_PTR(SimulatedVoxelParticle)

DAXA_DECL_TASK_HEAD_BEGIN(SimParticleSimCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(SimulatedVoxelParticle), simulated_voxel_particles)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(PackedParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(PackedParticleVertex), shadow_cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(PackedParticleVertex), splat_rendered_particle_verts)
DAXA_DECL_TASK_HEAD_END
struct SimParticleSimComputePush {
    DAXA_TH_BLOB(SimParticleSimCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(SimParticleCubeParticleRaster)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(DRAW_INDIRECT_INFO_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(PackedParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(SimulatedVoxelParticle), simulated_voxel_particles)
DAXA_TH_BUFFER(INDEX_READ, indices)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct SimParticleCubeParticleRasterPush {
    DAXA_TH_BLOB(SimParticleCubeParticleRaster, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(SimParticleCubeParticleShadowRaster)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(DRAW_INDIRECT_INFO_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(PackedParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(SimulatedVoxelParticle), simulated_voxel_particles)
DAXA_TH_BUFFER(INDEX_READ, indices)
DAXA_TH_IMAGE_INDEX(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct SimParticleCubeParticleShadowRasterPush {
    DAXA_TH_BLOB(SimParticleCubeParticleShadowRaster, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(SimParticleSplatParticleRaster)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(DRAW_INDIRECT_INFO_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(PackedParticleVertex), splat_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(SimulatedVoxelParticle), simulated_voxel_particles)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct SimParticleSplatParticleRasterPush {
    DAXA_TH_BLOB(SimParticleSplatParticleRaster, uses)
};

#if defined(__cplusplus)

struct SimParticles {
    TemporalBuffer cube_rendered_particle_verts;
    TemporalBuffer shadow_cube_rendered_particle_verts;
    TemporalBuffer splat_rendered_particle_verts;
    TemporalBuffer simulated_voxel_particles;

    void simulate(GpuContext &gpu_context, VoxelWorldBuffers &voxel_world_buffers, daxa::TaskBufferView particles_state) {
        cube_rendered_particle_verts = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(PackedParticleVertex) * std::max<daxa_u32>(MAX_SIMULATED_VOXEL_PARTICLES, 1),
            .name = "sim_particles.cube_rendered_particle_verts",
        });
        shadow_cube_rendered_particle_verts = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(PackedParticleVertex) * std::max<daxa_u32>(MAX_SIMULATED_VOXEL_PARTICLES, 1),
            .name = "sim_particles.shadow_cube_rendered_particle_verts",
        });
        splat_rendered_particle_verts = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(PackedParticleVertex) * std::max<daxa_u32>(MAX_SIMULATED_VOXEL_PARTICLES, 1),
            .name = "sim_particles.splat_rendered_particle_verts",
        });
        simulated_voxel_particles = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(SimulatedVoxelParticle) * std::max<daxa_u32>(MAX_SIMULATED_VOXEL_PARTICLES, 1),
            .name = "sim_particles.simulated_voxel_particles",
        });

        gpu_context.frame_task_graph.use_persistent_buffer(cube_rendered_particle_verts.task_resource);
        gpu_context.frame_task_graph.use_persistent_buffer(shadow_cube_rendered_particle_verts.task_resource);
        gpu_context.frame_task_graph.use_persistent_buffer(splat_rendered_particle_verts.task_resource);
        gpu_context.frame_task_graph.use_persistent_buffer(simulated_voxel_particles.task_resource);

        gpu_context.add(ComputeTask<SimParticleSimCompute::Task, SimParticleSimComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"voxels/particles/sim_particle/sim.comp.glsl"},
            .extra_defines = {daxa::ShaderDefine{.name = "SIM_PARTICLE", .value = "1"}},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{SimParticleSimCompute::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{SimParticleSimCompute::AT.particles_state, particles_state}},
                VOXELS_BUFFER_USES_ASSIGN(SimParticleSimCompute, voxel_world_buffers),
                daxa::TaskViewVariant{std::pair{SimParticleSimCompute::AT.simulated_voxel_particles, simulated_voxel_particles.task_resource}},
                daxa::TaskViewVariant{std::pair{SimParticleSimCompute::AT.cube_rendered_particle_verts, cube_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{SimParticleSimCompute::AT.shadow_cube_rendered_particle_verts, shadow_cube_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{SimParticleSimCompute::AT.splat_rendered_particle_verts, splat_rendered_particle_verts.task_resource}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SimParticleSimComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch_indirect({
                    .indirect_buffer = ti.get(SimParticleSimCompute::AT.particles_state).ids[0],
                    .offset = offsetof(VoxelParticlesState, simulation_dispatch),
                });
            },
        });
    }

    void render_cubes(GpuContext &gpu_context, GbufferDepth &gbuffer_depth, daxa::TaskImageView velocity_image, daxa::TaskImageView shadow_depth, daxa::TaskBufferView particles_state, daxa::TaskBufferView cube_index_buffer) {
        gpu_context.add(RasterTask<SimParticleCubeParticleRaster::Task, SimParticleCubeParticleRasterPush, NoTaskInfo>{
            .vert_source = daxa::ShaderFile{"voxels/particles/cube.raster.glsl"},
            .frag_source = daxa::ShaderFile{"voxels/particles/cube.raster.glsl"},
            .color_attachments = {
                {.format = daxa::Format::R32G32B32A32_UINT},
                {.format = daxa::Format::R16G16B16A16_SFLOAT},
                {.format = daxa::Format::A2B10G10R10_UNORM_PACK32},
            },
            .depth_test = daxa::DepthTestInfo{
                .depth_attachment_format = daxa::Format::D32_SFLOAT,
                .enable_depth_write = true,
                .depth_test_compare_op = daxa::CompareOp::GREATER,
            },
            .raster = {
                .primitive_topology = daxa::PrimitiveTopology::TRIANGLE_FAN,
                .face_culling = daxa::FaceCullFlagBits::NONE,
            },
            .extra_defines = {daxa::ShaderDefine{.name = "SIM_PARTICLE", .value = "1"}},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{SimParticleCubeParticleRaster::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{SimParticleCubeParticleRaster::AT.particles_state, particles_state}},
                daxa::TaskViewVariant{std::pair{SimParticleCubeParticleRaster::AT.cube_rendered_particle_verts, cube_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{SimParticleCubeParticleRaster::AT.indices, cube_index_buffer}},
                daxa::TaskViewVariant{std::pair{SimParticleCubeParticleRaster::AT.simulated_voxel_particles, simulated_voxel_particles.task_resource}},
                daxa::TaskViewVariant{std::pair{SimParticleCubeParticleRaster::AT.g_buffer_image_id, gbuffer_depth.gbuffer}},
                daxa::TaskViewVariant{std::pair{SimParticleCubeParticleRaster::AT.velocity_image_id, velocity_image}},
                daxa::TaskViewVariant{std::pair{SimParticleCubeParticleRaster::AT.vs_normal_image_id, gbuffer_depth.geometric_normal}},
                daxa::TaskViewVariant{std::pair{SimParticleCubeParticleRaster::AT.depth_image_id, gbuffer_depth.depth.current()}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, SimParticleCubeParticleRasterPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(SimParticleCubeParticleRaster::AT.g_buffer_image_id).ids[0]).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .color_attachments = {
                        {.image_view = ti.get(SimParticleCubeParticleRaster::AT.g_buffer_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                        {.image_view = ti.get(SimParticleCubeParticleRaster::AT.velocity_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                        {.image_view = ti.get(SimParticleCubeParticleRaster::AT.vs_normal_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                    },
                    .depth_attachment = {{.image_view = ti.get(SimParticleCubeParticleRaster::AT.depth_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.set_index_buffer({
                    .id = ti.get(SimParticleCubeParticleRaster::AT.indices).ids[0],
                    .index_type = daxa::IndexType::uint16,
                });
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(SimParticleCubeParticleRaster::AT.particles_state).ids[0],
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, sim_particle) + offsetof(ParticleDrawParams, cube_draw_params),
                    .is_indexed = true,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });

        gpu_context.add(RasterTask<SimParticleCubeParticleShadowRaster::Task, SimParticleCubeParticleShadowRasterPush, NoTaskInfo>{
            .vert_source = daxa::ShaderFile{"voxels/particles/cube.raster.glsl"},
            .frag_source = daxa::ShaderFile{"voxels/particles/cube.raster.glsl"},
            .depth_test = daxa::DepthTestInfo{
                .depth_attachment_format = daxa::Format::D32_SFLOAT,
                .enable_depth_write = true,
                .depth_test_compare_op = daxa::CompareOp::GREATER,
            },
            .raster = {
                .primitive_topology = daxa::PrimitiveTopology::TRIANGLE_FAN,
                .face_culling = daxa::FaceCullFlagBits::NONE,
            },
            .extra_defines = {daxa::ShaderDefine{.name = "SIM_PARTICLE", .value = "1"}, daxa::ShaderDefine{.name = "SHADOW_MAP", .value = "1"}},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{SimParticleCubeParticleShadowRaster::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{SimParticleCubeParticleShadowRaster::AT.particles_state, particles_state}},
                daxa::TaskViewVariant{std::pair{SimParticleCubeParticleShadowRaster::AT.cube_rendered_particle_verts, shadow_cube_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{SimParticleCubeParticleShadowRaster::AT.simulated_voxel_particles, simulated_voxel_particles.task_resource}},
                daxa::TaskViewVariant{std::pair{SimParticleCubeParticleShadowRaster::AT.indices, cube_index_buffer}},
                daxa::TaskViewVariant{std::pair{SimParticleCubeParticleShadowRaster::AT.depth_image_id, shadow_depth}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, SimParticleCubeParticleShadowRasterPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(SimParticleCubeParticleShadowRaster::AT.depth_image_id).ids[0]).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .depth_attachment = {{.image_view = ti.get(SimParticleCubeParticleShadowRaster::AT.depth_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::CLEAR, .clear_value = daxa::DepthValue{0.0f, 0}}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.set_index_buffer({
                    .id = ti.get(SimParticleCubeParticleShadowRaster::AT.indices).ids[0],
                    .index_type = daxa::IndexType::uint16,
                });
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(SimParticleCubeParticleShadowRaster::AT.particles_state).ids[0],
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, sim_particle) + offsetof(ParticleDrawParams, shadow_cube_draw_params),
                    .is_indexed = true,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });
        debug_utils::DebugDisplay::add_pass({.name = "voxel particle shadow depth", .task_image_id = shadow_depth, .type = DEBUG_IMAGE_TYPE_DEFAULT});
    }

    void render_splats(GpuContext &gpu_context, GbufferDepth &gbuffer_depth, daxa::TaskImageView velocity_image, daxa::TaskImageView shadow_depth, daxa::TaskBufferView particles_state) {
        gpu_context.add(RasterTask<SimParticleSplatParticleRaster::Task, SimParticleSplatParticleRasterPush, NoTaskInfo>{
            .vert_source = daxa::ShaderFile{"voxels/particles/splat.raster.glsl"},
            .frag_source = daxa::ShaderFile{"voxels/particles/splat.raster.glsl"},
            .color_attachments = {
                {.format = daxa::Format::R32G32B32A32_UINT},
                {.format = daxa::Format::R16G16B16A16_SFLOAT},
                {.format = daxa::Format::A2B10G10R10_UNORM_PACK32},
            },
            .depth_test = daxa::DepthTestInfo{
                .depth_attachment_format = daxa::Format::D32_SFLOAT,
                .enable_depth_write = true,
                .depth_test_compare_op = daxa::CompareOp::GREATER,
            },
            .raster = {
                .primitive_topology = daxa::PrimitiveTopology::POINT_LIST,
                .face_culling = daxa::FaceCullFlagBits::NONE,
            },
            .extra_defines = {daxa::ShaderDefine{.name = "SIM_PARTICLE", .value = "1"}},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{SimParticleSplatParticleRaster::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{SimParticleSplatParticleRaster::AT.particles_state, particles_state}},
                daxa::TaskViewVariant{std::pair{SimParticleSplatParticleRaster::AT.splat_rendered_particle_verts, splat_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{SimParticleSplatParticleRaster::AT.simulated_voxel_particles, simulated_voxel_particles.task_resource}},
                daxa::TaskViewVariant{std::pair{SimParticleSplatParticleRaster::AT.g_buffer_image_id, gbuffer_depth.gbuffer}},
                daxa::TaskViewVariant{std::pair{SimParticleSplatParticleRaster::AT.velocity_image_id, velocity_image}},
                daxa::TaskViewVariant{std::pair{SimParticleSplatParticleRaster::AT.vs_normal_image_id, gbuffer_depth.geometric_normal}},
                daxa::TaskViewVariant{std::pair{SimParticleSplatParticleRaster::AT.depth_image_id, gbuffer_depth.depth.current()}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, SimParticleSplatParticleRasterPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(SimParticleSplatParticleRaster::AT.g_buffer_image_id).ids[0]).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .color_attachments = {
                        {.image_view = ti.get(SimParticleSplatParticleRaster::AT.g_buffer_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                        {.image_view = ti.get(SimParticleSplatParticleRaster::AT.velocity_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                        {.image_view = ti.get(SimParticleSplatParticleRaster::AT.vs_normal_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                    },
                    .depth_attachment = {{.image_view = ti.get(SimParticleSplatParticleRaster::AT.depth_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(SimParticleSplatParticleRaster::AT.particles_state).ids[0],
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, sim_particle) + offsetof(ParticleDrawParams, splat_draw_params),
                    .is_indexed = false,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });
    }
};

#endif
