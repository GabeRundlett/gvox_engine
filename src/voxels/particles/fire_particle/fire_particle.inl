#pragma once

#include "../common.inl"

#define MAX_FIRE_PARTICLES (1 << 16)

struct FireParticle {
    daxa_f32vec3 origin;
    PackedVoxel packed_voxel;
    daxa_u32 flags;
};
DAXA_DECL_BUFFER_PTR(FireParticle)

DECL_SIMPLE_STATIC_ALLOCATOR(FireParticleAllocator, FireParticle, MAX_FIRE_PARTICLES, daxa_u32)

DAXA_DECL_TASK_HEAD_BEGIN(FireParticleSimCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
SIMPLE_STATIC_ALLOCATOR_USE_BUFFERS(COMPUTE_SHADER_READ_WRITE, FireParticleAllocator)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(PackedParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(PackedParticleVertex), shadow_cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(PackedParticleVertex), splat_rendered_particle_verts)
DAXA_DECL_TASK_HEAD_END
struct FireParticleSimComputePush {
    DAXA_TH_BLOB(FireParticleSimCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(FireParticleCubeParticleRaster)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(DRAW_INDIRECT_INFO_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(PackedParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(FireParticle), fire_particles)
DAXA_TH_BUFFER(INDEX_READ, indices)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct FireParticleCubeParticleRasterPush {
    DAXA_TH_BLOB(FireParticleCubeParticleRaster, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(FireParticleCubeParticleShadowRaster)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(DRAW_INDIRECT_INFO_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(PackedParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(FireParticle), fire_particles)
DAXA_TH_BUFFER(INDEX_READ, indices)
DAXA_TH_IMAGE_INDEX(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct FireParticleCubeParticleShadowRasterPush {
    DAXA_TH_BLOB(FireParticleCubeParticleShadowRaster, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(FireParticleSplatParticleRaster)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(DRAW_INDIRECT_INFO_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(PackedParticleVertex), splat_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(FireParticle), fire_particles)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct FireParticleSplatParticleRasterPush {
    DAXA_TH_BLOB(FireParticleSplatParticleRaster, uses)
};

#if defined(__cplusplus)

struct FireParticles {
    TemporalBuffer cube_rendered_particle_verts;
    TemporalBuffer shadow_cube_rendered_particle_verts;
    TemporalBuffer splat_rendered_particle_verts;
    StaticAllocatorBufferState<FireParticleAllocator> fire_particle_allocator;

    void init(GpuContext &gpu_context) {
        fire_particle_allocator.init(gpu_context);
    }

    void simulate(GpuContext &gpu_context, VoxelWorldBuffers &voxel_world_buffers, daxa::TaskBufferView particles_state) {
        cube_rendered_particle_verts = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(PackedParticleVertex) * std::max<daxa_u32>(MAX_FIRE_PARTICLES, 1),
            .name = "fire_particle.cube_rendered_particle_verts",
        });
        shadow_cube_rendered_particle_verts = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(PackedParticleVertex) * std::max<daxa_u32>(MAX_FIRE_PARTICLES, 1),
            .name = "fire_particle.shadow_cube_rendered_particle_verts",
        });
        splat_rendered_particle_verts = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(PackedParticleVertex) * std::max<daxa_u32>(MAX_FIRE_PARTICLES, 1),
            .name = "fire_particle.splat_rendered_particle_verts",
        });

        gpu_context.frame_task_graph.use_persistent_buffer(cube_rendered_particle_verts.task_resource);
        gpu_context.frame_task_graph.use_persistent_buffer(shadow_cube_rendered_particle_verts.task_resource);
        gpu_context.frame_task_graph.use_persistent_buffer(splat_rendered_particle_verts.task_resource);

        gpu_context.add(ComputeTask<FireParticleSimCompute::Task, FireParticleSimComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"voxels/particles/fire_particle/sim.comp.glsl"},
            .extra_defines = {daxa::ShaderDefine{.name = "FIRE_PARTICLE", .value = "1"}},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{FireParticleSimCompute::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{FireParticleSimCompute::AT.particles_state, particles_state}},
                VOXELS_BUFFER_USES_ASSIGN(FireParticleSimCompute, voxel_world_buffers),
                SIMPLE_STATIC_ALLOCATOR_BUFFER_USES_ASSIGN(FireParticleSimCompute, FireParticleAllocator, fire_particle_allocator),
                daxa::TaskViewVariant{std::pair{FireParticleSimCompute::AT.cube_rendered_particle_verts, cube_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{FireParticleSimCompute::AT.shadow_cube_rendered_particle_verts, shadow_cube_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{FireParticleSimCompute::AT.splat_rendered_particle_verts, splat_rendered_particle_verts.task_resource}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, FireParticleSimComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(MAX_FIRE_PARTICLES + 63) / 64, 1, 1});
            },
        });
    }

    void render_cubes(GpuContext &gpu_context, GbufferDepth &gbuffer_depth, daxa::TaskImageView velocity_image, daxa::TaskImageView shadow_depth, daxa::TaskBufferView particles_state, daxa::TaskBufferView cube_index_buffer) {
        gpu_context.add(RasterTask<FireParticleCubeParticleRaster::Task, FireParticleCubeParticleRasterPush, NoTaskInfo>{
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
            .extra_defines = {daxa::ShaderDefine{.name = "FIRE_PARTICLE", .value = "1"}},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{FireParticleCubeParticleRaster::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{FireParticleCubeParticleRaster::AT.particles_state, particles_state}},
                daxa::TaskViewVariant{std::pair{FireParticleCubeParticleRaster::AT.cube_rendered_particle_verts, cube_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{FireParticleCubeParticleRaster::AT.indices, cube_index_buffer}},
                daxa::TaskViewVariant{std::pair{FireParticleCubeParticleRaster::AT.fire_particles, fire_particle_allocator.element_buffer.task_resource}},
                daxa::TaskViewVariant{std::pair{FireParticleCubeParticleRaster::AT.g_buffer_image_id, gbuffer_depth.gbuffer}},
                daxa::TaskViewVariant{std::pair{FireParticleCubeParticleRaster::AT.velocity_image_id, velocity_image}},
                daxa::TaskViewVariant{std::pair{FireParticleCubeParticleRaster::AT.vs_normal_image_id, gbuffer_depth.geometric_normal}},
                daxa::TaskViewVariant{std::pair{FireParticleCubeParticleRaster::AT.depth_image_id, gbuffer_depth.depth.current()}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, FireParticleCubeParticleRasterPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(FireParticleCubeParticleRaster::AT.g_buffer_image_id).ids[0]).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .color_attachments = {
                        {.image_view = ti.get(FireParticleCubeParticleRaster::AT.g_buffer_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                        {.image_view = ti.get(FireParticleCubeParticleRaster::AT.velocity_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                        {.image_view = ti.get(FireParticleCubeParticleRaster::AT.vs_normal_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                    },
                    .depth_attachment = {{.image_view = ti.get(FireParticleCubeParticleRaster::AT.depth_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.set_index_buffer({
                    .id = ti.get(FireParticleCubeParticleRaster::AT.indices).ids[0],
                    .index_type = daxa::IndexType::uint16,
                });
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(FireParticleCubeParticleRaster::AT.particles_state).ids[0],
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, fire_particle) + offsetof(ParticleDrawParams, cube_draw_params),
                    .is_indexed = true,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });

        gpu_context.add(RasterTask<FireParticleCubeParticleShadowRaster::Task, FireParticleCubeParticleShadowRasterPush, NoTaskInfo>{
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
            .extra_defines = {daxa::ShaderDefine{.name = "FIRE_PARTICLE", .value = "1"}, daxa::ShaderDefine{.name = "SHADOW_MAP", .value = "1"}},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{FireParticleCubeParticleShadowRaster::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{FireParticleCubeParticleShadowRaster::AT.particles_state, particles_state}},
                daxa::TaskViewVariant{std::pair{FireParticleCubeParticleShadowRaster::AT.cube_rendered_particle_verts, shadow_cube_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{FireParticleCubeParticleShadowRaster::AT.fire_particles, fire_particle_allocator.element_buffer.task_resource}},
                daxa::TaskViewVariant{std::pair{FireParticleCubeParticleShadowRaster::AT.indices, cube_index_buffer}},
                daxa::TaskViewVariant{std::pair{FireParticleCubeParticleShadowRaster::AT.depth_image_id, shadow_depth}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, FireParticleCubeParticleShadowRasterPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(FireParticleCubeParticleShadowRaster::AT.depth_image_id).ids[0]).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .depth_attachment = {{.image_view = ti.get(FireParticleCubeParticleShadowRaster::AT.depth_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.set_index_buffer({
                    .id = ti.get(FireParticleCubeParticleShadowRaster::AT.indices).ids[0],
                    .index_type = daxa::IndexType::uint16,
                });
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(FireParticleCubeParticleShadowRaster::AT.particles_state).ids[0],
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, fire_particle) + offsetof(ParticleDrawParams, shadow_cube_draw_params),
                    .is_indexed = true,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });
    }

    void render_splats(GpuContext &gpu_context, GbufferDepth &gbuffer_depth, daxa::TaskImageView velocity_image, daxa::TaskImageView shadow_depth, daxa::TaskBufferView particles_state) {
        gpu_context.add(RasterTask<FireParticleSplatParticleRaster::Task, FireParticleSplatParticleRasterPush, NoTaskInfo>{
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
            .extra_defines = {daxa::ShaderDefine{.name = "FIRE_PARTICLE", .value = "1"}},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{FireParticleSplatParticleRaster::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{FireParticleSplatParticleRaster::AT.particles_state, particles_state}},
                daxa::TaskViewVariant{std::pair{FireParticleSplatParticleRaster::AT.splat_rendered_particle_verts, splat_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{FireParticleSplatParticleRaster::AT.fire_particles, fire_particle_allocator.element_buffer.task_resource}},
                daxa::TaskViewVariant{std::pair{FireParticleSplatParticleRaster::AT.g_buffer_image_id, gbuffer_depth.gbuffer}},
                daxa::TaskViewVariant{std::pair{FireParticleSplatParticleRaster::AT.velocity_image_id, velocity_image}},
                daxa::TaskViewVariant{std::pair{FireParticleSplatParticleRaster::AT.vs_normal_image_id, gbuffer_depth.geometric_normal}},
                daxa::TaskViewVariant{std::pair{FireParticleSplatParticleRaster::AT.depth_image_id, gbuffer_depth.depth.current()}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, FireParticleSplatParticleRasterPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(FireParticleSplatParticleRaster::AT.g_buffer_image_id).ids[0]).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .color_attachments = {
                        {.image_view = ti.get(FireParticleSplatParticleRaster::AT.g_buffer_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                        {.image_view = ti.get(FireParticleSplatParticleRaster::AT.velocity_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                        {.image_view = ti.get(FireParticleSplatParticleRaster::AT.vs_normal_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                    },
                    .depth_attachment = {{.image_view = ti.get(FireParticleSplatParticleRaster::AT.depth_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(FireParticleSplatParticleRaster::AT.particles_state).ids[0],
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, fire_particle) + offsetof(ParticleDrawParams, splat_draw_params),
                    .is_indexed = false,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });
    }
};

#endif
