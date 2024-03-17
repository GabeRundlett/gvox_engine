#pragma once

#include "../common.inl"

#define MAX_TREE_PARTICLES (1 << 18)

struct TreeParticle {
    daxa_f32vec3 origin;
    PackedVoxel packed_voxel;
    daxa_u32 flags;
};
DAXA_DECL_BUFFER_PTR(TreeParticle)

DECL_SIMPLE_STATIC_ALLOCATOR(TreeParticleAllocator, TreeParticle, MAX_TREE_PARTICLES, daxa_u32)
#define CONSERVATIVE_PARTICLE_PER_TREE_PARTICLE 2

DAXA_DECL_TASK_HEAD_BEGIN(TreeParticleSimCompute, 6 + VOXEL_BUFFER_USE_N + SIMPLE_STATIC_ALLOCATOR_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
SIMPLE_STATIC_ALLOCATOR_USE_BUFFERS(COMPUTE_SHADER_READ_WRITE, TreeParticleAllocator)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(PackedParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(PackedParticleVertex), shadow_cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(PackedParticleVertex), splat_rendered_particle_verts)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_DECL_TASK_HEAD_END
struct TreeParticleSimComputePush {
    DAXA_TH_BLOB(TreeParticleSimCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(TreeParticleCubeParticleRaster, 9)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(DRAW_INDIRECT_INFO_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(PackedParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(TreeParticle), tree_particles)
DAXA_TH_BUFFER(INDEX_READ, indices)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct TreeParticleCubeParticleRasterPush {
    DAXA_TH_BLOB(TreeParticleCubeParticleRaster, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(TreeParticleCubeParticleRasterShadow, 6)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(DRAW_INDIRECT_INFO_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(PackedParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(TreeParticle), tree_particles)
DAXA_TH_BUFFER(INDEX_READ, indices)
DAXA_TH_IMAGE_INDEX(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct TreeParticleCubeParticleRasterShadowPush {
    DAXA_TH_BLOB(TreeParticleCubeParticleRasterShadow, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(TreeParticleSplatParticleRaster, 8)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(DRAW_INDIRECT_INFO_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(PackedParticleVertex), splat_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(TreeParticle), tree_particles)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct TreeParticleSplatParticleRasterPush {
    DAXA_TH_BLOB(TreeParticleSplatParticleRaster, uses)
};

#if defined(__cplusplus)

struct TreeParticles {
    TemporalBuffer cube_rendered_particle_verts;
    TemporalBuffer shadow_cube_rendered_particle_verts;
    TemporalBuffer splat_rendered_particle_verts;
    StaticAllocatorBufferState<TreeParticleAllocator> tree_particle_allocator;

    void init(GpuContext &gpu_context) {
        tree_particle_allocator.init(gpu_context);
    }

    void simulate(GpuContext &gpu_context, VoxelWorldBuffers &voxel_world_buffers, daxa::TaskBufferView particles_state) {
        cube_rendered_particle_verts = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(PackedParticleVertex) * std::max<daxa_u32>(MAX_TREE_PARTICLES * CONSERVATIVE_PARTICLE_PER_TREE_PARTICLE, 1),
            .name = "tree_particle.cube_rendered_particle_verts",
        });
        shadow_cube_rendered_particle_verts = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(PackedParticleVertex) * std::max<daxa_u32>(MAX_TREE_PARTICLES * CONSERVATIVE_PARTICLE_PER_TREE_PARTICLE, 1),
            .name = "tree_particle.shadow_cube_rendered_particle_verts",
        });
        splat_rendered_particle_verts = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(PackedParticleVertex) * std::max<daxa_u32>(MAX_TREE_PARTICLES * CONSERVATIVE_PARTICLE_PER_TREE_PARTICLE, 1),
            .name = "tree_particle.splat_rendered_particle_verts",
        });

        gpu_context.frame_task_graph.use_persistent_buffer(cube_rendered_particle_verts.task_resource);
        gpu_context.frame_task_graph.use_persistent_buffer(shadow_cube_rendered_particle_verts.task_resource);
        gpu_context.frame_task_graph.use_persistent_buffer(splat_rendered_particle_verts.task_resource);

        gpu_context.add(ComputeTask<TreeParticleSimCompute, TreeParticleSimComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"voxels/particles/tree_particle/sim.comp.glsl"},
            .extra_defines = {daxa::ShaderDefine{.name = "TREE_PARTICLE", .value = "1"}},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TreeParticleSimCompute::gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{TreeParticleSimCompute::particles_state, particles_state}},
                VOXELS_BUFFER_USES_ASSIGN(TreeParticleSimCompute, voxel_world_buffers),
                SIMPLE_STATIC_ALLOCATOR_BUFFER_USES_ASSIGN(TreeParticleSimCompute, TreeParticleAllocator, tree_particle_allocator),
                daxa::TaskViewVariant{std::pair{TreeParticleSimCompute::cube_rendered_particle_verts, cube_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{TreeParticleSimCompute::shadow_cube_rendered_particle_verts, shadow_cube_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{TreeParticleSimCompute::splat_rendered_particle_verts, splat_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{TreeParticleSimCompute::value_noise_texture, gpu_context.task_value_noise_image.view().view({.layer_count = 256})}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TreeParticleSimComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(MAX_TREE_PARTICLES + 63) / 64, 1, 1});
            },
        });
    }

    void render_cubes(GpuContext &gpu_context, GbufferDepth &gbuffer_depth, daxa::TaskImageView velocity_image, daxa::TaskImageView shadow_depth, daxa::TaskBufferView particles_state, daxa::TaskBufferView cube_index_buffer) {
        gpu_context.add(RasterTask<TreeParticleCubeParticleRaster, TreeParticleCubeParticleRasterPush, NoTaskInfo>{
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
            .extra_defines = {daxa::ShaderDefine{.name = "TREE_PARTICLE", .value = "1"}},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TreeParticleCubeParticleRaster::gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{TreeParticleCubeParticleRaster::particles_state, particles_state}},
                daxa::TaskViewVariant{std::pair{TreeParticleCubeParticleRaster::cube_rendered_particle_verts, cube_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{TreeParticleCubeParticleRaster::indices, cube_index_buffer}},
                daxa::TaskViewVariant{std::pair{TreeParticleCubeParticleRaster::tree_particles, tree_particle_allocator.element_buffer.task_resource}},
                daxa::TaskViewVariant{std::pair{TreeParticleCubeParticleRaster::g_buffer_image_id, gbuffer_depth.gbuffer}},
                daxa::TaskViewVariant{std::pair{TreeParticleCubeParticleRaster::velocity_image_id, velocity_image}},
                daxa::TaskViewVariant{std::pair{TreeParticleCubeParticleRaster::vs_normal_image_id, gbuffer_depth.geometric_normal}},
                daxa::TaskViewVariant{std::pair{TreeParticleCubeParticleRaster::depth_image_id, gbuffer_depth.depth.current()}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, TreeParticleCubeParticleRasterPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(TreeParticleCubeParticleRaster::g_buffer_image_id).ids[0]).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .color_attachments = {
                        {.image_view = ti.get(TreeParticleCubeParticleRaster::g_buffer_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                        {.image_view = ti.get(TreeParticleCubeParticleRaster::velocity_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                        {.image_view = ti.get(TreeParticleCubeParticleRaster::vs_normal_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                    },
                    .depth_attachment = {{.image_view = ti.get(TreeParticleCubeParticleRaster::depth_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.set_index_buffer({
                    .id = ti.get(TreeParticleCubeParticleRaster::indices).ids[0],
                    .index_type = daxa::IndexType::uint16,
                });
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(TreeParticleCubeParticleRaster::particles_state).ids[0],
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, tree_particle) + offsetof(ParticleDrawParams, cube_draw_params),
                    .is_indexed = true,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });

        gpu_context.add(RasterTask<TreeParticleCubeParticleRasterShadow, TreeParticleCubeParticleRasterShadowPush, NoTaskInfo>{
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
            .extra_defines = {daxa::ShaderDefine{.name = "TREE_PARTICLE", .value = "1"}, daxa::ShaderDefine{.name = "SHADOW_MAP", .value = "1"}},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TreeParticleCubeParticleRasterShadow::gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{TreeParticleCubeParticleRasterShadow::particles_state, particles_state}},
                daxa::TaskViewVariant{std::pair{TreeParticleCubeParticleRasterShadow::cube_rendered_particle_verts, shadow_cube_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{TreeParticleCubeParticleRasterShadow::tree_particles, tree_particle_allocator.element_buffer.task_resource}},
                daxa::TaskViewVariant{std::pair{TreeParticleCubeParticleRasterShadow::indices, cube_index_buffer}},
                daxa::TaskViewVariant{std::pair{TreeParticleCubeParticleRasterShadow::depth_image_id, shadow_depth}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, TreeParticleCubeParticleRasterShadowPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(TreeParticleCubeParticleRasterShadow::depth_image_id).ids[0]).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .depth_attachment = {{.image_view = ti.get(TreeParticleCubeParticleRasterShadow::depth_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.set_index_buffer({
                    .id = ti.get(TreeParticleCubeParticleRasterShadow::indices).ids[0],
                    .index_type = daxa::IndexType::uint16,
                });
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(TreeParticleCubeParticleRasterShadow::particles_state).ids[0],
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, tree_particle) + offsetof(ParticleDrawParams, shadow_cube_draw_params),
                    .is_indexed = true,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });
    }

    void render_splats(GpuContext &gpu_context, GbufferDepth &gbuffer_depth, daxa::TaskImageView velocity_image, daxa::TaskImageView shadow_depth, daxa::TaskBufferView particles_state) {
        gpu_context.add(RasterTask<TreeParticleSplatParticleRaster, TreeParticleSplatParticleRasterPush, NoTaskInfo>{
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
            .extra_defines = {daxa::ShaderDefine{.name = "TREE_PARTICLE", .value = "1"}},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TreeParticleSplatParticleRaster::gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{TreeParticleSplatParticleRaster::particles_state, particles_state}},
                daxa::TaskViewVariant{std::pair{TreeParticleSplatParticleRaster::splat_rendered_particle_verts, splat_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{TreeParticleSplatParticleRaster::tree_particles, tree_particle_allocator.element_buffer.task_resource}},
                daxa::TaskViewVariant{std::pair{TreeParticleSplatParticleRaster::g_buffer_image_id, gbuffer_depth.gbuffer}},
                daxa::TaskViewVariant{std::pair{TreeParticleSplatParticleRaster::velocity_image_id, velocity_image}},
                daxa::TaskViewVariant{std::pair{TreeParticleSplatParticleRaster::vs_normal_image_id, gbuffer_depth.geometric_normal}},
                daxa::TaskViewVariant{std::pair{TreeParticleSplatParticleRaster::depth_image_id, gbuffer_depth.depth.current()}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, TreeParticleSplatParticleRasterPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(TreeParticleSplatParticleRaster::g_buffer_image_id).ids[0]).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .color_attachments = {
                        {.image_view = ti.get(TreeParticleSplatParticleRaster::g_buffer_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                        {.image_view = ti.get(TreeParticleSplatParticleRaster::velocity_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                        {.image_view = ti.get(TreeParticleSplatParticleRaster::vs_normal_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                    },
                    .depth_attachment = {{.image_view = ti.get(TreeParticleSplatParticleRaster::depth_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(TreeParticleSplatParticleRaster::particles_state).ids[0],
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, tree_particle) + offsetof(ParticleDrawParams, splat_draw_params),
                    .is_indexed = false,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });
    }
};

#endif
