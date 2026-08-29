#ifndef VOXELS_PARTICLES_GRASS_GRASS_INL
#define VOXELS_PARTICLES_GRASS_GRASS_INL

#include <voxels/particles/common.inl>

#define MAX_GRASS_BLADES (1 << 22)

struct GrassStrand {
    daxa_f32vec3 origin;
    PackedVoxel packed_voxel;
    daxa_u32 flags;
};
DAXA_DECL_BUFFER_PTR(GrassStrand)

DECL_SIMPLE_STATIC_ALLOCATOR(GrassStrandAllocator, GrassStrand, MAX_GRASS_BLADES, daxa_u32)

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(GrassStrandSimCompute)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(INDIRECT_COMMAND_READ | READ_WRITE, daxa_RWBufferPtr(GrassStrandAllocator), GrassStrandAllocator_allocator_buffer)
DAXA_TH_BUFFER(READ_WRITE, GrassStrandAllocator_heap)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(PackedParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(PackedParticleVertex), shadow_cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(PackedParticleVertex), splat_rendered_particle_verts)
DAXA_TH_IMAGE(SAMPLE, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_DECL_TASK_HEAD_END
struct GrassStrandSimComputePush {
    DAXA_TH_BLOB(GrassStrandSimCompute, uses)
};

DAXA_DECL_RASTER_TASK_HEAD_BEGIN(GrassStrandCubeParticleRaster)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(INDIRECT_COMMAND_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(PackedParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GrassStrand), grass_strands)
DAXA_TH_BUFFER(INDEX_INPUT_READ, indices)
DAXA_TH_IMAGE(VS::SAMPLE, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct GrassStrandCubeParticleRasterPush {
    DAXA_TH_BLOB(GrassStrandCubeParticleRaster, uses)
};

DAXA_DECL_RASTER_TASK_HEAD_BEGIN(GrassStrandCubeParticleShadowRaster)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(INDIRECT_COMMAND_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(PackedParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GrassStrand), grass_strands)
DAXA_TH_BUFFER(INDEX_INPUT_READ, indices)
DAXA_TH_IMAGE(VS::SAMPLE, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_TH_IMAGE_INDEX(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct GrassStrandCubeParticleShadowRasterPush {
    DAXA_TH_BLOB(GrassStrandCubeParticleShadowRaster, uses)
};

DAXA_DECL_RASTER_TASK_HEAD_BEGIN(GrassStrandSplatParticleRaster)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(INDIRECT_COMMAND_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(PackedParticleVertex), splat_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GrassStrand), grass_strands)
DAXA_TH_IMAGE(VS::SAMPLE, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct GrassStrandSplatParticleRasterPush {
    DAXA_TH_BLOB(GrassStrandSplatParticleRaster, uses)
};

#if defined(__cplusplus)
#include "renderer/kajiya/gbuffer.hpp"

struct GrassStrands {
    TemporalBuffer cube_rendered_particle_verts;
    TemporalBuffer shadow_cube_rendered_particle_verts;
    TemporalBuffer splat_rendered_particle_verts;
    StaticAllocatorBufferState<GrassStrandAllocator> grass_allocator;

    void init(GpuContext &gpu_context) {
        grass_allocator.init(gpu_context);
    }

    void simulate(GpuContext &gpu_context, daxa::TaskBufferView particles_state) {
        cube_rendered_particle_verts = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(PackedParticleVertex) * std::max<daxa_u32>(MAX_GRASS_BLADES * 3, 1),
            .name = "grass.cube_rendered_particle_verts",
        });
        shadow_cube_rendered_particle_verts = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(PackedParticleVertex) * std::max<daxa_u32>(MAX_GRASS_BLADES * 3, 1),
            .name = "grass.shadow_cube_rendered_particle_verts",
        });
        splat_rendered_particle_verts = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(PackedParticleVertex) * std::max<daxa_u32>(MAX_GRASS_BLADES * 3, 1),
            .name = "grass.splat_rendered_particle_verts",
        });

        gpu_context.frame_task_graph.register_buffer(cube_rendered_particle_verts.task_resource);
        gpu_context.frame_task_graph.register_buffer(shadow_cube_rendered_particle_verts.task_resource);
        gpu_context.frame_task_graph.register_buffer(splat_rendered_particle_verts.task_resource);

        gpu_context.add(ComputeTask<GrassStrandSimCompute::Info, GrassStrandSimComputePush, NoTaskInfo>{
            .source = "voxels/particles/grass/sim.comp.glsl",
            .extra_defines = {ShaderDefine{.name = "GRASS", .value = "1"}},
            .views = GrassStrandSimCompute::Views{
                .gpu_input = gpu_context.task_input_buffer.view(),
                .particles_state = particles_state,
                .GrassStrandAllocator_allocator_buffer = grass_allocator.allocator_buffer.task_resource.view(),
                .GrassStrandAllocator_heap = grass_allocator.element_buffer.task_resource.view(),
                .cube_rendered_particle_verts = cube_rendered_particle_verts.task_resource.view(),
                .shadow_cube_rendered_particle_verts = shadow_cube_rendered_particle_verts.task_resource.view(),
                .splat_rendered_particle_verts = splat_rendered_particle_verts.task_resource.view(),
                .value_noise_texture = gpu_context.task_value_noise_image_view,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, GrassStrandSimComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch_indirect({
                    .indirect_buffer = ti.get(GrassStrandSimCompute::AT.GrassStrandAllocator_allocator_buffer).id,
                    .offset = offsetof(GrassStrandAllocator, element_count_dispatch.x),
                });
            },
        });
    }

    void render_cubes(GpuContext &gpu_context, GbufferDepth &gbuffer_depth, daxa::TaskImageView velocity_image, daxa::TaskImageView shadow_depth, daxa::TaskBufferView particles_state, daxa::TaskBufferView cube_index_buffer) {
        gpu_context.add(RasterTask<GrassStrandCubeParticleRaster::Info, GrassStrandCubeParticleRasterPush, NoTaskInfo>{
            .vert_source = "voxels/particles/cube.raster.glsl",
            .frag_source = "voxels/particles/cube.raster.glsl",
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
            .extra_defines = {ShaderDefine{.name = "GRASS", .value = "1"}},
            .views = GrassStrandCubeParticleRaster::Views{
                .gpu_input = gpu_context.task_input_buffer.view(),
                .particles_state = particles_state,
                .cube_rendered_particle_verts = cube_rendered_particle_verts.task_resource.view(),
                .grass_strands = grass_allocator.element_buffer.task_resource.view(),
                .indices = cube_index_buffer,
                .value_noise_texture = gpu_context.task_value_noise_image_view,
                .g_buffer_image_id = gbuffer_depth.gbuffer,
                .velocity_image_id = velocity_image,
                .vs_normal_image_id = gbuffer_depth.geometric_normal,
                .depth_image_id = gbuffer_depth.depth.current().view(),
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, GrassStrandCubeParticleRasterPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.image_info(ti.get(GrassStrandCubeParticleRaster::AT.g_buffer_image_id).id).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .color_attachments = {
                        {.image_view = ti.get(GrassStrandCubeParticleRaster::AT.g_buffer_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                        {.image_view = ti.get(GrassStrandCubeParticleRaster::AT.velocity_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                        {.image_view = ti.get(GrassStrandCubeParticleRaster::AT.vs_normal_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                    },
                    .depth_attachment = {{.image_view = ti.get(GrassStrandCubeParticleRaster::AT.depth_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.set_index_buffer({
                    .buffer = ti.get(GrassStrandCubeParticleRaster::AT.indices).id,
                    .index_type = daxa::IndexType::uint16,
                });
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(GrassStrandCubeParticleRaster::AT.particles_state).id,
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, grass) + offsetof(ParticleDrawParams, cube_draw_params),
                    .is_indexed = true,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });

        gpu_context.add(RasterTask<GrassStrandCubeParticleShadowRaster::Info, GrassStrandCubeParticleShadowRasterPush, NoTaskInfo>{
            .vert_source = "voxels/particles/cube.raster.glsl",
            .frag_source = "voxels/particles/cube.raster.glsl",
            .depth_test = daxa::DepthTestInfo{
                .depth_attachment_format = daxa::Format::D32_SFLOAT,
                .enable_depth_write = true,
                .depth_test_compare_op = daxa::CompareOp::GREATER,
            },
            .raster = {
                .primitive_topology = daxa::PrimitiveTopology::TRIANGLE_FAN,
                .face_culling = daxa::FaceCullFlagBits::NONE,
            },
            .extra_defines = {ShaderDefine{.name = "GRASS", .value = "1"}, ShaderDefine{.name = "SHADOW_MAP", .value = "1"}},
            .views = GrassStrandCubeParticleShadowRaster::Views{
                .gpu_input = gpu_context.task_input_buffer.view(),
                .particles_state = particles_state,
                .cube_rendered_particle_verts = shadow_cube_rendered_particle_verts.task_resource.view(),
                .grass_strands = grass_allocator.element_buffer.task_resource.view(),
                .indices = cube_index_buffer,
                .value_noise_texture = gpu_context.task_value_noise_image_view,
                .depth_image_id = shadow_depth,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, GrassStrandCubeParticleShadowRasterPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.image_info(ti.get(GrassStrandCubeParticleShadowRaster::AT.depth_image_id).id).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .depth_attachment = {{.image_view = ti.get(GrassStrandCubeParticleShadowRaster::AT.depth_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::CLEAR}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.set_index_buffer({
                    .buffer = ti.get(GrassStrandCubeParticleShadowRaster::AT.indices).id,
                    .index_type = daxa::IndexType::uint16,
                });
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(GrassStrandCubeParticleShadowRaster::AT.particles_state).id,
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, grass) + offsetof(ParticleDrawParams, shadow_cube_draw_params),
                    .is_indexed = true,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });
    }

    void render_splats(GpuContext &gpu_context, GbufferDepth &gbuffer_depth, daxa::TaskImageView velocity_image, daxa::TaskImageView shadow_depth, daxa::TaskBufferView particles_state) {
        gpu_context.add(RasterTask<GrassStrandSplatParticleRaster::Info, GrassStrandSplatParticleRasterPush, NoTaskInfo>{
            .vert_source = "voxels/particles/splat.raster.glsl",
            .frag_source = "voxels/particles/splat.raster.glsl",
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
            .extra_defines = {ShaderDefine{.name = "GRASS", .value = "1"}},
            .views = GrassStrandSplatParticleRaster::Views{
                .gpu_input = gpu_context.task_input_buffer.view(),
                .particles_state = particles_state,
                .splat_rendered_particle_verts = splat_rendered_particle_verts.task_resource.view(),
                .grass_strands = grass_allocator.element_buffer.task_resource.view(),
                .value_noise_texture = gpu_context.task_value_noise_image_view,
                .g_buffer_image_id = gbuffer_depth.gbuffer,
                .velocity_image_id = velocity_image,
                .vs_normal_image_id = gbuffer_depth.geometric_normal,
                .depth_image_id = gbuffer_depth.depth.current().view(),
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, GrassStrandSplatParticleRasterPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.image_info(ti.get(GrassStrandSplatParticleRaster::AT.g_buffer_image_id).id).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .color_attachments = {
                        {.image_view = ti.get(GrassStrandSplatParticleRaster::AT.g_buffer_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                        {.image_view = ti.get(GrassStrandSplatParticleRaster::AT.velocity_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                        {.image_view = ti.get(GrassStrandSplatParticleRaster::AT.vs_normal_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                    },
                    .depth_attachment = {{.image_view = ti.get(GrassStrandSplatParticleRaster::AT.depth_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(GrassStrandSplatParticleRaster::AT.particles_state).id,
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, grass) + offsetof(ParticleDrawParams, splat_draw_params),
                    .is_indexed = false,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });
    }
};

#endif

#endif // VOXELS_PARTICLES_GRASS_GRASS_INL
