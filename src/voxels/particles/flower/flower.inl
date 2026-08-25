#pragma once

#include <voxels/particles/common.inl>

#define FLOWER_TYPE_NONE 0
#define FLOWER_TYPE_DANDELION 1
#define FLOWER_TYPE_DANDELION_WHITE 2
#define FLOWER_TYPE_TULIP 3
#define FLOWER_TYPE_LAVENDER 4

#define MAX_FLOWERS (1 << 17)

struct Flower {
    daxa_f32vec3 origin;
    PackedVoxel packed_voxel;
    daxa_u32 type;
    daxa_u32 flags;
};
DAXA_DECL_BUFFER_PTR(Flower)

DECL_SIMPLE_STATIC_ALLOCATOR(FlowerAllocator, Flower, MAX_FLOWERS, daxa_u32)
#define CONSERVATIVE_PARTICLE_PER_FLOWER (6 + 18 + 3)

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(FlowerSimCompute)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(INDIRECT_COMMAND_READ | READ_WRITE, daxa_RWBufferPtr(FlowerAllocator), FlowerAllocator_allocator_buffer)
DAXA_TH_BUFFER(READ_WRITE, FlowerAllocator_heap)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(PackedParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(PackedParticleVertex), shadow_cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(PackedParticleVertex), splat_rendered_particle_verts)
DAXA_TH_IMAGE(SAMPLE, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_DECL_TASK_HEAD_END
struct FlowerSimComputePush {
    DAXA_TH_BLOB(FlowerSimCompute, uses)
};

DAXA_DECL_RASTER_TASK_HEAD_BEGIN(FlowerCubeParticleRaster)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(INDIRECT_COMMAND_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(PackedParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(Flower), flowers)
DAXA_TH_BUFFER(INDEX_INPUT_READ, indices)
DAXA_TH_IMAGE(VS::SAMPLE, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct FlowerCubeParticleRasterPush {
    DAXA_TH_BLOB(FlowerCubeParticleRaster, uses)
};

DAXA_DECL_RASTER_TASK_HEAD_BEGIN(FlowerCubeParticleShadowRaster)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(INDIRECT_COMMAND_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(PackedParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(Flower), flowers)
DAXA_TH_BUFFER(INDEX_INPUT_READ, indices)
DAXA_TH_IMAGE(VS::SAMPLE, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_TH_IMAGE_INDEX(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct FlowerCubeParticleShadowRasterPush {
    DAXA_TH_BLOB(FlowerCubeParticleShadowRaster, uses)
};

DAXA_DECL_RASTER_TASK_HEAD_BEGIN(FlowerSplatParticleRaster)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(INDIRECT_COMMAND_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(PackedParticleVertex), splat_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(Flower), flowers)
DAXA_TH_IMAGE(VS::SAMPLE, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct FlowerSplatParticleRasterPush {
    DAXA_TH_BLOB(FlowerSplatParticleRaster, uses)
};

#if defined(__cplusplus)
#include "renderer/kajiya/gbuffer.hpp"

struct Flowers {
    TemporalBuffer cube_rendered_particle_verts;
    TemporalBuffer shadow_cube_rendered_particle_verts;
    TemporalBuffer splat_rendered_particle_verts;
    StaticAllocatorBufferState<FlowerAllocator> flower_allocator;

    void init(GpuContext &gpu_context) {
        flower_allocator.init(gpu_context);
    }

    void simulate(GpuContext &gpu_context, daxa::TaskBufferView particles_state) {
        cube_rendered_particle_verts = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(PackedParticleVertex) * std::max<daxa_u32>(MAX_FLOWERS * CONSERVATIVE_PARTICLE_PER_FLOWER, 1),
            .name = "flower.cube_rendered_particle_verts",
        });
        shadow_cube_rendered_particle_verts = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(PackedParticleVertex) * std::max<daxa_u32>(MAX_FLOWERS * CONSERVATIVE_PARTICLE_PER_FLOWER, 1),
            .name = "flower.shadow_cube_rendered_particle_verts",
        });
        splat_rendered_particle_verts = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(PackedParticleVertex) * std::max<daxa_u32>(MAX_FLOWERS * CONSERVATIVE_PARTICLE_PER_FLOWER, 1),
            .name = "flower.splat_rendered_particle_verts",
        });

        gpu_context.frame_task_graph.register_buffer(cube_rendered_particle_verts.task_resource);
        gpu_context.frame_task_graph.register_buffer(shadow_cube_rendered_particle_verts.task_resource);
        gpu_context.frame_task_graph.register_buffer(splat_rendered_particle_verts.task_resource);

        gpu_context.add(ComputeTask<FlowerSimCompute::Info, FlowerSimComputePush, NoTaskInfo>{
            .source = "voxels/particles/flower/sim.comp.glsl",
            .extra_defines = {ShaderDefine{.name = "FLOWER", .value = "1"}},
            .views = FlowerSimCompute::Views{
                .gpu_input = gpu_context.task_input_buffer.view(),
                .particles_state = particles_state,
                .FlowerAllocator_allocator_buffer = flower_allocator.allocator_buffer.task_resource.view(),
                .FlowerAllocator_heap = flower_allocator.element_buffer.task_resource.view(),
                .cube_rendered_particle_verts = cube_rendered_particle_verts.task_resource.view(),
                .shadow_cube_rendered_particle_verts = shadow_cube_rendered_particle_verts.task_resource.view(),
                .splat_rendered_particle_verts = splat_rendered_particle_verts.task_resource.view(),
                .value_noise_texture = gpu_context.task_value_noise_image_view,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, FlowerSimComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch_indirect({
                    .indirect_buffer = ti.get(FlowerSimCompute::AT.FlowerAllocator_allocator_buffer).id,
                    .offset = offsetof(FlowerAllocator, element_count_dispatch.x),
                });
            },
        });
    }

    void render_cubes(GpuContext &gpu_context, GbufferDepth &gbuffer_depth, daxa::TaskImageView velocity_image, daxa::TaskImageView shadow_depth, daxa::TaskBufferView particles_state, daxa::TaskBufferView cube_index_buffer) {
        gpu_context.add(RasterTask<FlowerCubeParticleRaster::Info, FlowerCubeParticleRasterPush, NoTaskInfo>{
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
            .extra_defines = {ShaderDefine{.name = "FLOWER", .value = "1"}},
            .views = FlowerCubeParticleRaster::Views{
                .gpu_input = gpu_context.task_input_buffer.view(),
                .particles_state = particles_state,
                .cube_rendered_particle_verts = cube_rendered_particle_verts.task_resource.view(),
                .flowers = flower_allocator.element_buffer.task_resource.view(),
                .indices = cube_index_buffer,
                .value_noise_texture = gpu_context.task_value_noise_image_view,
                .g_buffer_image_id = gbuffer_depth.gbuffer,
                .velocity_image_id = velocity_image,
                .vs_normal_image_id = gbuffer_depth.geometric_normal,
                .depth_image_id = gbuffer_depth.depth.current().view(),
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, FlowerCubeParticleRasterPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.image_info(ti.get(FlowerCubeParticleRaster::AT.g_buffer_image_id).id).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .color_attachments = {
                        {.image_view = ti.get(FlowerCubeParticleRaster::AT.g_buffer_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                        {.image_view = ti.get(FlowerCubeParticleRaster::AT.velocity_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                        {.image_view = ti.get(FlowerCubeParticleRaster::AT.vs_normal_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                    },
                    .depth_attachment = {{.image_view = ti.get(FlowerCubeParticleRaster::AT.depth_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.set_index_buffer({
                    .buffer = ti.get(FlowerCubeParticleRaster::AT.indices).id,
                    .index_type = daxa::IndexType::uint16,
                });
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(FlowerCubeParticleRaster::AT.particles_state).id,
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, flower) + offsetof(ParticleDrawParams, cube_draw_params),
                    .is_indexed = true,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });

        gpu_context.add(RasterTask<FlowerCubeParticleShadowRaster::Info, FlowerCubeParticleShadowRasterPush, NoTaskInfo>{
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
            .extra_defines = {ShaderDefine{.name = "FLOWER", .value = "1"}, ShaderDefine{.name = "SHADOW_MAP", .value = "1"}},
            .views = FlowerCubeParticleShadowRaster::Views{
                .gpu_input = gpu_context.task_input_buffer.view(),
                .particles_state = particles_state,
                .cube_rendered_particle_verts = shadow_cube_rendered_particle_verts.task_resource.view(),
                .flowers = flower_allocator.element_buffer.task_resource.view(),
                .indices = cube_index_buffer,
                .value_noise_texture = gpu_context.task_value_noise_image_view,
                .depth_image_id = shadow_depth,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, FlowerCubeParticleShadowRasterPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.image_info(ti.get(FlowerCubeParticleShadowRaster::AT.depth_image_id).id).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .depth_attachment = {{.image_view = ti.get(FlowerCubeParticleShadowRaster::AT.depth_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.set_index_buffer({
                    .buffer = ti.get(FlowerCubeParticleShadowRaster::AT.indices).id,
                    .index_type = daxa::IndexType::uint16,
                });
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(FlowerCubeParticleShadowRaster::AT.particles_state).id,
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, flower) + offsetof(ParticleDrawParams, shadow_cube_draw_params),
                    .is_indexed = true,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });
    }

    void render_splats(GpuContext &gpu_context, GbufferDepth &gbuffer_depth, daxa::TaskImageView velocity_image, daxa::TaskImageView shadow_depth, daxa::TaskBufferView particles_state) {
        gpu_context.add(RasterTask<FlowerSplatParticleRaster::Info, FlowerSplatParticleRasterPush, NoTaskInfo>{
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
            .extra_defines = {ShaderDefine{.name = "FLOWER", .value = "1"}},
            .views = FlowerSplatParticleRaster::Views{
                .gpu_input = gpu_context.task_input_buffer.view(),
                .particles_state = particles_state,
                .splat_rendered_particle_verts = splat_rendered_particle_verts.task_resource.view(),
                .flowers = flower_allocator.element_buffer.task_resource.view(),
                .value_noise_texture = gpu_context.task_value_noise_image_view,
                .g_buffer_image_id = gbuffer_depth.gbuffer,
                .velocity_image_id = velocity_image,
                .vs_normal_image_id = gbuffer_depth.geometric_normal,
                .depth_image_id = gbuffer_depth.depth.current().view(),
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, FlowerSplatParticleRasterPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.image_info(ti.get(FlowerSplatParticleRaster::AT.g_buffer_image_id).id).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .color_attachments = {
                        {.image_view = ti.get(FlowerSplatParticleRaster::AT.g_buffer_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                        {.image_view = ti.get(FlowerSplatParticleRaster::AT.velocity_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                        {.image_view = ti.get(FlowerSplatParticleRaster::AT.vs_normal_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD},
                    },
                    .depth_attachment = {{.image_view = ti.get(FlowerSplatParticleRaster::AT.depth_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(FlowerSplatParticleRaster::AT.particles_state).id,
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, flower) + offsetof(ParticleDrawParams, splat_draw_params),
                    .is_indexed = false,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });
    }
};

#endif
