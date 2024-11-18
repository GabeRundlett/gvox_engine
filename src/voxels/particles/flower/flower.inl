#pragma once

#include "../common.inl"

#define FLOWER_TYPE_NONE 0
#define FLOWER_TYPE_DANDELION 1
#define FLOWER_TYPE_DANDELION_WHITE 2
#define FLOWER_TYPE_TULIP 3
#define FLOWER_TYPE_LAVENDER 4

#define MAX_FLOWERS (1 << 16)

struct Flower {
    daxa_f32vec3 origin;
    PackedVoxel packed_voxel;
    daxa_u32 type;
};
DAXA_DECL_BUFFER_PTR(Flower)

DECL_SIMPLE_STATIC_ALLOCATOR(FlowerAllocator, Flower, MAX_FLOWERS, daxa_u32)
#define CONSERVATIVE_PARTICLE_PER_FLOWER (6 + 18 + 3)

DAXA_DECL_TASK_HEAD_BEGIN(FlowerSimCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
SIMPLE_STATIC_ALLOCATOR_USE_BUFFERS(COMPUTE_SHADER_READ_WRITE, FlowerAllocator)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(PackedParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(PackedParticleVertex), shadow_cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(PackedParticleVertex), splat_rendered_particle_verts)
DAXA_TH_IMAGE(VERTEX_SHADER_SAMPLED, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_DECL_TASK_HEAD_END
struct FlowerSimComputePush {
    DAXA_TH_BLOB(FlowerSimCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(FlowerCubeParticleRaster)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(DRAW_INDIRECT_INFO_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(PackedParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Flower), flowers)
DAXA_TH_BUFFER(INDEX_READ, indices)
DAXA_TH_IMAGE(VERTEX_SHADER_SAMPLED, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct FlowerCubeParticleRasterPush {
    DAXA_TH_BLOB(FlowerCubeParticleRaster, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(FlowerCubeParticleShadowRaster)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(DRAW_INDIRECT_INFO_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(PackedParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Flower), flowers)
DAXA_TH_BUFFER(INDEX_READ, indices)
DAXA_TH_IMAGE(VERTEX_SHADER_SAMPLED, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_TH_IMAGE_INDEX(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct FlowerCubeParticleShadowRasterPush {
    DAXA_TH_BLOB(FlowerCubeParticleShadowRaster, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(FlowerSplatParticleRaster)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(DRAW_INDIRECT_INFO_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(PackedParticleVertex), splat_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(Flower), flowers)
DAXA_TH_IMAGE(VERTEX_SHADER_SAMPLED, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct FlowerSplatParticleRasterPush {
    DAXA_TH_BLOB(FlowerSplatParticleRaster, uses)
};

#if defined(__cplusplus)

struct Flowers {
    TemporalBuffer cube_rendered_particle_verts;
    TemporalBuffer shadow_cube_rendered_particle_verts;
    TemporalBuffer splat_rendered_particle_verts;
    StaticAllocatorBufferState<FlowerAllocator> flower_allocator;

    void init(GpuContext &gpu_context) {
        flower_allocator.init(gpu_context);
    }

    void simulate(GpuContext &gpu_context, VoxelWorldBuffers &voxel_world_buffers, daxa::TaskBufferView particles_state) {
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

        gpu_context.frame_task_graph.use_persistent_buffer(cube_rendered_particle_verts.task_resource);
        gpu_context.frame_task_graph.use_persistent_buffer(shadow_cube_rendered_particle_verts.task_resource);
        gpu_context.frame_task_graph.use_persistent_buffer(splat_rendered_particle_verts.task_resource);

        gpu_context.add(ComputeTask<FlowerSimCompute::Task, FlowerSimComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"voxels/particles/flower/sim.comp.glsl"},
            .extra_defines = {daxa::ShaderDefine{.name = "FLOWER", .value = "1"}},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{FlowerSimCompute::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{FlowerSimCompute::AT.particles_state, particles_state}},
                VOXELS_BUFFER_USES_ASSIGN(FlowerSimCompute, voxel_world_buffers),
                SIMPLE_STATIC_ALLOCATOR_BUFFER_USES_ASSIGN(FlowerSimCompute, FlowerAllocator, flower_allocator),
                daxa::TaskViewVariant{std::pair{FlowerSimCompute::AT.cube_rendered_particle_verts, cube_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{FlowerSimCompute::AT.shadow_cube_rendered_particle_verts, shadow_cube_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{FlowerSimCompute::AT.splat_rendered_particle_verts, splat_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{FlowerSimCompute::AT.value_noise_texture, gpu_context.task_value_noise_image_view}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, FlowerSimComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(MAX_FLOWERS + 63) / 64, 1, 1});
            },
        });
    }

    void render_cubes(GpuContext &gpu_context, GbufferDepth &gbuffer_depth, daxa::TaskImageView velocity_image, daxa::TaskImageView shadow_depth, daxa::TaskBufferView particles_state, daxa::TaskBufferView cube_index_buffer) {
        gpu_context.add(RasterTask<FlowerCubeParticleRaster::Task, FlowerCubeParticleRasterPush, NoTaskInfo>{
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
            .extra_defines = {daxa::ShaderDefine{.name = "FLOWER", .value = "1"}},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{FlowerCubeParticleRaster::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{FlowerCubeParticleRaster::AT.particles_state, particles_state}},
                daxa::TaskViewVariant{std::pair{FlowerCubeParticleRaster::AT.cube_rendered_particle_verts, cube_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{FlowerCubeParticleRaster::AT.indices, cube_index_buffer}},
                daxa::TaskViewVariant{std::pair{FlowerCubeParticleRaster::AT.flowers, flower_allocator.element_buffer.task_resource}},
                daxa::TaskViewVariant{std::pair{FlowerCubeParticleRaster::AT.value_noise_texture, gpu_context.task_value_noise_image_view}},
                daxa::TaskViewVariant{std::pair{FlowerCubeParticleRaster::AT.g_buffer_image_id, gbuffer_depth.gbuffer}},
                daxa::TaskViewVariant{std::pair{FlowerCubeParticleRaster::AT.velocity_image_id, velocity_image}},
                daxa::TaskViewVariant{std::pair{FlowerCubeParticleRaster::AT.vs_normal_image_id, gbuffer_depth.geometric_normal}},
                daxa::TaskViewVariant{std::pair{FlowerCubeParticleRaster::AT.depth_image_id, gbuffer_depth.depth.current()}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, FlowerCubeParticleRasterPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(FlowerCubeParticleRaster::AT.g_buffer_image_id).ids[0]).value();
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
                    .id = ti.get(FlowerCubeParticleRaster::AT.indices).ids[0],
                    .index_type = daxa::IndexType::uint16,
                });
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(FlowerCubeParticleRaster::AT.particles_state).ids[0],
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, flower) + offsetof(ParticleDrawParams, cube_draw_params),
                    .is_indexed = true,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });

        gpu_context.add(RasterTask<FlowerCubeParticleShadowRaster::Task, FlowerCubeParticleShadowRasterPush, NoTaskInfo>{
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
            .extra_defines = {daxa::ShaderDefine{.name = "FLOWER", .value = "1"}, daxa::ShaderDefine{.name = "SHADOW_MAP", .value = "1"}},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{FlowerCubeParticleShadowRaster::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{FlowerCubeParticleShadowRaster::AT.particles_state, particles_state}},
                daxa::TaskViewVariant{std::pair{FlowerCubeParticleShadowRaster::AT.cube_rendered_particle_verts, shadow_cube_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{FlowerCubeParticleShadowRaster::AT.flowers, flower_allocator.element_buffer.task_resource}},
                daxa::TaskViewVariant{std::pair{FlowerCubeParticleShadowRaster::AT.indices, cube_index_buffer}},
                daxa::TaskViewVariant{std::pair{FlowerCubeParticleShadowRaster::AT.value_noise_texture, gpu_context.task_value_noise_image_view}},
                daxa::TaskViewVariant{std::pair{FlowerCubeParticleShadowRaster::AT.depth_image_id, shadow_depth}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, FlowerCubeParticleShadowRasterPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(FlowerCubeParticleShadowRaster::AT.depth_image_id).ids[0]).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .depth_attachment = {{.image_view = ti.get(FlowerCubeParticleShadowRaster::AT.depth_image_id).view_ids[0], .load_op = daxa::AttachmentLoadOp::LOAD}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.set_index_buffer({
                    .id = ti.get(FlowerCubeParticleShadowRaster::AT.indices).ids[0],
                    .index_type = daxa::IndexType::uint16,
                });
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(FlowerCubeParticleShadowRaster::AT.particles_state).ids[0],
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, flower) + offsetof(ParticleDrawParams, shadow_cube_draw_params),
                    .is_indexed = true,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });
    }

    void render_splats(GpuContext &gpu_context, GbufferDepth &gbuffer_depth, daxa::TaskImageView velocity_image, daxa::TaskImageView shadow_depth, daxa::TaskBufferView particles_state) {
        gpu_context.add(RasterTask<FlowerSplatParticleRaster::Task, FlowerSplatParticleRasterPush, NoTaskInfo>{
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
            .extra_defines = {daxa::ShaderDefine{.name = "FLOWER", .value = "1"}},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{FlowerSplatParticleRaster::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{FlowerSplatParticleRaster::AT.particles_state, particles_state}},
                daxa::TaskViewVariant{std::pair{FlowerSplatParticleRaster::AT.splat_rendered_particle_verts, splat_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{FlowerSplatParticleRaster::AT.flowers, flower_allocator.element_buffer.task_resource}},
                daxa::TaskViewVariant{std::pair{FlowerSplatParticleRaster::AT.value_noise_texture, gpu_context.task_value_noise_image_view}},
                daxa::TaskViewVariant{std::pair{FlowerSplatParticleRaster::AT.g_buffer_image_id, gbuffer_depth.gbuffer}},
                daxa::TaskViewVariant{std::pair{FlowerSplatParticleRaster::AT.velocity_image_id, velocity_image}},
                daxa::TaskViewVariant{std::pair{FlowerSplatParticleRaster::AT.vs_normal_image_id, gbuffer_depth.geometric_normal}},
                daxa::TaskViewVariant{std::pair{FlowerSplatParticleRaster::AT.depth_image_id, gbuffer_depth.depth.current()}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, FlowerSplatParticleRasterPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(FlowerSplatParticleRaster::AT.g_buffer_image_id).ids[0]).value();
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
                    .draw_command_buffer = ti.get(FlowerSplatParticleRaster::AT.particles_state).ids[0],
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, flower) + offsetof(ParticleDrawParams, splat_draw_params),
                    .is_indexed = false,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });
    }
};

#endif
