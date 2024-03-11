#pragma once

#include <core.inl>
#include <application/input.inl>

#include <voxels/brushes.inl>
#include <utilities/allocator.inl>

struct IndirectDrawParams {
    daxa_u32 vertex_count;
    daxa_u32 instance_count;
    daxa_u32 first_vertex;
    daxa_u32 first_instance;
};

struct IndirectDrawIndexedParams {
    daxa_u32 index_count;
    daxa_u32 instance_count;
    daxa_u32 first_index;
    daxa_u32 vertex_offset;
    daxa_u32 first_instance;
};

struct GrassStrand {
    daxa_f32vec3 origin;
    PackedVoxel packed_voxel;
    daxa_u32 flags;
};
DAXA_DECL_BUFFER_PTR(GrassStrand)

DECL_SIMPLE_STATIC_ALLOCATOR(GrassStrandAllocator, GrassStrand, MAX_GRASS_BLADES, daxa_u32)

struct VoxelParticlesState {
    daxa_u32vec3 simulation_dispatch;
    daxa_u32 place_count;
    daxa_u32vec3 place_bounds_min;
    daxa_u32vec3 place_bounds_max;
    IndirectDrawIndexedParams cube_draw_params;
    IndirectDrawParams splat_draw_params;
};
DAXA_DECL_BUFFER_PTR(VoxelParticlesState)

struct SimulatedVoxelParticle {
    daxa_f32vec3 pos;
    daxa_f32 duration_alive;
    daxa_f32vec3 vel;
    PackedVoxel packed_voxel;
    daxa_u32 flags;
};
DAXA_DECL_BUFFER_PTR(SimulatedVoxelParticle)

struct ParticleVertex {
    daxa_f32vec3 pos;
    daxa_u32 id;
};
DAXA_DECL_BUFFER_PTR(ParticleVertex)

DAXA_DECL_TASK_HEAD_BEGIN(VoxelParticlePerframeCompute, 4 + VOXEL_BUFFER_USE_N + SIMPLE_STATIC_ALLOCATOR_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(GpuOutput), gpu_output)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(SimulatedVoxelParticle), simulated_voxel_particles)
VOXELS_USE_BUFFERS(daxa_RWBufferPtr, COMPUTE_SHADER_READ_WRITE)
SIMPLE_STATIC_ALLOCATOR_USE_BUFFERS(COMPUTE_SHADER_READ_WRITE, GrassStrandAllocator)
DAXA_DECL_TASK_HEAD_END
struct VoxelParticlePerframeComputePush {
    DAXA_TH_BLOB(VoxelParticlePerframeCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(VoxelParticleSimCompute, 6 + VOXEL_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(SimulatedVoxelParticle), simulated_voxel_particles)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(ParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(ParticleVertex), splat_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), placed_voxel_particles)
DAXA_DECL_TASK_HEAD_END
struct VoxelParticleSimComputePush {
    DAXA_TH_BLOB(VoxelParticleSimCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(GrassStrandSimCompute, 6 + VOXEL_BUFFER_USE_N + SIMPLE_STATIC_ALLOCATOR_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
SIMPLE_STATIC_ALLOCATOR_USE_BUFFERS(COMPUTE_SHADER_READ_WRITE, GrassStrandAllocator)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(ParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(ParticleVertex), splat_rendered_particle_verts)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), placed_voxel_particles)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_DECL_TASK_HEAD_END
struct GrassStrandSimComputePush {
    DAXA_TH_BLOB(GrassStrandSimCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(CubeParticleRaster, 6)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_BufferPtr(ParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER(INDEX_READ, indices)
DAXA_TH_IMAGE_INDEX(COLOR_ATTACHMENT, REGULAR_2D, render_image)
DAXA_TH_IMAGE_INDEX(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct CubeParticleRasterPush {
    DAXA_TH_BLOB(CubeParticleRaster, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(CubeParticleRasterShadow, 5)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_BufferPtr(ParticleVertex), cube_rendered_particle_verts)
DAXA_TH_BUFFER(INDEX_READ, indices)
DAXA_TH_IMAGE_INDEX(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct CubeParticleRasterShadowPush {
    DAXA_TH_BLOB(CubeParticleRasterShadow, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(SplatParticleRaster, 5)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_BufferPtr(ParticleVertex), splat_rendered_particle_verts)
DAXA_TH_IMAGE_INDEX(COLOR_ATTACHMENT, REGULAR_2D, render_image)
DAXA_TH_IMAGE_INDEX(DEPTH_ATTACHMENT, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct SplatParticleRasterPush {
    DAXA_TH_BLOB(SplatParticleRaster, uses)
};

#if defined(__cplusplus)

struct VoxelParticles {
    TemporalBuffer global_state;
    TemporalBuffer simulated_voxel_particles;
    TemporalBuffer cube_rendered_particle_verts;
    TemporalBuffer splat_rendered_particle_verts;
    TemporalBuffer placed_voxel_particles;
    TemporalBuffer cube_index_buffer;
    TemporalBuffer splat_index_buffer;

    StaticAllocatorBufferState<GrassStrandAllocator> grass_allocator;

    void record_startup(GpuContext &gpu_context) {
        global_state = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(VoxelParticlesState),
            .name = "globals_buffer",
        });

        gpu_context.startup_task_graph.use_persistent_buffer(global_state.task_resource);

        gpu_context.startup_task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, global_state.task_resource),
            },
            .task = [this](daxa::TaskInterface const &ti) {
                ti.recorder.clear_buffer({
                    .buffer = global_state.task_resource.get_state().buffers[0],
                    .offset = 0,
                    .size = sizeof(VoxelParticlesState),
                    .clear_value = 0,
                });
            },
            .name = "Clear",
        });

        static constexpr auto cube_indices = std::array<uint16_t, 8>{0, 1, 2, 3, 4, 5, 6, 1};
        cube_index_buffer = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(cube_indices),
            .name = "particles.cube_index_buffer",
        });

        gpu_context.startup_task_graph.use_persistent_buffer(cube_index_buffer.task_resource);

        gpu_context.startup_task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, cube_index_buffer.task_resource),
            },
            .task = [this](daxa::TaskInterface const &ti) {
                auto staging_buffer = ti.device.create_buffer({
                    .size = sizeof(cube_indices),
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "cube_staging_buffer",
                });
                ti.recorder.destroy_buffer_deferred(staging_buffer);
                auto *buffer_ptr = ti.device.get_host_address_as<std::remove_cv_t<decltype(cube_indices)>>(staging_buffer).value();
                *buffer_ptr = cube_indices;
                ti.recorder.copy_buffer_to_buffer({
                    .src_buffer = staging_buffer,
                    .dst_buffer = cube_index_buffer.task_resource.get_state().buffers[0],
                    .size = sizeof(cube_indices),
                });
            },
            .name = "Particle Index Upload",
        });

        grass_allocator.init(gpu_context);
    }

    void simulate(GpuContext &gpu_context, VoxelWorldBuffers &voxel_world_buffers) {
        simulated_voxel_particles = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(SimulatedVoxelParticle) * std::max<daxa_u32>(MAX_SIMULATED_VOXEL_PARTICLES, 1),
            .name = "simulated_voxel_particles",
        });
        cube_rendered_particle_verts = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(ParticleVertex) * std::max<daxa_u32>(MAX_RENDERED_VOXEL_PARTICLES, 1),
            .name = "cube_rendered_particle_verts",
        });
        splat_rendered_particle_verts = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(ParticleVertex) * std::max<daxa_u32>(MAX_RENDERED_VOXEL_PARTICLES, 1),
            .name = "splat_rendered_particle_verts",
        });
        placed_voxel_particles = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(daxa_u32) * std::max<daxa_u32>(MAX_SIMULATED_VOXEL_PARTICLES, 1),
            .name = "placed_voxel_particles",
        });

        gpu_context.frame_task_graph.use_persistent_buffer(global_state.task_resource);
        gpu_context.frame_task_graph.use_persistent_buffer(simulated_voxel_particles.task_resource);
        gpu_context.frame_task_graph.use_persistent_buffer(cube_rendered_particle_verts.task_resource);
        gpu_context.frame_task_graph.use_persistent_buffer(splat_rendered_particle_verts.task_resource);
        gpu_context.frame_task_graph.use_persistent_buffer(placed_voxel_particles.task_resource);
        gpu_context.frame_task_graph.use_persistent_buffer(cube_index_buffer.task_resource);

        if constexpr (MAX_RENDERED_VOXEL_PARTICLES == 0) {
            return;
        }

        gpu_context.add(ComputeTask<VoxelParticlePerframeCompute, VoxelParticlePerframeComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"voxels/particles/perframe.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{VoxelParticlePerframeCompute::gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{VoxelParticlePerframeCompute::gpu_output, gpu_context.task_output_buffer}},
                daxa::TaskViewVariant{std::pair{VoxelParticlePerframeCompute::particles_state, global_state.task_resource}},
                daxa::TaskViewVariant{std::pair{VoxelParticlePerframeCompute::simulated_voxel_particles, simulated_voxel_particles.task_resource}},
                VOXELS_BUFFER_USES_ASSIGN(VoxelParticlePerframeCompute, voxel_world_buffers),
                SIMPLE_STATIC_ALLOCATOR_BUFFER_USES_ASSIGN(VoxelParticlePerframeCompute, GrassStrandAllocator, grass_allocator),
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, VoxelParticlePerframeComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({1, 1, 1});
            },
        });
        gpu_context.add(ComputeTask<VoxelParticleSimCompute, VoxelParticleSimComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"voxels/particles/sim.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{VoxelParticleSimCompute::gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{VoxelParticleSimCompute::particles_state, global_state.task_resource}},
                VOXELS_BUFFER_USES_ASSIGN(VoxelParticleSimCompute, voxel_world_buffers),
                daxa::TaskViewVariant{std::pair{VoxelParticleSimCompute::simulated_voxel_particles, simulated_voxel_particles.task_resource}},
                daxa::TaskViewVariant{std::pair{VoxelParticleSimCompute::cube_rendered_particle_verts, cube_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{VoxelParticleSimCompute::splat_rendered_particle_verts, splat_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{VoxelParticleSimCompute::placed_voxel_particles, placed_voxel_particles.task_resource}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, VoxelParticleSimComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch_indirect({
                    .indirect_buffer = ti.get(VoxelParticleSimCompute::particles_state).ids[0],
                    .offset = offsetof(VoxelParticlesState, simulation_dispatch),
                });
            },
        });

        gpu_context.add(ComputeTask<GrassStrandSimCompute, GrassStrandSimComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"voxels/particles/grass_sim.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{GrassStrandSimCompute::gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{GrassStrandSimCompute::particles_state, global_state.task_resource}},
                VOXELS_BUFFER_USES_ASSIGN(GrassStrandSimCompute, voxel_world_buffers),
                SIMPLE_STATIC_ALLOCATOR_BUFFER_USES_ASSIGN(GrassStrandSimCompute, GrassStrandAllocator, grass_allocator),
                daxa::TaskViewVariant{std::pair{GrassStrandSimCompute::cube_rendered_particle_verts, cube_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{GrassStrandSimCompute::splat_rendered_particle_verts, splat_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{GrassStrandSimCompute::placed_voxel_particles, placed_voxel_particles.task_resource}},
                daxa::TaskViewVariant{std::pair{GrassStrandSimCompute::value_noise_texture, gpu_context.task_value_noise_image.view().view({.layer_count = 256})}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, GrassStrandSimComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(MAX_GRASS_BLADES + 63) / 64, 1, 1});
            },
        });
    }

    auto render(GpuContext &gpu_context) -> std::array<daxa::TaskImageView, 3> {
        auto format = daxa::Format::R32_UINT;
        auto raster_color_image = gpu_context.frame_task_graph.create_transient_image({
            .format = format,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "raster_color_image",
        });
        auto raster_depth_image = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::D32_SFLOAT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "raster_depth_image",
        });

        gpu_context.add(RasterTask<CubeParticleRaster, CubeParticleRasterPush, NoTaskInfo>{
            .vert_source = daxa::ShaderFile{"voxels/particles/cube.raster.glsl"},
            .frag_source = daxa::ShaderFile{"voxels/particles/cube.raster.glsl"},
            .color_attachments = {{
                .format = format,
            }},
            .depth_test = daxa::DepthTestInfo{
                .depth_attachment_format = daxa::Format::D32_SFLOAT,
                .enable_depth_write = true,
                .depth_test_compare_op = daxa::CompareOp::GREATER,
            },
            .raster = {
                .primitive_topology = daxa::PrimitiveTopology::TRIANGLE_FAN,
                .face_culling = daxa::FaceCullFlagBits::NONE,
            },
            .views = std::array{
                daxa::TaskViewVariant{std::pair{CubeParticleRaster::gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{CubeParticleRaster::particles_state, global_state.task_resource}},
                daxa::TaskViewVariant{std::pair{CubeParticleRaster::cube_rendered_particle_verts, cube_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{CubeParticleRaster::indices, cube_index_buffer.task_resource}},
                daxa::TaskViewVariant{std::pair{CubeParticleRaster::render_image, raster_color_image}},
                daxa::TaskViewVariant{std::pair{CubeParticleRaster::depth_image_id, raster_depth_image}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, CubeParticleRasterPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(CubeParticleRaster::render_image).ids[0]).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .color_attachments = {{.image_view = ti.get(CubeParticleRaster::render_image).ids[0].default_view(), .load_op = daxa::AttachmentLoadOp::CLEAR, .clear_value = std::array<daxa_f32, 4>{0.0f, 0.0f, 0.0f, 0.0f}}},
                    .depth_attachment = {{.image_view = ti.get(CubeParticleRaster::depth_image_id).ids[0].default_view(), .load_op = daxa::AttachmentLoadOp::CLEAR, .clear_value = daxa::DepthValue{0.0f, 0}}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.set_index_buffer({
                    .id = ti.get(CubeParticleRaster::indices).ids[0],
                    .index_type = daxa::IndexType::uint16,
                });
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(CubeParticleRaster::particles_state).ids[0],
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, cube_draw_params),
                    .is_indexed = true,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });

        auto raster_shadow_depth_image = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::D32_SFLOAT,
            .size = {2048, 2048, 1},
            .name = "raster_shadow_depth_image",
        });

        gpu_context.add(RasterTask<CubeParticleRasterShadow, CubeParticleRasterShadowPush, NoTaskInfo>{
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
            .extra_defines = {daxa::ShaderDefine{.name = "SHADOW_MAP", .value = "1"}},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{CubeParticleRasterShadow::gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{CubeParticleRasterShadow::particles_state, global_state.task_resource}},
                daxa::TaskViewVariant{std::pair{CubeParticleRasterShadow::cube_rendered_particle_verts, cube_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{CubeParticleRasterShadow::indices, cube_index_buffer.task_resource}},
                daxa::TaskViewVariant{std::pair{CubeParticleRasterShadow::depth_image_id, raster_shadow_depth_image}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, CubeParticleRasterShadowPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(CubeParticleRasterShadow::depth_image_id).ids[0]).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .depth_attachment = {{.image_view = ti.get(CubeParticleRasterShadow::depth_image_id).ids[0].default_view(), .load_op = daxa::AttachmentLoadOp::CLEAR, .clear_value = daxa::DepthValue{0.0f, 0}}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.set_index_buffer({
                    .id = ti.get(CubeParticleRasterShadow::indices).ids[0],
                    .index_type = daxa::IndexType::uint16,
                });
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(CubeParticleRasterShadow::particles_state).ids[0],
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, cube_draw_params),
                    .is_indexed = true,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });
        debug_utils::DebugDisplay::add_pass({.name = "voxel particle shadow depth", .task_image_id = raster_shadow_depth_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        gpu_context.add(RasterTask<SplatParticleRaster, SplatParticleRasterPush, NoTaskInfo>{
            .vert_source = daxa::ShaderFile{"voxels/particles/splat.raster.glsl"},
            .frag_source = daxa::ShaderFile{"voxels/particles/splat.raster.glsl"},
            .color_attachments = {{
                .format = format,
            }},
            .depth_test = daxa::DepthTestInfo{
                .depth_attachment_format = daxa::Format::D32_SFLOAT,
                .enable_depth_write = true,
                .depth_test_compare_op = daxa::CompareOp::GREATER,
            },
            .raster = {
                .primitive_topology = daxa::PrimitiveTopology::POINT_LIST,
                .face_culling = daxa::FaceCullFlagBits::NONE,
            },
            .views = std::array{
                daxa::TaskViewVariant{std::pair{SplatParticleRaster::gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{SplatParticleRaster::particles_state, global_state.task_resource}},
                daxa::TaskViewVariant{std::pair{SplatParticleRaster::splat_rendered_particle_verts, splat_rendered_particle_verts.task_resource}},
                daxa::TaskViewVariant{std::pair{SplatParticleRaster::render_image, raster_color_image}},
                daxa::TaskViewVariant{std::pair{SplatParticleRaster::depth_image_id, raster_depth_image}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, SplatParticleRasterPush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(SplatParticleRaster::render_image).ids[0]).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .color_attachments = {{.image_view = ti.get(SplatParticleRaster::render_image).ids[0].default_view(), .load_op = daxa::AttachmentLoadOp::LOAD}},
                    .depth_attachment = {{.image_view = ti.get(SplatParticleRaster::depth_image_id).ids[0].default_view(), .load_op = daxa::AttachmentLoadOp::LOAD}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.draw_indirect({
                    .draw_command_buffer = ti.get(SplatParticleRaster::particles_state).ids[0],
                    .indirect_buffer_offset = offsetof(VoxelParticlesState, splat_draw_params),
                    .is_indexed = false,
                });
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "voxel particles", .task_image_id = raster_color_image, .type = DEBUG_IMAGE_TYPE_DEFAULT_UINT});
        debug_utils::DebugDisplay::add_pass({.name = "voxel particles depth", .task_image_id = raster_depth_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        return {raster_color_image, raster_depth_image, raster_shadow_depth_image};
    }
};

#endif
