#pragma once

#include <core.inl>

#include <utilities/allocator.inl>

#include <renderer/core.inl>

#include <voxels/particles/common.inl>
#include <voxels/particles/grass/grass.inl>
#include <voxels/particles/flower/flower.inl>
#include <voxels/particles/sim_particle/sim_particle.inl>
#include <voxels/particles/tree_particle/tree_particle.inl>
#include <voxels/particles/fire_particle/fire_particle.inl>

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(VoxelParticlePerframeCompute)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
SIMPLE_STATIC_ALLOCATOR_USE_BUFFERS(READ_WRITE, GrassStrandAllocator)
SIMPLE_STATIC_ALLOCATOR_USE_BUFFERS(READ_WRITE, FlowerAllocator)
SIMPLE_STATIC_ALLOCATOR_USE_BUFFERS(READ_WRITE, TreeParticleAllocator)
SIMPLE_STATIC_ALLOCATOR_USE_BUFFERS(READ_WRITE, FireParticleAllocator)
DAXA_DECL_TASK_HEAD_END
struct VoxelParticlePerframeComputePush {
    DAXA_TH_BLOB(VoxelParticlePerframeCompute, uses)
};

#if defined(__cplusplus)

#include <application/settings.hpp>

struct VoxelParticles {
    TemporalBuffer global_state;
    TemporalBuffer cube_index_buffer;
    SimParticles sim_particles;
    GrassStrands grass;
    Flowers flowers;
    TreeParticles tree_particles;
    FireParticles fire_particles;

    void record_startup(GpuContext &gpu_context) {
        global_state = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(VoxelParticlesState),
            .name = "globals_buffer",
        });

        gpu_context.startup_task_graph.register_buffer(global_state.task_resource);

        gpu_context.startup_task_graph.add_task(
            daxa::InlineTask::Transfer("Clear")
                .writes(global_state.task_resource)
                .executes([this](daxa::TaskInterface ti) {
                    ti.recorder.clear_buffer({
                        .buffer = global_state.task_resource.id(),
                        .offset = 0,
                        .size = sizeof(VoxelParticlesState),
                        .clear_value = 0,
                    });
                }));

        static constexpr auto cube_indices = std::array<uint16_t, 8>{0, 1, 2, 3, 4, 5, 6, 1};
        cube_index_buffer = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(cube_indices),
            .name = "particles.cube_index_buffer",
        });

        gpu_context.startup_task_graph.register_buffer(cube_index_buffer.task_resource);

        gpu_context.startup_task_graph.add_task(
            daxa::InlineTask::Transfer("Particle Index Upload")
                .writes(cube_index_buffer.task_resource)
                .executes([this](daxa::TaskInterface ti) {
                    auto staging_buffer = ti.device.create_buffer({
                        .size = sizeof(cube_indices),
                        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                        .name = "cube_staging_buffer",
                    });
                    ti.recorder.destroy_buffer_deferred(staging_buffer);
                    auto *buffer_ptr = ti.device.buffer_host_address_as<std::remove_cv_t<decltype(cube_indices)>>(staging_buffer).value();
                    *buffer_ptr = cube_indices;
                    ti.recorder.copy_buffer_to_buffer({
                        .src_buffer = staging_buffer,
                        .dst_buffer = cube_index_buffer.task_resource.id(),
                        .size = sizeof(cube_indices),
                    });
                }));

        grass.init(gpu_context);
        flowers.init(gpu_context);
        tree_particles.init(gpu_context);
        fire_particles.init(gpu_context);
    }

    void simulate(GpuContext &gpu_context, VoxelWorldBuffers &voxel_world_buffers) {
        gpu_context.frame_task_graph.register_buffer(global_state.task_resource);
        gpu_context.frame_task_graph.register_buffer(cube_index_buffer.task_resource);

        gpu_context.add(ComputeTask<VoxelParticlePerframeCompute::Info, VoxelParticlePerframeComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"voxels/particles/perframe.comp.glsl"},
            .views = VoxelParticlePerframeCompute::Views{
                .gpu_input = gpu_context.task_input_buffer.view(),
                .particles_state = global_state.task_resource.view(),
                SIMPLE_STATIC_ALLOCATOR_BUFFER_USES_ASSIGN(VoxelParticlePerframeCompute, GrassStrandAllocator, grass.grass_allocator),
                SIMPLE_STATIC_ALLOCATOR_BUFFER_USES_ASSIGN(VoxelParticlePerframeCompute, FlowerAllocator, flowers.flower_allocator),
                SIMPLE_STATIC_ALLOCATOR_BUFFER_USES_ASSIGN(VoxelParticlePerframeCompute, TreeParticleAllocator, tree_particles.tree_particle_allocator),
                SIMPLE_STATIC_ALLOCATOR_BUFFER_USES_ASSIGN(VoxelParticlePerframeCompute, FireParticleAllocator, fire_particles.fire_particle_allocator),
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, VoxelParticlePerframeComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({1, 1, 1});
            },
        });

        if constexpr (MAX_SIMULATED_VOXEL_PARTICLES != 0) {
            sim_particles.simulate(gpu_context, voxel_world_buffers, global_state.task_resource);
        }

        if constexpr (MAX_GRASS_BLADES != 0) {
            grass.simulate(gpu_context, voxel_world_buffers, global_state.task_resource);
        }

        if constexpr (MAX_FLOWERS != 0) {
            flowers.simulate(gpu_context, voxel_world_buffers, global_state.task_resource);
        }

        if constexpr (MAX_TREE_PARTICLES != 0) {
            tree_particles.simulate(gpu_context, voxel_world_buffers, global_state.task_resource);
        }

        if constexpr (MAX_FIRE_PARTICLES != 0) {
            fire_particles.simulate(gpu_context, voxel_world_buffers, global_state.task_resource);
        }
    }

    auto render(GpuContext &gpu_context, GbufferDepth &gbuffer_depth, daxa::TaskImageView velocity_image) -> daxa::TaskImageView {
        auto raster_shadow_depth_image = gpu_context.frame_task_graph.create_task_image({
            .format = daxa::Format::D32_SFLOAT,
            .size = {2048, 2048, 1},
            .name = "raster_shadow_depth_image",
        });

        AppSettings::add<settings::Checkbox>({"Graphics", "Draw Particles", {.value = true}, {.task_graph_depends = true}});
        auto draw_particles = AppSettings::get<settings::Checkbox>("Graphics", "Draw Particles").value;

        if (draw_particles) {
            sim_particles.render_cubes(gpu_context, gbuffer_depth, velocity_image, raster_shadow_depth_image, global_state.task_resource, cube_index_buffer.task_resource);
            grass.render_cubes(gpu_context, gbuffer_depth, velocity_image, raster_shadow_depth_image, global_state.task_resource, cube_index_buffer.task_resource);
            flowers.render_cubes(gpu_context, gbuffer_depth, velocity_image, raster_shadow_depth_image, global_state.task_resource, cube_index_buffer.task_resource);
            tree_particles.render_cubes(gpu_context, gbuffer_depth, velocity_image, raster_shadow_depth_image, global_state.task_resource, cube_index_buffer.task_resource);
            fire_particles.render_cubes(gpu_context, gbuffer_depth, velocity_image, raster_shadow_depth_image, global_state.task_resource, cube_index_buffer.task_resource);

            sim_particles.render_splats(gpu_context, gbuffer_depth, velocity_image, raster_shadow_depth_image, global_state.task_resource);
            grass.render_splats(gpu_context, gbuffer_depth, velocity_image, raster_shadow_depth_image, global_state.task_resource);
            flowers.render_splats(gpu_context, gbuffer_depth, velocity_image, raster_shadow_depth_image, global_state.task_resource);
            tree_particles.render_splats(gpu_context, gbuffer_depth, velocity_image, raster_shadow_depth_image, global_state.task_resource);
            fire_particles.render_splats(gpu_context, gbuffer_depth, velocity_image, raster_shadow_depth_image, global_state.task_resource);
        }

        return raster_shadow_depth_image;
    }
};

#endif
