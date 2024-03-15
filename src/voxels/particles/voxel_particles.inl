#pragma once

#include <core.inl>

#include <voxels/brushes.inl>
#include <utilities/allocator.inl>

#include <renderer/core.inl>

#include "common.inl"
#include "grass/grass.inl"
#include "sim_particle/sim_particle.inl"

DAXA_DECL_TASK_HEAD_BEGIN(VoxelParticlePerframeCompute, 3 + VOXEL_BUFFER_USE_N + SIMPLE_STATIC_ALLOCATOR_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(GpuOutput), gpu_output)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelParticlesState), particles_state)
VOXELS_USE_BUFFERS(daxa_RWBufferPtr, COMPUTE_SHADER_READ_WRITE)
SIMPLE_STATIC_ALLOCATOR_USE_BUFFERS(COMPUTE_SHADER_READ_WRITE, GrassStrandAllocator)
DAXA_DECL_TASK_HEAD_END
struct VoxelParticlePerframeComputePush {
    DAXA_TH_BLOB(VoxelParticlePerframeCompute, uses)
};

#if defined(__cplusplus)

struct VoxelParticles {
    TemporalBuffer global_state;
    TemporalBuffer cube_index_buffer;
    SimParticles sim_particles;
    GrassStrands grass;

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

        grass.init(gpu_context);
    }

    void simulate(GpuContext &gpu_context, VoxelWorldBuffers &voxel_world_buffers) {
        gpu_context.frame_task_graph.use_persistent_buffer(global_state.task_resource);
        gpu_context.frame_task_graph.use_persistent_buffer(cube_index_buffer.task_resource);

        gpu_context.add(ComputeTask<VoxelParticlePerframeCompute, VoxelParticlePerframeComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"voxels/particles/perframe.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{VoxelParticlePerframeCompute::gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{VoxelParticlePerframeCompute::gpu_output, gpu_context.task_output_buffer}},
                daxa::TaskViewVariant{std::pair{VoxelParticlePerframeCompute::particles_state, global_state.task_resource}},
                VOXELS_BUFFER_USES_ASSIGN(VoxelParticlePerframeCompute, voxel_world_buffers),
                SIMPLE_STATIC_ALLOCATOR_BUFFER_USES_ASSIGN(VoxelParticlePerframeCompute, GrassStrandAllocator, grass.grass_allocator),
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
    }

    auto render(GpuContext &gpu_context, GbufferDepth &gbuffer_depth, daxa::TaskImageView velocity_image) -> daxa::TaskImageView {
        auto raster_shadow_depth_image = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::D32_SFLOAT,
            .size = {2048, 2048, 1},
            .name = "raster_shadow_depth_image",
        });

        sim_particles.render_cubes(gpu_context, gbuffer_depth, velocity_image, raster_shadow_depth_image, global_state.task_resource, cube_index_buffer.task_resource);
        grass.render_cubes(gpu_context, gbuffer_depth, velocity_image, raster_shadow_depth_image, global_state.task_resource, cube_index_buffer.task_resource);

        sim_particles.render_splats(gpu_context, gbuffer_depth, velocity_image, raster_shadow_depth_image, global_state.task_resource);
        grass.render_splats(gpu_context, gbuffer_depth, velocity_image, raster_shadow_depth_image, global_state.task_resource);

        return raster_shadow_depth_image;
    }
};

#endif
