#pragma once

#include <shared/core.inl>

#if VOXEL_PARTICLE_SIM_COMPUTE || defined(__cplusplus)
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

#if defined(__cplusplus)

struct VoxelParticleSimComputeTaskState {
    AsyncManagedComputePipeline pipeline;

    VoxelParticleSimComputeTaskState(AsyncPipelineManager &pipeline_manager) {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"voxels/voxel_particle_sim.comp.glsl"},
                .compile_options = {.defines = {{"VOXEL_PARTICLE_SIM_COMPUTE", "1"}}},
            },
            .push_constant_size = sizeof(VoxelParticleSimComputePush),
            .name = "voxel_particle_sim",
        });
    }

    void record_commands(VoxelParticleSimComputePush const &push, daxa::CommandRecorder &recorder, daxa::BufferId globals_buffer_id) {
        if (!pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(pipeline.get());
        recorder.push_constant(push);
        recorder.dispatch_indirect({
            .indirect_buffer = globals_buffer_id,
            .offset = offsetof(GpuGlobals, voxel_particles_state) + offsetof(VoxelParticlesState, simulation_dispatch),
        });
    }
};

struct VoxelParticleSimComputeTask {
    VoxelParticleSimCompute::Uses uses;
    std::string name = "VoxelParticleSimCompute";
    VoxelParticleSimComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto push = VoxelParticleSimComputePush{};
        ti.copy_task_head_to(&push.uses);
        state->record_commands(push, recorder, uses.globals.buffer());
    }
};

#endif
