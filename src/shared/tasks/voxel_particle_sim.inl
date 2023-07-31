#pragma once

#include <shared/core.inl>

DAXA_DECL_TASK_USES_BEGIN(VoxelParticleSimComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_RWBufferPtr(GpuGlobals), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_BUFFER(voxel_malloc_page_allocator, daxa_RWBufferPtr(VoxelMallocPageAllocator), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(voxel_chunks, daxa_BufferPtr(VoxelLeafChunk), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(simulated_voxel_particles, daxa_RWBufferPtr(SimulatedVoxelParticle), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_BUFFER(rendered_voxel_particles, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_BUFFER(placed_voxel_particles, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_USES_END()

#if defined(__cplusplus)

struct VoxelParticleSimComputeTaskState {
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    VoxelParticleSimComputeTaskState(daxa::PipelineManager &pipeline_manager) {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"voxel_particle_sim.comp.glsl"},
                .compile_options = {.defines = {{"VOXEL_PARTICLE_SIM_COMPUTE", "1"}}},
            },
            .name = "voxel_particle_sim",
        });
        if (compile_result.is_err()) {
            AppUi::Console::s_instance->add_log(compile_result.message());
            return;
        }
        pipeline = compile_result.value();
        if (!compile_result.value()->is_valid()) {
            AppUi::Console::s_instance->add_log(compile_result.message());
        }
    }
    auto pipeline_is_valid() -> bool { return pipeline && pipeline->is_valid(); }

    void record_commands(daxa::CommandList &cmd_list, daxa::BufferId globals_buffer_id) {
        if (!pipeline_is_valid()) {
            return;
        }
        cmd_list.set_pipeline(*pipeline);
        cmd_list.dispatch_indirect({
            .indirect_buffer = globals_buffer_id,
            .offset = offsetof(GpuGlobals, voxel_particles_state) + offsetof(VoxelParticlesState, simulation_dispatch),
        });
    }
};

struct VoxelParticleSimComputeTask : VoxelParticleSimComputeUses {
    VoxelParticleSimComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        state->record_commands(cmd_list, uses.globals.buffer());
    }
};

#endif
