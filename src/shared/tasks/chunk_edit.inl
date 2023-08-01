#pragma once

#include <shared/core.inl>

#if CHUNK_EDIT_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(ChunkEditComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_BufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(gvox_model, daxa_BufferPtr(GpuGvoxModel), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(voxel_chunks, daxa_BufferPtr(VoxelLeafChunk), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(temp_voxel_chunks, daxa_RWBufferPtr(TempVoxelChunk), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_BUFFER(voxel_malloc_page_allocator, daxa_RWBufferPtr(VoxelMallocPageAllocator), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(simulated_voxel_particles, daxa_BufferPtr(SimulatedVoxelParticle), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(placed_voxel_particles, daxa_BufferPtr(daxa_u32), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(value_noise_texture, REGULAR_2D_ARRAY, COMPUTE_SHADER_SAMPLED)
DAXA_DECL_TASK_USES_END()
#endif

#if defined(__cplusplus)

struct ChunkEditComputeTaskState {
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    ChunkEditComputeTaskState(daxa::PipelineManager &pipeline_manager) {
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"chunk_edit.comp.glsl"},
                .compile_options = {.defines = {{"CHUNK_EDIT_COMPUTE", "1"}}},
            },
            .name = "chunk_edit",
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
            .offset = offsetof(GpuGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, chunk_edit_dispatch),
        });
    }
};

struct ChunkEditComputeTask : ChunkEditComputeUses {
    ChunkEditComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        state->record_commands(cmd_list, uses.globals.buffer());
    }
};

#endif
