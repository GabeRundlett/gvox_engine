#pragma once

#include <shared/core.inl>

DAXA_DECL_TASK_USES_BEGIN(ChunkOptComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(globals, daxa_BufferPtr(GpuGlobals), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(temp_voxel_chunks, daxa_BufferPtr(TempVoxelChunk), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(voxel_chunks, daxa_RWBufferPtr(VoxelLeafChunk), COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_USES_END()

#if defined(__cplusplus)

template <int PASS_INDEX>
struct ChunkOptComputeTaskState {
    std::shared_ptr<daxa::ComputePipeline> pipeline;

    ChunkOptComputeTaskState(daxa::PipelineManager &pipeline_manager) {
        char const define_str[2] = {'0' + PASS_INDEX, '\0'};
        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"chunk_opt.comp.glsl"},
                .compile_options = {
                    .defines = {
                        {"CHUNK_OPT_COMPUTE", "1"},
                        {"CHUNK_OPT_STAGE", define_str},
                    },
                },
            },
            .name = "chunk_op",
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

    auto get_pass_indirect_offset() {
        if constexpr (PASS_INDEX == 0) {
            return offsetof(GpuGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, subchunk_x2x4_dispatch);
        } else {
            return offsetof(GpuGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, subchunk_x8up_dispatch);
        }
    }

    void record_commands(daxa::CommandList &cmd_list, daxa::BufferId globals_buffer_id) {
        if (!pipeline_is_valid()) {
            return;
        }
        cmd_list.set_pipeline(*pipeline);
        cmd_list.dispatch_indirect({
            .indirect_buffer = globals_buffer_id,
            .offset = get_pass_indirect_offset(),
        });
    }
};

template <int PASS_INDEX>
struct ChunkOptComputeTask : ChunkOptComputeUses {
    ChunkOptComputeTaskState<PASS_INDEX> *state;
    void callback(daxa::TaskInterface const &ti) {
        auto cmd_list = ti.get_command_list();
        cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        state->record_commands(cmd_list, uses.globals.buffer());
    }
};

using ChunkOpt_x2x4_ComputeTaskState = ChunkOptComputeTaskState<0>;
using ChunkOpt_x8up_ComputeTaskState = ChunkOptComputeTaskState<1>;
using ChunkOpt_x2x4_ComputeTask = ChunkOptComputeTask<0>;
using ChunkOpt_x8up_ComputeTask = ChunkOptComputeTask<1>;

#endif
