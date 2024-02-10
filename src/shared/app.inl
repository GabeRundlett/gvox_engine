#pragma once

#include <shared/core.inl>

#include <shared/input.inl>
#include <shared/globals.inl>

#include <shared/voxels/voxels.inl>

DAXA_DECL_TASK_HEAD_BEGIN(StartupCompute, 2 + VOXEL_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(GpuGlobals), globals)
VOXELS_USE_BUFFERS(daxa_RWBufferPtr, COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_HEAD_END
struct StartupComputePush {
    DAXA_TH_BLOB(StartupCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(PerframeCompute, 4 + VOXEL_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(GpuOutput), gpu_output)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(SimulatedVoxelParticle), simulated_voxel_particles)
VOXELS_USE_BUFFERS(daxa_RWBufferPtr, COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_HEAD_END
struct PerframeComputePush {
    DAXA_TH_BLOB(PerframeCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(TestCompute, 1)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32), data)
DAXA_DECL_TASK_HEAD_END
struct TestComputePush {
    DAXA_TH_BLOB(TestCompute, uses)
};

#if defined(__cplusplus)

inline void test_compute(RecordContext &record_ctx) {
    auto test_buffer = record_ctx.task_graph.create_transient_buffer({
        .size = static_cast<daxa_u32>(sizeof(uint32_t) * 8 * 8 * 8 * 64 * 64 * 64),
        .name = "test_buffer",
    });

    record_ctx.add(ComputeTask<TestCompute, TestComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"test.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{TestCompute::data, test_buffer}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TestComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            auto volume_size = uint32_t(8 * 64);
            ti.recorder.dispatch({(volume_size + 7) / 8, (volume_size + 7) / 8, (volume_size + 7) / 8});
        },
    });
}

#endif
