#pragma once

#include <core.inl>

DAXA_DECL_TASK_HEAD_BEGIN(PrefixScan1Compute, 1)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), inout_buf)
DAXA_DECL_TASK_HEAD_END
struct PrefixScan1ComputePush {
    daxa_u32 element_n;
    DAXA_TH_BLOB(PrefixScan1Compute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(PrefixScan2Compute, 2)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), input_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32), output_buf)
DAXA_DECL_TASK_HEAD_END
struct PrefixScan2ComputePush {
    daxa_u32 element_n;
    DAXA_TH_BLOB(PrefixScan2Compute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(PrefixScanMergeCompute, 2)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), inout_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), segment_sum_buf)
DAXA_DECL_TASK_HEAD_END
struct PrefixScanMergeComputePush {
    daxa_u32 element_n;
    DAXA_TH_BLOB(PrefixScanMergeCompute, uses)
};

#if defined(__cplusplus)

inline void inclusive_prefix_scan_u32_1m(GpuContext &gpu_context, daxa::TaskBufferView input_buf) {
    const auto SEGMENT_SIZE = uint32_t{1024};
    auto segment_sum_buf = gpu_context.frame_task_graph.create_transient_buffer({
        .size = sizeof(uint32_t) * SEGMENT_SIZE,
        .name = "segment_sum_buf",
    });

    gpu_context.add(ComputeTask<PrefixScan1Compute, PrefixScan1ComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"kajiya/prefix_scan.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{PrefixScan1Compute::inout_buf, input_buf}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, PrefixScan1ComputePush &push, NoTaskInfo const &) {
            push.element_n = static_cast<uint32_t>(ti.device.info_buffer(ti.get(PrefixScan1Compute::inout_buf).ids[0]).value().size / sizeof(uint32_t));
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({(SEGMENT_SIZE * SEGMENT_SIZE / 2 + 511) / 512});
        },
    });

    gpu_context.add(ComputeTask<PrefixScan2Compute, PrefixScan2ComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"kajiya/prefix_scan.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{PrefixScan2Compute::input_buf, input_buf}},
            daxa::TaskViewVariant{std::pair{PrefixScan2Compute::output_buf, segment_sum_buf}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, PrefixScan2ComputePush &push, NoTaskInfo const &) {
            push.element_n = static_cast<uint32_t>(ti.device.info_buffer(ti.get(PrefixScan2Compute::input_buf).ids[0]).value().size / sizeof(uint32_t));
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({(SEGMENT_SIZE / 2 + 511) / 512});
        },
    });

    gpu_context.add(ComputeTask<PrefixScanMergeCompute, PrefixScanMergeComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"kajiya/prefix_scan.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{PrefixScanMergeCompute::inout_buf, input_buf}},
            daxa::TaskViewVariant{std::pair{PrefixScanMergeCompute::segment_sum_buf, segment_sum_buf}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, PrefixScanMergeComputePush &push, NoTaskInfo const &) {
            push.element_n = static_cast<uint32_t>(ti.device.info_buffer(ti.get(PrefixScanMergeCompute::inout_buf).ids[0]).value().size / sizeof(uint32_t));
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({(SEGMENT_SIZE * SEGMENT_SIZE / 2 + 511) / 512});
        },
    });
}

#endif
