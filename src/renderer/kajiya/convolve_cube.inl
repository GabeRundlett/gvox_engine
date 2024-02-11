#pragma once

#include <core.inl>
#include <application/input.inl>

DAXA_DECL_TASK_HEAD_BEGIN(ConvolveCubeCompute, 3)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, CUBE, sky_cube)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D_ARRAY, ibl_cube)
DAXA_DECL_TASK_HEAD_END
struct ConvolveCubeComputePush {
    ConvolveCubeCompute uses;
};

#if defined(__cplusplus)

inline void convolve_cube(RecordContext &record_ctx, daxa::TaskImageView input_cube, daxa::TaskImageView output_cube) {
    record_ctx.add(ComputeTask<ConvolveCubeCompute, ConvolveCubeComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"kajiya/convolve_cube.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{ConvolveCubeCompute::gpu_input, record_ctx.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{ConvolveCubeCompute::sky_cube, input_cube}},
            daxa::TaskViewVariant{std::pair{ConvolveCubeCompute::ibl_cube, output_cube}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ConvolveCubeComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({(IBL_CUBE_RES + 7) / 8, (IBL_CUBE_RES + 7) / 8, 6});
        },
    });
}

#endif
