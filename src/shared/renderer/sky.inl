#pragma once

#include <shared/core.inl>

#if SkyTransmittanceComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(SkyTransmittanceCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, transmittance_lut)
DAXA_DECL_TASK_HEAD_END
struct SkyTransmittanceComputePush {
    SkyTransmittanceCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(SkyTransmittanceComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewId transmittance_lut = push.uses.transmittance_lut;
#endif
#endif
#if SkyMultiscatteringComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(SkyMultiscatteringCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, multiscattering_lut)
DAXA_DECL_TASK_HEAD_END
struct SkyMultiscatteringComputePush {
    SkyMultiscatteringCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(SkyMultiscatteringComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewId transmittance_lut = push.uses.transmittance_lut;
daxa_ImageViewId multiscattering_lut = push.uses.multiscattering_lut;
#endif
#endif
#if SkySkyComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(SkySkyCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, multiscattering_lut)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, sky_lut)
DAXA_DECL_TASK_HEAD_END
struct SkySkyComputePush {
    SkySkyCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(SkySkyComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewId transmittance_lut = push.uses.transmittance_lut;
daxa_ImageViewId multiscattering_lut = push.uses.multiscattering_lut;
daxa_ImageViewId sky_lut = push.uses.sky_lut;
#endif
#endif

#if defined(__cplusplus)

inline auto render_sky(RecordContext &record_ctx) -> std::pair<daxa::TaskImageView, daxa::TaskImageView> {
    auto transmittance_lut = record_ctx.task_graph.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {SKY_TRANSMITTANCE_RES.x, SKY_TRANSMITTANCE_RES.y, 1},
        .name = "transmittance_lut",
    });
    auto multiscattering_lut = record_ctx.task_graph.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {SKY_MULTISCATTERING_RES.x, SKY_MULTISCATTERING_RES.y, 1},
        .name = "multiscattering_lut",
    });
    auto sky_lut = record_ctx.task_graph.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {SKY_SKY_RES.x, SKY_SKY_RES.y, 1},
        .name = "sky_lut",
    });

    record_ctx.add(ComputeTask<SkyTransmittanceCompute, SkyTransmittanceComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"sky.comp.glsl"},
        .uses = {
            .gpu_input = record_ctx.task_input_buffer,
            .transmittance_lut = transmittance_lut,
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SkyTransmittanceCompute::Uses &, SkyTransmittanceComputePush &push, NoTaskInfo const &) {
            ti.get_recorder().set_pipeline(pipeline);
            ti.get_recorder().push_constant(push);
            ti.get_recorder().dispatch({(SKY_TRANSMITTANCE_RES.x + 7) / 8, (SKY_TRANSMITTANCE_RES.y + 3) / 4});
        },
    });
    record_ctx.add(ComputeTask<SkyMultiscatteringCompute, SkyMultiscatteringComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"sky.comp.glsl"},
        .uses = {
            .gpu_input = record_ctx.task_input_buffer,
            .transmittance_lut = transmittance_lut,
            .multiscattering_lut = multiscattering_lut,
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SkyMultiscatteringCompute::Uses &, SkyMultiscatteringComputePush &push, NoTaskInfo const &) {
            ti.get_recorder().set_pipeline(pipeline);
            ti.get_recorder().push_constant(push);
            ti.get_recorder().dispatch({SKY_MULTISCATTERING_RES.x, SKY_MULTISCATTERING_RES.y});
        },
    });
    record_ctx.add(ComputeTask<SkySkyCompute, SkySkyComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"sky.comp.glsl"},
        .uses = {
            .gpu_input = record_ctx.task_input_buffer,
            .transmittance_lut = transmittance_lut,
            .multiscattering_lut = multiscattering_lut,
            .sky_lut = sky_lut,
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SkySkyCompute::Uses &, SkySkyComputePush &push, NoTaskInfo const &) {
            ti.get_recorder().set_pipeline(pipeline);
            ti.get_recorder().push_constant(push);
            ti.get_recorder().dispatch({(SKY_SKY_RES.x + 7) / 8, (SKY_SKY_RES.y + 3) / 4});
        },
    });

    return {sky_lut, transmittance_lut};
}

#endif
