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
#if SkyCubeComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(SkyCubeCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, sky_lut)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D_ARRAY, sky_cube)
DAXA_DECL_TASK_HEAD_END
struct SkyCubeComputePush {
    SkyCubeCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(SkyCubeComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewId transmittance_lut = push.uses.transmittance_lut;
daxa_ImageViewId sky_lut = push.uses.sky_lut;
daxa_ImageViewId sky_cube = push.uses.sky_cube;
#endif
#endif
#if IblCubeComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(IblCubeCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, CUBE, sky_cube)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, CUBE, ibl_cube)
DAXA_DECL_TASK_HEAD_END
struct IblCubeComputePush {
    IblCubeCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(IblCubeComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewId sky_cube = push.uses.sky_cube;
daxa_ImageViewId ibl_cube = push.uses.ibl_cube;
#endif
#endif

#if defined(__cplusplus)

struct SkyRenderer {
    daxa::ImageId sky_cube_image;
    daxa::ImageId ibl_cube_image;
    daxa::TaskImage task_sky_cube{{.name = "task_sky_cube"}};
    daxa::TaskImage task_ibl_cube{{.name = "task_ibl_cube"}};

    void create(daxa::Device &device) {
        sky_cube_image = device.create_image({
            .flags = daxa::ImageCreateFlagBits::COMPATIBLE_CUBE,
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {SKY_CUBE_RES, SKY_CUBE_RES, 1},
            .array_layer_count = 6,
            .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::TRANSFER_DST,
            .name = "sky_cube",
        });
        ibl_cube_image = device.create_image({
            .flags = daxa::ImageCreateFlagBits::COMPATIBLE_CUBE,
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {IBL_CUBE_RES, IBL_CUBE_RES, 1},
            .array_layer_count = 6,
            .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::TRANSFER_DST,
            .name = "ibl_cube",
        });
        task_sky_cube.set_images({.images = std::array{sky_cube_image}});
        task_ibl_cube.set_images({.images = std::array{ibl_cube_image}});
    }
    void destroy(daxa::Device &device) const {
        device.destroy_image(sky_cube_image);
        device.destroy_image(ibl_cube_image);
    }

    void use_images(RecordContext &record_ctx) {
        record_ctx.task_graph.use_persistent_image(task_sky_cube);
        record_ctx.task_graph.use_persistent_image(task_ibl_cube);
    }

    void render(RecordContext &record_ctx, daxa::TaskImageView input_sky_cube) {
        input_sky_cube = input_sky_cube;
        auto sky_cube = task_sky_cube.view().view({.layer_count = 6});
        auto ibl_cube = task_ibl_cube.view().view({.layer_count = 6});

        record_ctx.task_graph.add_task({
            .uses = {
                daxa::TaskImageUse<daxa::TaskImageAccess::TRANSFER_READ>{input_sky_cube},
                daxa::TaskImageUse<daxa::TaskImageAccess::TRANSFER_WRITE>{sky_cube},
            },
            .task = [=](daxa::TaskInterface ti) {
                ti.get_recorder().copy_image_to_image({
                    .src_image = ti.uses[input_sky_cube].image(),
                    .src_image_layout = ti.uses[input_sky_cube].layout(),
                    .dst_image = ti.uses[sky_cube].image(),
                    .dst_image_layout = ti.uses[sky_cube].layout(),
                    .src_slice = {.layer_count = 6},
                    .dst_slice = {.layer_count = 6},
                    .extent = ti.get_device().info_image(ti.uses[input_sky_cube].image()).value().size,
                });
            },
            .name = "transfer sky cube",
        });

        record_ctx.add(ComputeTask<IblCubeCompute, IblCubeComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"convolve_cube.comp.glsl"},
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .sky_cube = sky_cube,
                .ibl_cube = ibl_cube,
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, IblCubeCompute::Uses &, IblCubeComputePush &push, NoTaskInfo const &) {
                ti.get_recorder().set_pipeline(pipeline);
                ti.get_recorder().push_constant(push);
                ti.get_recorder().dispatch({(IBL_CUBE_RES + 7) / 8, (IBL_CUBE_RES + 7) / 8, 6});
            },
        });
    }
};

#if IMMEDIATE_SKY
inline auto generate_procedural_sky(RecordContext &record_ctx) -> std::pair<daxa::TaskImageView, daxa::TaskImageView> {
#else
inline auto generate_procedural_sky(RecordContext &record_ctx) -> daxa::TaskImageView {
#endif
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

    AppUi::DebugDisplay::s_instance->passes.push_back({.name = "transmittance_lut", .task_image_id = transmittance_lut, .type = DEBUG_IMAGE_TYPE_DEFAULT});
    AppUi::DebugDisplay::s_instance->passes.push_back({.name = "multiscattering_lut", .task_image_id = multiscattering_lut, .type = DEBUG_IMAGE_TYPE_DEFAULT});
    AppUi::DebugDisplay::s_instance->passes.push_back({.name = "sky_lut", .task_image_id = sky_lut, .type = DEBUG_IMAGE_TYPE_DEFAULT});

    auto sky_cube = record_ctx.task_graph.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {SKY_CUBE_RES, SKY_CUBE_RES, 1},
        .array_layer_count = 6,
        .name = "procedural_sky_cube",
    });
    sky_cube = sky_cube.view({.layer_count = 6});

    record_ctx.add(ComputeTask<SkyCubeCompute, SkyCubeComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"sky.comp.glsl"},
        .uses = {
            .gpu_input = record_ctx.task_input_buffer,
            .transmittance_lut = transmittance_lut,
            .sky_lut = sky_lut,
            .sky_cube = sky_cube,
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SkyCubeCompute::Uses &, SkyCubeComputePush &push, NoTaskInfo const &) {
            ti.get_recorder().set_pipeline(pipeline);
            ti.get_recorder().push_constant(push);
            ti.get_recorder().dispatch({(SKY_CUBE_RES + 7) / 8, (SKY_CUBE_RES + 7) / 8, 6});
        },
    });

    // AppUi::DebugDisplay::s_instance->passes.push_back({.name = "sky_cube", .task_image_id = sky_cube, .type = DEBUG_IMAGE_TYPE_CUBEMAP});

#if IMMEDIATE_SKY

    auto ibl_cube = record_ctx.task_graph.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {IBL_CUBE_RES, IBL_CUBE_RES, 1},
        .array_layer_count = 6,
        .name = "ibl_cube",
    });
    ibl_cube = ibl_cube.view({.layer_count = 6});
    record_ctx.add(ComputeTask<IblCubeCompute, IblCubeComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"convolve_cube.comp.glsl"},
        .uses = {
            .gpu_input = record_ctx.task_input_buffer,
            .sky_cube = sky_cube,
            .ibl_cube = ibl_cube,
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, IblCubeCompute::Uses &, IblCubeComputePush &push, NoTaskInfo const &) {
            ti.get_recorder().set_pipeline(pipeline);
            ti.get_recorder().push_constant(push);
            ti.get_recorder().dispatch({(IBL_CUBE_RES + 7) / 8, (IBL_CUBE_RES + 7) / 8, 6});
        },
    });

    return {sky_cube, ibl_cube};
#else
    return sky_cube;
#endif
}

#endif
