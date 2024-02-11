#pragma once

#include <core.inl>
#include <renderer/core.inl>

#include <renderer/kajiya/convolve_cube.inl>

DAXA_DECL_TASK_HEAD_BEGIN(SkyTransmittanceCompute, 2)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, transmittance_lut)
DAXA_DECL_TASK_HEAD_END
struct SkyTransmittanceComputePush {
    SkyTransmittanceCompute uses;
};
DAXA_DECL_TASK_HEAD_BEGIN(SkyMultiscatteringCompute, 3)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, multiscattering_lut)
DAXA_DECL_TASK_HEAD_END
struct SkyMultiscatteringComputePush {
    SkyMultiscatteringCompute uses;
};
DAXA_DECL_TASK_HEAD_BEGIN(SkySkyCompute, 4)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, multiscattering_lut)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, sky_lut)
DAXA_DECL_TASK_HEAD_END
struct SkySkyComputePush {
    SkySkyCompute uses;
};
DAXA_DECL_TASK_HEAD_BEGIN(SkyCubeCompute, 4)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, sky_lut)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D_ARRAY, sky_cube)
DAXA_DECL_TASK_HEAD_END
struct SkyCubeComputePush {
    SkyCubeCompute uses;
};

#if defined(__cplusplus)

#include <application/settings.hpp>
#include <numbers>

inline auto get_sky_settings() -> SkySettings {
    auto radians = [](float x) -> float {
        return x * std::numbers::pi_v<float> / 180.0f;
    };
    auto get_DensityProfileLayer = [](std::string_view name) -> DensityProfileLayer {
        return DensityProfileLayer{
            .const_term = AppSettings::get<settings::SliderFloat>("Atmosphere", std::string{name} + "_const_term").value,
            .exp_scale = AppSettings::get<settings::SliderFloat>("Atmosphere", std::string{name} + "_exp_scale").value,
            .exp_term = AppSettings::get<settings::SliderFloat>("Atmosphere", std::string{name} + "_exp_term").value,
            .layer_width = AppSettings::get<settings::SliderFloat>("Atmosphere", std::string{name} + "_layer_width").value,
            .lin_term = AppSettings::get<settings::SliderFloat>("Atmosphere", std::string{name} + "_lin_term").value,
        };
    };

    auto result = SkySettings{};
    auto sun_angle_x = AppSettings::get<settings::SliderFloat>("Sun", "Angle X").value;
    auto sun_angle_y = AppSettings::get<settings::SliderFloat>("Sun", "Angle Y").value;
    result.sun_angular_radius_cos = std::cos(radians(AppSettings::get<settings::SliderFloat>("Sun", "Angular Radius").value));
    result.atmosphere_bottom = AppSettings::get<settings::InputFloat>("Atmosphere", "atmosphere_bottom").value;
    result.atmosphere_top = AppSettings::get<settings::InputFloat>("Atmosphere", "atmosphere_top").value;
    result.mie_scattering = AppSettings::get<settings::InputFloat3>("Atmosphere", "mie_scattering").value;
    result.mie_extinction = AppSettings::get<settings::InputFloat3>("Atmosphere", "mie_extinction").value;
    result.mie_scale_height = AppSettings::get<settings::SliderFloat>("Atmosphere", "mie_scale_height").value;
    result.mie_phase_function_g = AppSettings::get<settings::SliderFloat>("Atmosphere", "mie_phase_function_g").value;
    result.mie_density[0] = get_DensityProfileLayer("mie_density_0");
    result.mie_density[1] = get_DensityProfileLayer("mie_density_1");
    result.mie_density[1].exp_scale = -1.0f / result.mie_scale_height;
    result.rayleigh_scattering = AppSettings::get<settings::InputFloat3>("Atmosphere", "rayleigh_scattering").value;
    result.rayleigh_scale_height = AppSettings::get<settings::SliderFloat>("Atmosphere", "rayleigh_scale_height").value;
    result.rayleigh_density[0] = get_DensityProfileLayer("rayleigh_density_0");
    result.rayleigh_density[1] = get_DensityProfileLayer("rayleigh_density_1");
    result.rayleigh_density[1].exp_scale = -1.0f / result.rayleigh_scale_height;
    result.absorption_extinction = AppSettings::get<settings::InputFloat3>("Atmosphere", "absorption_extinction").value;
    result.absorption_density[0] = get_DensityProfileLayer("absorption_density_0");
    result.absorption_density[1] = get_DensityProfileLayer("absorption_density_1");
    result.sun_direction = {
        daxa_f32(std::cos(radians(sun_angle_x)) * std::sin(radians(sun_angle_y))),
        daxa_f32(std::sin(radians(sun_angle_x)) * std::sin(radians(sun_angle_y))),
        daxa_f32(std::cos(radians(sun_angle_y))),
    };
    return result;
}

struct SkyRenderer {
    daxa::ImageId sky_cube_image;
    daxa::ImageId ibl_cube_image;
    daxa::TaskImage task_sky_cube{{.name = "task_sky_cube"}};
    daxa::TaskImage task_ibl_cube{{.name = "task_ibl_cube"}};

    void create(daxa::Device &device) {
        auto add_DensityProfileLayer = [](std::string_view name, DensityProfileLayer const &factory_default) {
            AppSettings::add(SettingInfo<settings::SliderFloat>{"Atmosphere", std::string{name} + "_const_term", {.value = factory_default.const_term, .min = 0.0f, .max = 5.0f}});
            AppSettings::add(SettingInfo<settings::SliderFloat>{"Atmosphere", std::string{name} + "_exp_scale", {.value = factory_default.exp_scale, .min = -1.0f, .max = 1.0f}});
            AppSettings::add(SettingInfo<settings::SliderFloat>{"Atmosphere", std::string{name} + "_exp_term", {.value = factory_default.exp_term, .min = -1.0f, .max = 1.0f}});
            AppSettings::add(SettingInfo<settings::SliderFloat>{"Atmosphere", std::string{name} + "_layer_width", {.value = factory_default.layer_width, .min = 0.1f, .max = 50.0f}});
            AppSettings::add(SettingInfo<settings::SliderFloat>{"Atmosphere", std::string{name} + "_lin_term", {.value = factory_default.lin_term, .min = -0.5f, .max = 0.5f}});
        };

        auto mie_scale_height = 1.2000000476837158f;
        auto rayleigh_scale_height = 8.696f;
        AppSettings::add(SettingInfo<settings::SliderFloat>{"Sun", "Angle X", {.value = 210.0f, .min = 0.0f, .max = 360.0f}});
        AppSettings::add(SettingInfo<settings::SliderFloat>{"Sun", "Angle Y", {.value = 25.0f, .min = 0.0f, .max = 180.0f}});
        AppSettings::add(SettingInfo<settings::SliderFloat>{"Sun", "Angular Radius", {.value = 0.25f, .min = 0.25f, .max = 30.0f}});
        AppSettings::add(SettingInfo<settings::InputFloat>{"Atmosphere", "atmosphere_bottom", {.value = 6360.0f}});
        AppSettings::add(SettingInfo<settings::InputFloat>{"Atmosphere", "atmosphere_top", {.value = 6460.0f}});
        AppSettings::add(SettingInfo<settings::InputFloat3>{"Atmosphere", "mie_scattering", {.value = {0.003996000159531832f, 0.003996000159531832f, 0.003996000159531832f}}});
        AppSettings::add(SettingInfo<settings::InputFloat3>{"Atmosphere", "mie_extinction", {.value = {0.00443999981507659f, 0.00443999981507659f, 0.00443999981507659f}}});
        AppSettings::add(SettingInfo<settings::SliderFloat>{"Atmosphere", "mie_scale_height", {.value = mie_scale_height, .min = 0.0f, .max = 10.0f}});
        AppSettings::add(SettingInfo<settings::SliderFloat>{"Atmosphere", "mie_phase_function_g", {.value = 0.800000011920929f, .min = 0.0f, .max = 1.0f}});
        add_DensityProfileLayer(
            "mie_density_0",
            DensityProfileLayer{
                .const_term = 0.0f,
                .exp_scale = 0.0f,
                .exp_term = 0.0f,
                .layer_width = 0.0f,
                .lin_term = 0.0f,
            });
        add_DensityProfileLayer(
            "mie_density_1",
            DensityProfileLayer{
                .const_term = 0.0f,
                .exp_scale = -1.0f / mie_scale_height,
                .exp_term = 1.0f,
                .layer_width = 0.0f,
                .lin_term = 0.0f,
            });
        AppSettings::add(SettingInfo<settings::InputFloat3>{"Atmosphere", "rayleigh_scattering", {.value = {0.006604931f, 0.013344918f, 0.029412623f}}});
        AppSettings::add(SettingInfo<settings::SliderFloat>{"Atmosphere", "rayleigh_scale_height", {.value = rayleigh_scale_height, .min = 0.0f, .max = 10.0f}});
        add_DensityProfileLayer(
            "rayleigh_density_0",
            DensityProfileLayer{
                .const_term = 0.0f,
                .exp_scale = 0.0f,
                .exp_term = 0.0f,
                .layer_width = 0.0f,
                .lin_term = 0.0f,
            });
        add_DensityProfileLayer(
            "rayleigh_density_1",
            DensityProfileLayer{
                .const_term = 0.0f,
                .exp_scale = -1.0f / rayleigh_scale_height,
                .exp_term = 1.0f,
                .layer_width = 0.0f,
                .lin_term = 0.0f,
            });
        AppSettings::add(SettingInfo<settings::InputFloat3>{"Atmosphere", "absorption_extinction", {.value = {0.00229072f, 0.00214036f, 0.0f}}});
        add_DensityProfileLayer(
            "absorption_density_0",
            DensityProfileLayer{
                .const_term = -0.6666600108146667f,
                .exp_scale = 0.0f,
                .exp_term = 0.0f,
                .layer_width = 25.0f,
                .lin_term = 0.06666599959135056f,
            });
        add_DensityProfileLayer(
            "absorption_density_1",
            DensityProfileLayer{
                .const_term = 2.6666600704193115f,
                .exp_scale = 0.0f,
                .exp_term = 0.0f,
                .layer_width = 0.0f,
                .lin_term = -0.06666599959135056f,
            });

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
        auto sky_cube = task_sky_cube.view().view({.layer_count = 6});
        auto ibl_cube = task_ibl_cube.view().view({.layer_count = 6});

        record_ctx.task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_READ, daxa::ImageViewType::REGULAR_2D, input_sky_cube),
                daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, daxa::ImageViewType::REGULAR_2D, sky_cube),
            },
            .task = [=](daxa::TaskInterface const &ti) {
                ti.recorder.copy_image_to_image({
                    .src_image = ti.get(daxa::TaskImageAttachmentIndex{0}).ids[0],
                    .src_image_layout = ti.get(daxa::TaskImageAttachmentIndex{0}).layout,
                    .dst_image = ti.get(daxa::TaskImageAttachmentIndex{1}).ids[0],
                    .dst_image_layout = ti.get(daxa::TaskImageAttachmentIndex{1}).layout,
                    .src_slice = {.layer_count = 6},
                    .dst_slice = {.layer_count = 6},
                    .extent = ti.device.info_image(ti.get(daxa::TaskImageAttachmentIndex{0}).ids[0]).value().size,
                });
            },
            .name = "transfer sky cube",
        });

        convolve_cube(record_ctx, sky_cube, ibl_cube);
    }
};

#if IMMEDIATE_SKY
inline auto generate_procedural_sky(RecordContext &record_ctx) -> std::array<daxa::TaskImageView, 4> {
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
        .views = std::array{
            daxa::TaskViewVariant{std::pair{SkyTransmittanceCompute::gpu_input, record_ctx.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{SkyTransmittanceCompute::transmittance_lut, transmittance_lut}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SkyTransmittanceComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({(SKY_TRANSMITTANCE_RES.x + 7) / 8, (SKY_TRANSMITTANCE_RES.y + 3) / 4});
        },
    });
    record_ctx.add(ComputeTask<SkyMultiscatteringCompute, SkyMultiscatteringComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"sky.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{SkyMultiscatteringCompute::gpu_input, record_ctx.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{SkyMultiscatteringCompute::transmittance_lut, transmittance_lut}},
            daxa::TaskViewVariant{std::pair{SkyMultiscatteringCompute::multiscattering_lut, multiscattering_lut}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SkyMultiscatteringComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({SKY_MULTISCATTERING_RES.x, SKY_MULTISCATTERING_RES.y});
        },
    });
    record_ctx.add(ComputeTask<SkySkyCompute, SkySkyComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"sky.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{SkySkyCompute::gpu_input, record_ctx.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{SkySkyCompute::transmittance_lut, transmittance_lut}},
            daxa::TaskViewVariant{std::pair{SkySkyCompute::multiscattering_lut, multiscattering_lut}},
            daxa::TaskViewVariant{std::pair{SkySkyCompute::sky_lut, sky_lut}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SkySkyComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({(SKY_SKY_RES.x + 7) / 8, (SKY_SKY_RES.y + 3) / 4});
        },
    });

    debug_utils::DebugDisplay::add_pass({.name = "transmittance_lut", .task_image_id = transmittance_lut, .type = DEBUG_IMAGE_TYPE_DEFAULT});
    debug_utils::DebugDisplay::add_pass({.name = "multiscattering_lut", .task_image_id = multiscattering_lut, .type = DEBUG_IMAGE_TYPE_DEFAULT});
    debug_utils::DebugDisplay::add_pass({.name = "sky_lut", .task_image_id = sky_lut, .type = DEBUG_IMAGE_TYPE_DEFAULT});

    auto sky_cube = record_ctx.task_graph.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {SKY_CUBE_RES, SKY_CUBE_RES, 1},
        .array_layer_count = 6,
        .name = "procedural_sky_cube",
    });
    sky_cube = sky_cube.view({.layer_count = 6});

    record_ctx.add(ComputeTask<SkyCubeCompute, SkyCubeComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"sky.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{SkyCubeCompute::gpu_input, record_ctx.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{SkyCubeCompute::transmittance_lut, transmittance_lut}},
            daxa::TaskViewVariant{std::pair{SkyCubeCompute::sky_lut, sky_lut}},
            daxa::TaskViewVariant{std::pair{SkyCubeCompute::sky_cube, sky_cube}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SkyCubeComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({(SKY_CUBE_RES + 7) / 8, (SKY_CUBE_RES + 7) / 8, 6});
        },
    });

    // debug_utils::DebugDisplay::add_pass({.name = "sky_cube", .task_image_id = sky_cube, .type = DEBUG_IMAGE_TYPE_CUBEMAP});

#if IMMEDIATE_SKY

    auto ibl_cube = record_ctx.task_graph.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {IBL_CUBE_RES, IBL_CUBE_RES, 1},
        .array_layer_count = 6,
        .name = "ibl_cube",
    });
    ibl_cube = ibl_cube.view({.layer_count = 6});

    convolve_cube(record_ctx, sky_cube, ibl_cube);

    return {sky_lut, transmittance_lut, sky_cube, ibl_cube};
#else
    return sky_cube;
#endif
}

#endif
