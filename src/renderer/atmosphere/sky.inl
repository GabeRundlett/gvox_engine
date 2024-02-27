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

#if defined(__cplusplus)

#include <application/settings.hpp>
#include <numbers>

inline void add_sky_settings() {
    auto add_DensityProfileLayer = [](std::string_view name, DensityProfileLayer const &factory_default) {
        AppSettings::add<settings::SliderFloat>({"Atmosphere Advanced", std::string{name} + "_const_term", {.value = factory_default.const_term, .min = 0.0f, .max = 5.0f}});
        AppSettings::add<settings::SliderFloat>({"Atmosphere Advanced", std::string{name} + "_exp_term", {.value = factory_default.exp_term, .min = -1.0f, .max = 1.0f}});
        AppSettings::add<settings::SliderFloat>({"Atmosphere Advanced", std::string{name} + "_layer_width", {.value = factory_default.layer_width, .min = 0.1f, .max = 50.0f}});
        AppSettings::add<settings::SliderFloat>({"Atmosphere Advanced", std::string{name} + "_lin_term", {.value = factory_default.lin_term, .min = -0.5f, .max = 0.5f}});
    };

    auto mie_scale_height = 1.2000000476837158f;
    auto rayleigh_scale_height = 8.696f;
    AppSettings::add<settings::SliderFloat>({"Sun", "Angle X", {.value = 210.0f, .min = 0.0f, .max = 360.0f}});
    AppSettings::add<settings::SliderFloat>({"Sun", "Angle Y", {.value = 25.0f, .min = 0.0f, .max = 180.0f}});
    AppSettings::add<settings::SliderFloat>({"Sun", "Angular Radius", {.value = 0.25f, .min = 0.25f, .max = 30.0f}});
    AppSettings::add<settings::Checkbox>({"Sun", "Animate", {.value = false}});
    AppSettings::add<settings::SliderFloat>({"Sun", "Animate Speed", {.value = 0.1f, .min = 0.001f, .max = 1.0f}});
    AppSettings::add<settings::InputFloat>({"Atmosphere", "atmosphere_bottom", {.value = 6360.0f}});
    AppSettings::add<settings::InputFloat>({"Atmosphere", "atmosphere_top", {.value = 6460.0f}});
    AppSettings::add<settings::InputFloat3>({"Atmosphere", "mie_scattering", {.value = {0.003996000159531832f, 0.003996000159531832f, 0.003996000159531832f}}});
    AppSettings::add<settings::InputFloat3>({"Atmosphere", "mie_extinction", {.value = {0.00443999981507659f, 0.00443999981507659f, 0.00443999981507659f}}});
    AppSettings::add<settings::SliderFloat>({"Atmosphere", "mie_scale_height", {.value = mie_scale_height, .min = 0.0f, .max = 10.0f}});
    AppSettings::add<settings::SliderFloat>({"Atmosphere", "mie_phase_function_g", {.value = 0.800000011920929f, .min = 0.0f, .max = 1.0f}});
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
    AppSettings::add<settings::InputFloat3>({"Atmosphere", "rayleigh_scattering", {.value = {0.006604931f, 0.013344918f, 0.029412623f}}});
    AppSettings::add<settings::SliderFloat>({"Atmosphere", "rayleigh_scale_height", {.value = rayleigh_scale_height, .min = 0.0f, .max = 10.0f}});
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
    AppSettings::add<settings::InputFloat3>({"Atmosphere", "absorption_extinction", {.value = {0.00229072f, 0.00214036f, 0.0f}}});
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
}
inline auto get_sky_settings(float time) -> SkySettings {
    add_sky_settings();
    auto radians = [](float x) -> float {
        return x * std::numbers::pi_v<float> / 180.0f;
    };
    auto get_DensityProfileLayer = [](std::string_view name) -> DensityProfileLayer {
        return DensityProfileLayer{
            .const_term = AppSettings::get<settings::SliderFloat>("Atmosphere Advanced", std::string{name} + "_const_term").value,
            .exp_term = AppSettings::get<settings::SliderFloat>("Atmosphere Advanced", std::string{name} + "_exp_term").value,
            .layer_width = AppSettings::get<settings::SliderFloat>("Atmosphere Advanced", std::string{name} + "_layer_width").value,
            .lin_term = AppSettings::get<settings::SliderFloat>("Atmosphere Advanced", std::string{name} + "_lin_term").value,
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

    sun_angle_y = radians(sun_angle_y);
    if (AppSettings::get<settings::Checkbox>("Sun", "Animate").value) {
        sun_angle_y += time * AppSettings::get<settings::SliderFloat>("Sun", "Animate Speed").value;
    }

    result.sun_direction = {
        daxa_f32(std::cos(radians(sun_angle_x)) * std::sin(sun_angle_y)),
        daxa_f32(std::sin(radians(sun_angle_x)) * std::sin(sun_angle_y)),
        daxa_f32(std::cos(sun_angle_y)),
    };
    return result;
}

struct SkyRenderer {
    TemporalImage temporal_transmittance_lut;
    TemporalImage temporal_sky_lut;
    TemporalImage temporal_ibl_cube;

    void render(RecordContext &record_ctx, daxa::TaskImageView sky_lut, daxa::TaskImageView transmittance_lut) {
        add_sky_settings();

        temporal_transmittance_lut = record_ctx.gpu_context->find_or_add_temporal_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {SKY_TRANSMITTANCE_RES.x, SKY_TRANSMITTANCE_RES.y, 1},
            .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST,
            .name = "temporal transmittance_lut",
        });
        temporal_sky_lut = record_ctx.gpu_context->find_or_add_temporal_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {SKY_SKY_RES.x, SKY_SKY_RES.y, 1},
            .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST,
            .name = "temporal sky_lut",
        });
        temporal_ibl_cube = record_ctx.gpu_context->find_or_add_temporal_image({
            .flags = daxa::ImageCreateFlagBits::COMPATIBLE_CUBE,
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {IBL_CUBE_RES, IBL_CUBE_RES, 1},
            .array_layer_count = 6,
            .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST,
            .name = "temporal ibl_cube",
        });

        record_ctx.task_graph.use_persistent_image(temporal_transmittance_lut.task_resource);
        record_ctx.task_graph.use_persistent_image(temporal_sky_lut.task_resource);
        record_ctx.task_graph.use_persistent_image(temporal_ibl_cube.task_resource);

        record_ctx.task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_READ, transmittance_lut),
                daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, temporal_transmittance_lut.task_resource),
                daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_READ, sky_lut),
                daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, temporal_sky_lut.task_resource),
            },
            .task = [](daxa::TaskInterface const &ti) {
                ti.recorder.copy_image_to_image({
                    .src_image = ti.get(daxa::TaskImageAttachmentIndex{0}).ids[0],
                    .src_image_layout = ti.get(daxa::TaskImageAttachmentIndex{0}).layout,
                    .dst_image = ti.get(daxa::TaskImageAttachmentIndex{1}).ids[0],
                    .dst_image_layout = ti.get(daxa::TaskImageAttachmentIndex{1}).layout,
                    .extent = {SKY_TRANSMITTANCE_RES.x, SKY_TRANSMITTANCE_RES.y, 1},
                });
                ti.recorder.copy_image_to_image({
                    .src_image = ti.get(daxa::TaskImageAttachmentIndex{2}).ids[0],
                    .src_image_layout = ti.get(daxa::TaskImageAttachmentIndex{2}).layout,
                    .dst_image = ti.get(daxa::TaskImageAttachmentIndex{3}).ids[0],
                    .dst_image_layout = ti.get(daxa::TaskImageAttachmentIndex{3}).layout,
                    .extent = {SKY_SKY_RES.x, SKY_SKY_RES.y, 1},
                });
            },
            .name = "copy sky_lut",
        });

        auto ibl_cube = temporal_ibl_cube.task_resource.view().view({.layer_count = 6});

        convolve_cube(record_ctx, sky_lut, transmittance_lut, ibl_cube);
    }
};

inline auto generate_procedural_sky(RecordContext &record_ctx) -> std::array<daxa::TaskImageView, 2> {
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
        .source = daxa::ShaderFile{"atmosphere/sky.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{SkyTransmittanceCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
            daxa::TaskViewVariant{std::pair{SkyTransmittanceCompute::transmittance_lut, transmittance_lut}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SkyTransmittanceComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({(SKY_TRANSMITTANCE_RES.x + 7) / 8, (SKY_TRANSMITTANCE_RES.y + 3) / 4});
        },
    });
    record_ctx.add(ComputeTask<SkyMultiscatteringCompute, SkyMultiscatteringComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"atmosphere/sky.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{SkyMultiscatteringCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
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
        .source = daxa::ShaderFile{"atmosphere/sky.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{SkySkyCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
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

    // TODO(grundlett): You need to fix this. The views can easily be invalid, as these one are.
    // These views are transients for a different task graph!!!
    // debug_utils::DebugDisplay::add_pass({.name = "transmittance_lut", .task_image_id = transmittance_lut, .type = DEBUG_IMAGE_TYPE_DEFAULT});
    // debug_utils::DebugDisplay::add_pass({.name = "multiscattering_lut", .task_image_id = multiscattering_lut, .type = DEBUG_IMAGE_TYPE_DEFAULT});
    // debug_utils::DebugDisplay::add_pass({.name = "sky_lut", .task_image_id = sky_lut, .type = DEBUG_IMAGE_TYPE_DEFAULT});

    return {sky_lut, transmittance_lut};
}

#endif
