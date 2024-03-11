#pragma once

#include <core.inl>
#include <renderer/core.inl>

DAXA_DECL_TASK_HEAD_BEGIN(SkyTransmittanceCompute, 2)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, transmittance_lut)
DAXA_DECL_TASK_HEAD_END
struct SkyTransmittanceComputePush {
    DAXA_TH_BLOB(SkyTransmittanceCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(SkyMultiscatteringCompute, 3)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, multiscattering_lut)
DAXA_DECL_TASK_HEAD_END
struct SkyMultiscatteringComputePush {
    DAXA_TH_BLOB(SkyMultiscatteringCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(SkySkyCompute, 4)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, multiscattering_lut)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, sky_lut)
DAXA_DECL_TASK_HEAD_END
struct SkySkyComputePush {
    DAXA_TH_BLOB(SkySkyCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(SkyAeCompute, 4)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, multiscattering_lut)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_3D, aerial_perspective_lut)
DAXA_DECL_TASK_HEAD_END
struct SkyAeComputePush {
    DAXA_TH_BLOB(SkyAeCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(ConvolveCubeCompute, 4)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, sky_lut)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D_ARRAY, ibl_cube)
DAXA_DECL_TASK_HEAD_END
struct ConvolveCubeComputePush {
    daxa_u32 flags;
    DAXA_TH_BLOB(ConvolveCubeCompute, uses)
};

#if defined(__cplusplus)

#include <application/settings.hpp>
#include <numbers>
#include <cmath>

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
    TemporalImage transmittance_lut;
    TemporalImage sky_lut;
    TemporalImage ibl_cube;
    TemporalImage aerial_perspective_lut;

    daxa::TaskGraph sky_render_task_graph;

    void generate_procedural_sky(GpuContext &gpu_context) {
        auto multiscattering_lut = sky_render_task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {SKY_MULTISCATTERING_RES.x, SKY_MULTISCATTERING_RES.y, 1},
            .name = "multiscattering_lut",
        });
        gpu_context.add(ComputeTask<SkyTransmittanceCompute, SkyTransmittanceComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"atmosphere/sky.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{SkyTransmittanceCompute::gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{SkyTransmittanceCompute::transmittance_lut, transmittance_lut.task_resource}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SkyTransmittanceComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(SKY_TRANSMITTANCE_RES.x + 7) / 8, (SKY_TRANSMITTANCE_RES.y + 3) / 4});
            },
            .task_graph_ptr = &sky_render_task_graph,
        });
        gpu_context.add(ComputeTask<SkyMultiscatteringCompute, SkyMultiscatteringComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"atmosphere/sky.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{SkyMultiscatteringCompute::gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{SkyMultiscatteringCompute::transmittance_lut, transmittance_lut.task_resource}},
                daxa::TaskViewVariant{std::pair{SkyMultiscatteringCompute::multiscattering_lut, multiscattering_lut}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SkyMultiscatteringComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({SKY_MULTISCATTERING_RES.x, SKY_MULTISCATTERING_RES.y});
            },
            .task_graph_ptr = &sky_render_task_graph,
        });
        gpu_context.add(ComputeTask<SkySkyCompute, SkySkyComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"atmosphere/sky.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{SkySkyCompute::gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{SkySkyCompute::transmittance_lut, transmittance_lut.task_resource}},
                daxa::TaskViewVariant{std::pair{SkySkyCompute::multiscattering_lut, multiscattering_lut}},
                daxa::TaskViewVariant{std::pair{SkySkyCompute::sky_lut, sky_lut.task_resource}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SkySkyComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(SKY_SKY_RES.x + 7) / 8, (SKY_SKY_RES.y + 3) / 4});
            },
            .task_graph_ptr = &sky_render_task_graph,
        });
        gpu_context.add(ComputeTask<SkyAeCompute, SkyAeComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"atmosphere/sky.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{SkyAeCompute::gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{SkyAeCompute::transmittance_lut, transmittance_lut.task_resource}},
                daxa::TaskViewVariant{std::pair{SkyAeCompute::multiscattering_lut, multiscattering_lut}},
                daxa::TaskViewVariant{std::pair{SkyAeCompute::aerial_perspective_lut, aerial_perspective_lut.task_resource}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SkyAeComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({SKY_AE_RES.x, SKY_AE_RES.y, 1});
            },
            .task_graph_ptr = &sky_render_task_graph,
        });

        // TODO(grundlett): You need to fix this. The views can easily be invalid, as these one are.
        // These views are transients for a different task graph!!!
        // debug_utils::DebugDisplay::add_pass({.name = "transmittance_lut", .task_image_id = transmittance_lut, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        // debug_utils::DebugDisplay::add_pass({.name = "multiscattering_lut", .task_image_id = multiscattering_lut, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        // debug_utils::DebugDisplay::add_pass({.name = "sky_lut", .task_image_id = sky_lut, .type = DEBUG_IMAGE_TYPE_DEFAULT});
    }

    void convolve_cube(GpuContext &gpu_context) {
        auto ibl_cube_view = ibl_cube.task_resource.view().view({.layer_count = 6});

        gpu_context.add(ComputeTask<ConvolveCubeCompute, ConvolveCubeComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"atmosphere/convolve_cube.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{ConvolveCubeCompute::gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{ConvolveCubeCompute::sky_lut, sky_lut.task_resource}},
                daxa::TaskViewVariant{std::pair{ConvolveCubeCompute::transmittance_lut, transmittance_lut.task_resource}},
                daxa::TaskViewVariant{std::pair{ConvolveCubeCompute::ibl_cube, ibl_cube_view}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ConvolveCubeComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                auto do_global_illumination = AppSettings::get<settings::Checkbox>("Graphics", "global_illumination").value;
                push.flags |= do_global_illumination ? 1 : 0;
                set_push_constant(ti, push);
                ti.recorder.dispatch({(IBL_CUBE_RES + 7) / 8, (IBL_CUBE_RES + 7) / 8, 6});
            },
            .task_graph_ptr = &sky_render_task_graph,
        });
    }

    void render(GpuContext &gpu_context) {
        add_sky_settings();

        sky_render_task_graph = daxa::TaskGraph({
            .device = gpu_context.device,
            .name = "sky_render_task_graph",
        });

        sky_render_task_graph.use_persistent_buffer(gpu_context.task_input_buffer);

        transmittance_lut = gpu_context.find_or_add_temporal_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {SKY_TRANSMITTANCE_RES.x, SKY_TRANSMITTANCE_RES.y, 1},
            .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST,
            .name = "temporal transmittance_lut",
        });
        sky_lut = gpu_context.find_or_add_temporal_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {SKY_SKY_RES.x, SKY_SKY_RES.y, 1},
            .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST,
            .name = "temporal sky_lut",
        });
        ibl_cube = gpu_context.find_or_add_temporal_image({
            .flags = daxa::ImageCreateFlagBits::COMPATIBLE_CUBE,
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {IBL_CUBE_RES, IBL_CUBE_RES, 1},
            .array_layer_count = 6,
            .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST,
            .name = "temporal ibl_cube",
        });
        aerial_perspective_lut = gpu_context.find_or_add_temporal_image({
            .dimensions = 3,
            .format = daxa::Format::R32G32B32A32_SFLOAT,
            .size = {SKY_AE_RES.x, SKY_AE_RES.y, SKY_AE_RES.z},
            .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST,
            .name = "temporal ae_lut",
        });

        sky_render_task_graph.use_persistent_image(transmittance_lut.task_resource);
        sky_render_task_graph.use_persistent_image(sky_lut.task_resource);
        sky_render_task_graph.use_persistent_image(ibl_cube.task_resource);
        sky_render_task_graph.use_persistent_image(aerial_perspective_lut.task_resource);

        generate_procedural_sky(gpu_context);

        convolve_cube(gpu_context);

        sky_render_task_graph.submit({});
        sky_render_task_graph.complete({});
    }
};

#endif
