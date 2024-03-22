#include <renderer/renderer.hpp>

#include <renderer/trace_primary.inl>
#include <renderer/trace_secondary.inl>
#include <renderer/postprocessing.inl>
#include <renderer/atmosphere/sky.inl>
#include <renderer/fsr.inl>

#include <renderer/kajiya/kajiya.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct RendererImpl {
    GbufferRenderer gbuffer_renderer;
    KajiyaRenderer kajiya_renderer;
    std::unique_ptr<Fsr2Renderer> fsr2_renderer;
    SkyRenderer sky;

    std::array<daxa_f32vec2, 128> halton_offsets{};
};

Renderer::Renderer() : impl{std::make_unique<RendererImpl>()} {
    auto &self = *impl;

    AppSettings::add<settings::SliderFloat>({"Camera", "Exposure Hist Clip Low", {.value = 0.1f, .min = 0.0f, .max = 1.0f}});
    AppSettings::add<settings::SliderFloat>({"Camera", "Exposure Hist Clip High", {.value = 0.1f, .min = 0.0f, .max = 1.0f}});
    AppSettings::add<settings::SliderFloat>({"Camera", "Exposure Reaction Speed", {.value = 3.0f, .min = 0.0f, .max = 10.0f}});
    AppSettings::add<settings::SliderFloat>({"Camera", "Exposure Shift", {.value = 0.0f, .min = -15.0f, .max = 15.0f}});

    AppSettings::add<settings::ComboBox>({"Graphics", "TAA Method", {.value = 1}, {.task_graph_depends = true, .options = {"None", "Kajiya TAA", "FSR 2.2"}}});
    AppSettings::add<settings::Checkbox>({"Graphics", "Update Sky", {.value = true}, {.task_graph_depends = true}});

    auto radical_inverse = [](daxa_u32 n, daxa_u32 base) -> daxa_f32 {
        auto val = 0.0f;
        auto inv_base = 1.0f / static_cast<daxa_f32>(base);
        auto inv_bi = inv_base;
        while (n > 0) {
            auto d_i = n % base;
            val += static_cast<daxa_f32>(d_i) * inv_bi;
            n = static_cast<daxa_u32>(static_cast<daxa_f32>(n) * inv_base);
            inv_bi *= inv_base;
        }
        return val;
    };

    for (daxa_u32 i = 0; i < self.halton_offsets.size(); ++i) {
        self.halton_offsets[i] = daxa_f32vec2{radical_inverse(i, 2) - 0.5f, radical_inverse(i, 3) - 0.5f};
    }
}
Renderer::~Renderer() = default;

void Renderer::begin_frame(GpuInput &gpu_input) {
    auto &self = *impl;

    gpu_input.sky_settings = get_sky_settings(gpu_input.time);

    gpu_input.pre_exposure = self.kajiya_renderer.post_processor.exposure_state.pre_mult;
    gpu_input.pre_exposure_prev = self.kajiya_renderer.post_processor.exposure_state.pre_mult_prev;
    gpu_input.pre_exposure_delta = self.kajiya_renderer.post_processor.exposure_state.pre_mult_delta;

    {
        auto center = glm::floor(std::bit_cast<glm::vec3>(gpu_input.player.pos) + std::bit_cast<glm::vec3>(gpu_input.player.forward) * 10.0f);
        auto eye = center - std::bit_cast<glm::vec3>(gpu_input.sky_settings.sun_direction) * 60.0f;
        auto projection = glm::ortho(-20.0f, 20.0f, -20.0f, 20.0f, 0.1f, 120.0f) * glm::lookAt(eye, center, glm::vec3(0, 0, 1));
        gpu_input.ws_to_shadow = std::bit_cast<daxa_f32mat4x4>(projection);
        gpu_input.shadow_to_ws = std::bit_cast<daxa_f32mat4x4>(glm::inverse(projection));
    }

    self.kajiya_renderer.ircache_renderer.update_eye_position(gpu_input);

    const auto taa_method = AppSettings::get<settings::ComboBox>("Graphics", "TAA Method").value;
    switch (taa_method) {
    case 0: break;
    case 1:
        gpu_input.halton_jitter = self.halton_offsets[gpu_input.frame_index % self.halton_offsets.size()];
        break;
    case 2:
        if (self.fsr2_renderer) {
            self.fsr2_renderer->next_frame();
            self.fsr2_renderer->state.delta_time = gpu_input.delta_time;
            gpu_input.halton_jitter = self.fsr2_renderer->state.jitter;
        }
        break;
    }

    auto update_sky = AppSettings::get<settings::Checkbox>("Graphics", "Update Sky").value;
    if (update_sky || gpu_input.frame_index == 0) {
        self.sky.sky_render_task_graph.execute({});
    }
}

void Renderer::end_frame(daxa::Device &device, float dt) {
    auto &self = *impl;
    self.gbuffer_renderer.next_frame();
    auto auto_exposure_settings = AutoExposureSettings{
        .histogram_clip_low = AppSettings::get<settings::SliderFloat>("Camera", "Exposure Hist Clip Low").value,
        .histogram_clip_high = AppSettings::get<settings::SliderFloat>("Camera", "Exposure Hist Clip High").value,
        .speed = AppSettings::get<settings::SliderFloat>("Camera", "Exposure Reaction Speed").value,
        .ev_shift = AppSettings::get<settings::SliderFloat>("Camera", "Exposure Shift").value,
    };
    self.kajiya_renderer.next_frame(device, auto_exposure_settings, dt);
}

auto Renderer::render(GpuContext &gpu_context, VoxelWorldBuffers &voxel_buffers, VoxelParticles &particles, daxa::TaskImageView output_image, daxa::Format output_format) -> daxa::TaskImageView {
    debug_utils::DebugDisplay::begin_passes();

    auto &self = *impl;

    self.sky.render(gpu_context);

    auto transmittance_lut = self.sky.transmittance_lut.task_resource.view();
    auto sky_lut = self.sky.sky_lut.task_resource.view();
    auto ibl_cube = self.sky.ibl_cube.task_resource.view().view({.base_array_layer = 0, .layer_count = 6});
    auto ae_lut = self.sky.aerial_perspective_lut.task_resource.view();
    gpu_context.frame_task_graph.use_persistent_image(self.sky.transmittance_lut.task_resource);
    gpu_context.frame_task_graph.use_persistent_image(self.sky.sky_lut.task_resource);
    gpu_context.frame_task_graph.use_persistent_image(self.sky.ibl_cube.task_resource);
    gpu_context.frame_task_graph.use_persistent_image(self.sky.aerial_perspective_lut.task_resource);

    debug_utils::DebugDisplay::add_pass({.name = "transmittance_lut", .task_image_id = transmittance_lut, .type = DEBUG_IMAGE_TYPE_DEFAULT});
    debug_utils::DebugDisplay::add_pass({.name = "sky_lut", .task_image_id = sky_lut, .type = DEBUG_IMAGE_TYPE_DEFAULT});
    debug_utils::DebugDisplay::add_pass({.name = "ibl_cube", .task_image_id = ibl_cube, .type = DEBUG_IMAGE_TYPE_CUBEMAP});
    debug_utils::DebugDisplay::add_pass({.name = "ae_lut", .task_image_id = ae_lut, .type = DEBUG_IMAGE_TYPE_3D});

    auto [gbuffer_depth, velocity_image] = self.gbuffer_renderer.render(gpu_context, voxel_buffers);
    auto particles_shadow_depth_image = particles.render(gpu_context, gbuffer_depth, velocity_image);

    auto shadow_mask = trace_shadows(gpu_context, gbuffer_depth, voxel_buffers, particles_shadow_depth_image);

    auto [debug_out_tex, reprojection_map] = self.kajiya_renderer.render(
        gpu_context,
        gbuffer_depth,
        velocity_image,
        shadow_mask,
        ibl_cube,
        sky_lut,
        transmittance_lut,
        ae_lut,
        voxel_buffers);

    auto antialiased_image = [&]() {
        const auto taa_method = AppSettings::get<settings::ComboBox>("Graphics", "TAA Method").value;
        switch (taa_method) {
        default: [[fallthrough]];
        case 0: {
            auto output_image = gpu_context.frame_task_graph.create_transient_image({
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {gpu_context.output_resolution.x, gpu_context.output_resolution.y, 1},
                .name = "output_image",
            });

            gpu_context.frame_task_graph.add_task({
                .attachments = {
                    daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_READ, daxa::ImageViewType::REGULAR_2D, debug_out_tex),
                    daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, daxa::ImageViewType::REGULAR_2D, output_image),
                },
                .task = [=](daxa::TaskInterface const &ti) {
                    auto image_a = ti.get(daxa::TaskImageAttachmentIndex{0}).ids[0];
                    auto image_b = ti.get(daxa::TaskImageAttachmentIndex{1}).ids[0];
                    auto image_a_info = ti.device.info_image(image_a).value();
                    auto image_b_info = ti.device.info_image(image_b).value();

                    ti.recorder.blit_image_to_image({
                        .src_image = image_a,
                        .src_image_layout = ti.get(daxa::TaskImageAttachmentIndex{0}).layout,
                        .dst_image = image_b,
                        .dst_image_layout = ti.get(daxa::TaskImageAttachmentIndex{1}).layout,
                        .src_offsets = {{{0, 0, 0}, {static_cast<int32_t>(image_a_info.size.x), static_cast<int32_t>(image_a_info.size.y), static_cast<int32_t>(image_a_info.size.z)}}},
                        .dst_offsets = {{{0, 0, 0}, {static_cast<int32_t>(image_b_info.size.x), static_cast<int32_t>(image_b_info.size.y), static_cast<int32_t>(image_b_info.size.z)}}},
                        .filter = daxa::Filter::LINEAR,
                    });
                },
                .name = "upscale_output_image",
            });

            return output_image;
        }
        case 1:
            return self.kajiya_renderer.upscale(gpu_context, debug_out_tex, gbuffer_depth.depth.current(), reprojection_map);
        case 2:
            self.fsr2_renderer = std::make_unique<Fsr2Renderer>(gpu_context.device, Fsr2Info{.render_resolution = gpu_context.render_resolution, .display_resolution = gpu_context.output_resolution});
            return self.fsr2_renderer->upscale(gpu_context, gbuffer_depth, debug_out_tex, reprojection_map);
        }
    }();

    [[maybe_unused]] auto post_processed_image = self.kajiya_renderer.post_process(gpu_context, antialiased_image, gpu_context.output_resolution);

    debug_utils::DebugDisplay::add_pass({.name = "[final]"});

    auto &dbg_disp = *debug_utils::DebugDisplay::s_instance;
    auto pass_iter = std::find_if(dbg_disp.passes.begin(), dbg_disp.passes.end(), [&](auto &pass) { return pass.name == dbg_disp.selected_pass_name; });
    if (pass_iter == dbg_disp.passes.end() || dbg_disp.selected_pass_name == "[final]") {
        tonemap_raster(gpu_context, antialiased_image, output_image, output_format);
    } else {
        debug_pass(gpu_context, *pass_iter, output_image, output_format);
    }

    return antialiased_image;
}
