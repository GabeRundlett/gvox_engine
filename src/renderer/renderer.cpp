#include <renderer/renderer.hpp>

#include <renderer/trace_primary.inl>
#include <renderer/trace_secondary.inl>
#include <renderer/postprocessing.inl>
#include <renderer/atmosphere/sky.inl>
#include <renderer/fsr.inl>

#include <renderer/kajiya/kajiya.hpp>

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
    AppSettings::add<settings::SliderFloat>({"Camera", "Exposure Shift", {.value = -0.5f, .min = -15.0f, .max = 15.0f}});

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

    gpu_input.sky_settings = get_sky_settings();

    gpu_input.pre_exposure = self.kajiya_renderer.post_processor.exposure_state.pre_mult;
    gpu_input.pre_exposure_prev = self.kajiya_renderer.post_processor.exposure_state.pre_mult_prev;
    gpu_input.pre_exposure_delta = self.kajiya_renderer.post_processor.exposure_state.pre_mult_delta;

    self.kajiya_renderer.ircache_renderer.update_eye_position(gpu_input);

    if constexpr (!ENABLE_TAA) {
        self.fsr2_renderer->next_frame();
        self.fsr2_renderer->state.delta_time = gpu_input.delta_time;
        gpu_input.halton_jitter = self.fsr2_renderer->state.jitter;
    } else {
        gpu_input.halton_jitter = self.halton_offsets[gpu_input.frame_index % self.halton_offsets.size()];
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

auto Renderer::render(RecordContext &record_ctx, VoxelWorldBuffers &voxel_buffers, VoxelParticles &particles, daxa::TaskImageView output_image, daxa::Format output_format) -> daxa::TaskImageView {
    debug_utils::DebugDisplay::begin_passes();

    auto &self = *impl;

    auto [sky_lut, transmittance_lut, ibl_cube] = generate_procedural_sky(record_ctx);
    debug_utils::DebugDisplay::add_pass({.name = "ibl_cube", .task_image_id = ibl_cube, .type = DEBUG_IMAGE_TYPE_CUBEMAP});

    auto [particles_color_image, particles_depth_image] = particles.render(record_ctx);
    auto [gbuffer_depth, velocity_image] = self.gbuffer_renderer.render(record_ctx, voxel_buffers, particles.simulated_voxel_particles.task_resource, particles_color_image, particles_depth_image);

    auto shadow_mask = trace_shadows(record_ctx, gbuffer_depth, voxel_buffers);

    auto [debug_out_tex, reprojection_map] = self.kajiya_renderer.render(
        record_ctx,
        gbuffer_depth,
        velocity_image,
        shadow_mask,
        ibl_cube,
        sky_lut,
        transmittance_lut,
        voxel_buffers);

    self.fsr2_renderer = std::make_unique<Fsr2Renderer>(record_ctx.gpu_context->device, Fsr2Info{.render_resolution = record_ctx.render_resolution, .display_resolution = record_ctx.output_resolution});

    auto antialiased_image = [&]() {
        if constexpr (ENABLE_TAA) {
            return self.kajiya_renderer.upscale(record_ctx, debug_out_tex, gbuffer_depth.depth.current(), reprojection_map);
        } else {
            return self.fsr2_renderer->upscale(record_ctx, gbuffer_depth, debug_out_tex, reprojection_map);
        }
    }();

    [[maybe_unused]] auto post_processed_image = self.kajiya_renderer.post_process(record_ctx, antialiased_image, record_ctx.output_resolution);

    debug_utils::DebugDisplay::add_pass({.name = "[final]"});

    auto &dbg_disp = *debug_utils::DebugDisplay::s_instance;
    auto pass_iter = std::find_if(dbg_disp.passes.begin(), dbg_disp.passes.end(), [&](auto &pass) { return pass.name == dbg_disp.selected_pass_name; });
    if (pass_iter == dbg_disp.passes.end() || dbg_disp.selected_pass_name == "[final]") {
        tonemap_raster(record_ctx, antialiased_image, output_image, output_format);
    } else {
        debug_pass(record_ctx, *pass_iter, output_image, output_format);
    }

    return antialiased_image;
}
