#include <renderer/renderer.hpp>

#include <renderer/trace_primary.inl>
#include <renderer/trace_secondary.inl>
#include <renderer/postprocessing.inl>
#include <renderer/sky.inl>
#include <renderer/fsr.inl>

#include <renderer/kajiya/kajiya.hpp>

struct RendererImpl {
    GbufferRenderer gbuffer_renderer;
    KajiyaRenderer kajiya_renderer;
    std::unique_ptr<Fsr2Renderer> fsr2_renderer;
    SkyRenderer sky;
};

Renderer::Renderer() : impl{std::make_unique<RendererImpl>()} {
}
Renderer::~Renderer() = default;

void Renderer::create(daxa::Device &device) {
    auto &self = *impl;
    self.kajiya_renderer.create(device);
    self.sky.create(device);
}

void Renderer::destroy(daxa::Device &device) {
    auto &self = *impl;
    self.kajiya_renderer.destroy(device);
    self.sky.destroy(device);
}

void Renderer::begin_frame(GpuInput &gpu_input, GpuOutput &gpu_output) {
    auto &self = *impl;

    gpu_input.sky_settings = get_sky_settings();

    gpu_input.pre_exposure = self.kajiya_renderer.post_processor.exposure_state.pre_mult;
    gpu_input.pre_exposure_prev = self.kajiya_renderer.post_processor.exposure_state.pre_mult_prev;
    gpu_input.pre_exposure_delta = self.kajiya_renderer.post_processor.exposure_state.pre_mult_delta;

    self.kajiya_renderer.ircache_renderer.update_eye_position(gpu_input, gpu_output);

    if constexpr (!ENABLE_TAA) {
        self.fsr2_renderer->next_frame();
        self.fsr2_renderer->state.delta_time = gpu_input.delta_time;
        gpu_input.halton_jitter = self.fsr2_renderer->state.jitter;
    }
}

void Renderer::end_frame(daxa::Device &device, RendererSettings const &settings, float dt) {
    auto &self = *impl;
    self.gbuffer_renderer.next_frame();
    self.kajiya_renderer.next_frame(device, settings.auto_exposure, dt);
}

auto Renderer::render(RecordContext &record_ctx, VoxelWorldBuffers &voxel_buffers, VoxelParticles &particles, daxa::TaskImageView output_image, daxa::Format output_format) -> daxa::TaskImageView {
    debug_utils::DebugDisplay::begin_passes();

    auto &self = *impl;

#if IMMEDIATE_SKY
    auto [sky_lut, transmittance_lut, sky_cube, ibl_cube] = generate_procedural_sky(record_ctx);
#else
    sky.use_images(record_ctx);
    auto sky_cube = sky.task_sky_cube.view().view({.layer_count = 6});
    auto ibl_cube = sky.task_ibl_cube.view().view({.layer_count = 6});
#endif
    debug_utils::DebugDisplay::add_pass({.name = "sky_cube", .task_image_id = sky_cube, .type = DEBUG_IMAGE_TYPE_CUBEMAP});
    debug_utils::DebugDisplay::add_pass({.name = "ibl_cube", .task_image_id = ibl_cube, .type = DEBUG_IMAGE_TYPE_CUBEMAP});

    auto [particles_color_image, particles_depth_image] = particles.render(record_ctx);
    auto [gbuffer_depth, velocity_image] = self.gbuffer_renderer.render(record_ctx, voxel_buffers, particles.task_simulated_voxel_particles_buffer, particles_color_image, particles_depth_image);

    auto shadow_mask = trace_shadows(record_ctx, gbuffer_depth, voxel_buffers);

    auto [debug_out_tex, reprojection_map] = self.kajiya_renderer.render(
        record_ctx,
        gbuffer_depth,
        velocity_image,
        shadow_mask,
        sky_cube,
        ibl_cube,
        transmittance_lut,
        voxel_buffers);

    self.fsr2_renderer = std::make_unique<Fsr2Renderer>(record_ctx.device, Fsr2Info{.render_resolution = record_ctx.render_resolution, .display_resolution = record_ctx.output_resolution});

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
