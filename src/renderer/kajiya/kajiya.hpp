#pragma once

#include <renderer/kajiya/rtdgi.inl>
#include <renderer/kajiya/rtr.inl>
#include <renderer/kajiya/shadow_denoiser.inl>
#include <renderer/kajiya/taa.inl>
#include <renderer/kajiya/calculate_reprojection_map.inl>
#include <renderer/kajiya/ssao.inl>

#include <renderer/kajiya/light_gbuffer.inl>
#include <renderer/kajiya/postprocessing.hpp>

struct KajiyaRenderer {
    IrcacheRenderer ircache_renderer;
    SsaoRenderer ssao_renderer;
    RtdgiRenderer rtdgi_renderer;
    RtrRenderer rtr_renderer;
    TaaRenderer taa_renderer;
    ShadowDenoiser shadow_denoiser;
    PostProcessor post_processor;

    bool do_global_illumination = true;
    bool denoise_shadow_mask = false;

    void next_frame(daxa::Device &device, AutoExposureSettings const &auto_exposure_settings, float dt) {
        if (do_global_illumination) {
            ssao_renderer.next_frame();
            rtdgi_renderer.next_frame();
            rtr_renderer.next_frame();
            ircache_renderer.next_frame();
        }
        post_processor.next_frame(device, auto_exposure_settings, dt);
        const auto taa_method = AppSettings::get<settings::ComboBox>("Graphics", "TAA Method").value;
        if (taa_method == 1) {
            taa_renderer.next_frame();
        }
        if (denoise_shadow_mask) {
            shadow_denoiser.next_frame();
        }
    }

    auto render(
        RecordContext &record_ctx,
        GbufferDepth &gbuffer_depth,
        daxa::TaskImageView velocity_image,
        daxa::TaskImageView shadow_mask,
        daxa::TaskImageView ibl_cube,
        daxa::TaskImageView sky_lut,
        daxa::TaskImageView transmittance_lut,
        VoxelWorldBuffers &voxel_buffers) -> std::pair<daxa::TaskImageView, daxa::TaskImageView> {
        AppSettings::add<settings::Checkbox>({"Graphics", "global_illumination", {.value = do_global_illumination}, {.task_graph_depends = true}});
        AppSettings::add<settings::Checkbox>({"Graphics", "denoise_shadow_mask", {.value = denoise_shadow_mask}, {.task_graph_depends = true}});

        do_global_illumination = AppSettings::get<settings::Checkbox>("Graphics", "global_illumination").value;
        denoise_shadow_mask = AppSettings::get<settings::Checkbox>("Graphics", "denoise_shadow_mask").value;

        auto reprojection_map = calculate_reprojection_map(record_ctx, gbuffer_depth, velocity_image);
        auto denoised_shadow_mask = [&]() {
            if (denoise_shadow_mask) {
                return shadow_denoiser.denoise_shadow_mask(record_ctx, gbuffer_depth, shadow_mask, reprojection_map);
            } else {
                return shadow_mask;
            }
        }();

        auto rtr = daxa::TaskImageView{};
        auto rtdgi = daxa::TaskImageView{};

        AppSettings::add<settings::Checkbox>({"Graphics", "global_illumination", {.value = do_global_illumination}, {.task_graph_depends = true}});

        auto ircache_state = IrcacheRenderState{};
        if (do_global_illumination) {
            ircache_state = ircache_renderer.prepare(record_ctx);
            auto traced_ircache = ircache_state.trace_irradiance(record_ctx, voxel_buffers, ibl_cube, transmittance_lut);
            ircache_state.sum_up_irradiance_for_sampling(record_ctx, traced_ircache);

            auto reprojected_rtdgi = rtdgi_renderer.reproject(record_ctx, reprojection_map);
            auto ssgi_tex = ssao_renderer.render(record_ctx, gbuffer_depth, reprojection_map);

            auto rtdgi_ = rtdgi_renderer.render(
                record_ctx,
                reprojected_rtdgi,
                gbuffer_depth,
                reprojection_map,
                ibl_cube,
                transmittance_lut,
                ircache_state,
                voxel_buffers,
                ssgi_tex);

            auto &rtdgi_irradiance = rtdgi_.screen_irradiance_tex;
            auto &rtdgi_candidates = rtdgi_.candidates;

            auto rtr_ = rtr_renderer.trace(
                record_ctx,
                gbuffer_depth,
                reprojection_map,
                sky_lut,
                transmittance_lut,
                voxel_buffers,
                rtdgi_irradiance,
                rtdgi_candidates,
                ircache_state);

            rtr = rtr_.filter(record_ctx, gbuffer_depth, reprojection_map, rtr_renderer.spatial_resolve_offsets_buf);
            rtdgi = rtdgi_irradiance;
        } else {
            rtr = record_ctx.task_graph.create_transient_image({
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .name = "rtr",
            });
            rtdgi = record_ctx.task_graph.create_transient_image({
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .name = "rtdgi",
            });
            clear_task_images(record_ctx.task_graph, std::array<daxa::TaskImageView, 2>{rtr, rtdgi});
        }

        auto debug_out_tex = light_gbuffer(
            record_ctx,
            gbuffer_depth,
            denoised_shadow_mask,
            rtr,
            rtdgi,
            ircache_state,
            sky_lut,
            ibl_cube,
            transmittance_lut);

        return {debug_out_tex, reprojection_map};
    }

    auto upscale(RecordContext &record_ctx, daxa::TaskImageView input_image, daxa::TaskImageView depth_image, daxa::TaskImageView reprojection_map) -> daxa::TaskImageView {
        return taa_renderer.render(record_ctx, input_image, depth_image, reprojection_map);
    }

    auto post_process(RecordContext &record_ctx, daxa::TaskImageView input_image, daxa_u32vec2 image_size) -> daxa::TaskImageView {
        return post_processor.process(record_ctx, input_image, image_size);
    }
};
