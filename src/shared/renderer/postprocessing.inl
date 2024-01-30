#pragma once

#include <shared/core.inl>

#include <shared/renderer/blur.inl>
#include <shared/renderer/calculate_histogram.inl>
#include <shared/renderer/ircache.inl>

#define SHADING_MODE_DEFAULT 0
#define SHADING_MODE_NO_TEXTURES 1
#define SHADING_MODE_DIFFUSE_GI 2
#define SHADING_MODE_REFLECTIONS 3
#define SHADING_MODE_RTX_OFF 4
#define SHADING_MODE_IRCACHE 5

#if LightGbufferComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(LightGbufferCompute, 21)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, gbuffer_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, shadow_mask_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, rtr_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, rtdgi_tex)
// DEFINE_IRCACHE_BINDINGS(5, 6, 7, 8, 9, 10, 11, 12, 13)
// DEFINE_WRC_BINDINGS(14)
// DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, temporal_output_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, output_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, CUBE, unconvolved_sky_cube_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, CUBE, sky_cube_tex)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, sky_lut)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(IrcacheBuffers), ircache_buffers)

DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE, ircache_grid_meta_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE, ircache_meta_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE, ircache_pool_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE, ircache_life_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE, ircache_entry_cell_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE, ircache_reposition_proposal_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE, ircache_irradiance_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE, ircache_reposition_proposal_count_buf)

DAXA_DECL_TASK_HEAD_END
struct LightGbufferComputePush {
    daxa_f32vec4 output_tex_size;
    daxa_u32 debug_shading_mode;
    daxa_u32 debug_show_wrc;
    DAXA_TH_BLOB(LightGbufferCompute, uses)
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(LightGbufferComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewId gbuffer_tex = push.uses.gbuffer_tex;
daxa_ImageViewId depth_tex = push.uses.depth_tex;
daxa_ImageViewId shadow_mask_tex = push.uses.shadow_mask_tex;
daxa_ImageViewId rtr_tex = push.uses.rtr_tex;
daxa_ImageViewId rtdgi_tex = push.uses.rtdgi_tex;
// IRCACHE and WRC
// daxa_ImageViewId temporal_output_tex = push.uses.temporal_output_tex;
daxa_ImageViewId output_tex = push.uses.output_tex;
daxa_ImageViewId unconvolved_sky_cube_tex = push.uses.unconvolved_sky_cube_tex;
daxa_ImageViewId sky_cube_tex = push.uses.sky_cube_tex;
daxa_ImageViewId sky_lut = push.uses.sky_lut;
daxa_ImageViewId transmittance_lut = push.uses.transmittance_lut;
daxa_BufferPtr(IrcacheBuffers) ircache_buffers = push.uses.ircache_buffers;

daxa_RWBufferPtr(IrcacheCell) ircache_grid_meta_buf = deref(ircache_buffers).ircache_grid_meta_buf;
daxa_BufferPtr(IrcacheMetadata) ircache_meta_buf = deref(ircache_buffers).ircache_meta_buf;
daxa_BufferPtr(daxa_u32) ircache_pool_buf = deref(ircache_buffers).ircache_pool_buf;
daxa_RWBufferPtr(daxa_u32) ircache_life_buf = deref(ircache_buffers).ircache_life_buf;
daxa_RWBufferPtr(daxa_u32) ircache_entry_cell_buf = deref(ircache_buffers).ircache_entry_cell_buf;
daxa_RWBufferPtr(VertexPacked) ircache_reposition_proposal_buf = deref(ircache_buffers).ircache_reposition_proposal_buf;
daxa_BufferPtr(daxa_f32vec4) ircache_irradiance_buf = deref(ircache_buffers).ircache_irradiance_buf;
daxa_BufferPtr(daxa_u32) ircache_reposition_proposal_count_buf = deref(ircache_buffers).ircache_reposition_proposal_count_buf;

#endif
#endif

#if PostprocessingRasterShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(PostprocessingRaster, 3)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(FRAGMENT_SHADER_SAMPLED, REGULAR_2D, composited_image_id)
DAXA_TH_IMAGE_ID(COLOR_ATTACHMENT, REGULAR_2D, render_image)
DAXA_DECL_TASK_HEAD_END
struct PostprocessingRasterPush {
    DAXA_TH_BLOB(PostprocessingRaster, uses)
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(PostprocessingRasterPush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewId composited_image_id = push.uses.composited_image_id;
daxa_ImageViewId render_image = push.uses.render_image;
#endif
#endif

#if DebugImageRasterShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(DebugImageRaster, 4)
DAXA_TH_BUFFER_PTR(FRAGMENT_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(FRAGMENT_SHADER_SAMPLED, REGULAR_2D, image_id)
DAXA_TH_IMAGE_ID(FRAGMENT_SHADER_SAMPLED, REGULAR_2D_ARRAY, cube_image_id)
DAXA_TH_IMAGE_ID(COLOR_ATTACHMENT, REGULAR_2D, render_image)
DAXA_DECL_TASK_HEAD_END
struct DebugImageRasterPush {
    daxa_u32 type;
    daxa_u32 cube_size;
    daxa_u32vec2 output_tex_size;
    DAXA_TH_BLOB(DebugImageRaster, uses)
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(DebugImageRasterPush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewId image_id = push.uses.image_id;
daxa_ImageViewId cube_image_id = push.uses.cube_image_id;
daxa_ImageViewId render_image = push.uses.render_image;
#endif
#endif

#if defined(__cplusplus)
#include <numeric>
#include <algorithm>
#include <fmt/format.h>

inline auto light_gbuffer(
    RecordContext &record_ctx,
    GbufferDepth &gbuffer_depth,
    daxa::TaskImageView shadow_mask,
    daxa::TaskImageView rtr,
    daxa::TaskImageView rtdgi,
    IrcacheRenderState &ircache,
    // wrc: &WrcRenderState,
    // temporal_output: &mut rg::Handle<Image>,
    // output: &mut rg::Handle<Image>,
    daxa::TaskImageView sky_cube,
    daxa::TaskImageView convolved_sky_cube,
    daxa::TaskImageView sky_lut,
    daxa::TaskImageView transmittance_lut) -> daxa::TaskImageView {

    auto output_image = record_ctx.task_graph.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
        .name = "composited_image",
    });

    record_ctx.add(ComputeTask<LightGbufferCompute, LightGbufferComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"postprocessing.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::gpu_input, record_ctx.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::globals, record_ctx.task_globals_buffer}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::gbuffer_tex, gbuffer_depth.gbuffer}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::depth_tex, gbuffer_depth.depth.current().view()}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::shadow_mask_tex, shadow_mask}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::rtr_tex, rtr}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::rtdgi_tex, rtdgi}},
            // daxa::TaskViewVariant{std::pair{LightGbufferCompute::temporal_output_tex, output_image}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::output_tex, output_image}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::unconvolved_sky_cube_tex, sky_cube}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::sky_cube_tex, convolved_sky_cube}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::sky_lut, sky_lut}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::transmittance_lut, transmittance_lut}},

            daxa::TaskViewVariant{std::pair{LightGbufferCompute::ircache_buffers, ircache.ircache_buffers}},

            daxa::TaskViewVariant{std::pair{LightGbufferCompute::ircache_grid_meta_buf, ircache.ircache_grid_meta_buf}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::ircache_meta_buf, ircache.ircache_meta_buf}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::ircache_pool_buf, ircache.ircache_pool_buf}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::ircache_life_buf, ircache.ircache_life_buf}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::ircache_entry_cell_buf, ircache.ircache_entry_cell_buf}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::ircache_reposition_proposal_buf, ircache.ircache_reposition_proposal_buf}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::ircache_irradiance_buf, ircache.ircache_irradiance_buf}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::ircache_reposition_proposal_count_buf, ircache.ircache_reposition_proposal_count_buf}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, LightGbufferComputePush &push, NoTaskInfo const &) {
            auto const image_info = ti.device.info_image(ti.get(LightGbufferCompute::gbuffer_tex).ids[0]).value();
            ti.recorder.set_pipeline(pipeline);
            push.debug_shading_mode = SHADING_MODE_DEFAULT;
            push.debug_show_wrc = 0;
            push.output_tex_size.x = static_cast<float>(image_info.size.x);
            push.output_tex_size.y = static_cast<float>(image_info.size.y);
            push.output_tex_size.z = 1.0f / push.output_tex_size.x;
            push.output_tex_size.w = 1.0f / push.output_tex_size.y;
            set_push_constant(ti, push);
            // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
            ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
        },
    });

    AppUi::DebugDisplay::s_instance->passes.push_back({.name = "lit gbuffer", .task_image_id = output_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});

    return output_image;
}

inline void tonemap_raster(RecordContext &record_ctx, daxa::TaskImageView antialiased_image, daxa::TaskImageView output_image, daxa::Format output_format) {
    record_ctx.add(RasterTask<PostprocessingRaster, PostprocessingRasterPush, NoTaskInfo>{
        .vert_source = daxa::ShaderFile{"FULL_SCREEN_TRIANGLE_VERTEX_SHADER"},
        .frag_source = daxa::ShaderFile{"postprocessing.comp.glsl"},
        .color_attachments = {{
            .format = output_format,
        }},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{PostprocessingRaster::gpu_input, record_ctx.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{PostprocessingRaster::composited_image_id, antialiased_image}},
            daxa::TaskViewVariant{std::pair{PostprocessingRaster::render_image, output_image}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, PostprocessingRasterPush &push, NoTaskInfo const &) {
            auto render_image = ti.get(PostprocessingRaster::render_image).ids[0];
            auto const image_info = ti.device.info_image(render_image).value();
            auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                .color_attachments = {{.image_view = render_image.default_view(), .load_op = daxa::AttachmentLoadOp::DONT_CARE, .clear_value = std::array<daxa_f32, 4>{0.0f, 0.0f, 0.0f, 0.0f}}},
                .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
            });
            renderpass_recorder.set_pipeline(pipeline);
            set_push_constant(ti, renderpass_recorder, push);
            renderpass_recorder.draw({.vertex_count = 3});
            ti.recorder = std::move(renderpass_recorder).end_renderpass();
        },
    });
}

inline void debug_pass(RecordContext &record_ctx, AppUi::Pass const &pass, daxa::TaskImageView output_image, daxa::Format output_format) {
    struct DebugImageRasterTaskInfo {
        daxa_u32 type;
    };

    record_ctx.add(RasterTask<DebugImageRaster, DebugImageRasterPush, DebugImageRasterTaskInfo>{
        .vert_source = daxa::ShaderFile{"FULL_SCREEN_TRIANGLE_VERTEX_SHADER"},
        .frag_source = daxa::ShaderFile{"postprocessing.comp.glsl"},
        .color_attachments = {{
            .format = output_format,
        }},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{DebugImageRaster::gpu_input, record_ctx.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{DebugImageRaster::image_id, pass.type == DEBUG_IMAGE_TYPE_CUBEMAP ? record_ctx.task_debug_texture : pass.task_image_id}},
            daxa::TaskViewVariant{std::pair{DebugImageRaster::cube_image_id, pass.type == DEBUG_IMAGE_TYPE_CUBEMAP ? pass.task_image_id : record_ctx.task_debug_texture}},
            daxa::TaskViewVariant{std::pair{DebugImageRaster::render_image, output_image}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, DebugImageRasterPush &push, DebugImageRasterTaskInfo const &info) {
            auto render_image = ti.get(DebugImageRaster::render_image).ids[0];
            auto const image_info = ti.device.info_image(render_image).value();
            auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                .color_attachments = {{.image_view = render_image.default_view(), .load_op = daxa::AttachmentLoadOp::DONT_CARE, .clear_value = std::array<daxa_f32, 4>{0.0f, 0.0f, 0.0f, 0.0f}}},
                .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
            });
            push.type = info.type;
            if (info.type == DEBUG_IMAGE_TYPE_CUBEMAP) {
                push.cube_size = ti.device.info_image(ti.get(DebugImageRaster::cube_image_id).ids[0]).value().size.x;
            }
            push.output_tex_size = {image_info.size.x, image_info.size.y};
            renderpass_recorder.set_pipeline(pipeline);
            set_push_constant(ti, renderpass_recorder, push);
            renderpass_recorder.draw({.vertex_count = 3});
            ti.recorder = std::move(renderpass_recorder).end_renderpass();
        },
        .info = {
            .type = pass.type,
        },
    });
}

struct ExposureState {
    float pre_mult = 1.0f;
    float post_mult = 1.0f;
    float pre_mult_prev = 1.0f;
    float pre_mult_delta = 1.0f;
};

struct DynamicExposureState {
    float ev_fast = 0.0f;
    float ev_slow = 0.0f;

    auto ev_smoothed() -> float {
        const float DYNAMIC_EXPOSURE_BIAS = -2.0f;
        auto &self = *this;
        return (self.ev_slow + self.ev_fast) * 0.5f + DYNAMIC_EXPOSURE_BIAS;
    }

    void update(float ev, float dt, float speed) {
        // dyn exposure update
        auto &self = *this;
        ev = std::clamp<float>(ev, LUMINANCE_HISTOGRAM_MIN_LOG2, LUMINANCE_HISTOGRAM_MAX_LOG2);
        dt = dt * speed; // std::exp2f(self.speed_log2);
        auto t_fast = 1.0f - std::expf(-1.0f * dt);
        self.ev_fast = (ev - self.ev_fast) * t_fast + self.ev_fast;

        auto t_slow = 1.0f - std::expf(-0.25f * dt);
        self.ev_slow = (ev - self.ev_slow) * t_slow + self.ev_slow;
    }
};

struct PostProcessor {
    daxa::Device device;
    std::array<daxa::BufferId, FRAMES_IN_FLIGHT + 1> histogram_buffers;
    daxa::TaskBuffer task_histogram_buffer;
    size_t histogram_buffer_index = 0;

    ExposureState exposure_state{};
    DynamicExposureState dynamic_exposure{};

    std::array<uint32_t, LUMINANCE_HISTOGRAM_BIN_COUNT> histogram{};

    PostProcessor(daxa::Device a_device) : device{std::move(a_device)} {
        uint32_t i = 0;
        for (auto &histogram_buffer : histogram_buffers) {
            histogram_buffer = device.create_buffer(daxa::BufferInfo{
                .size = static_cast<uint32_t>(sizeof(uint32_t) * LUMINANCE_HISTOGRAM_BIN_COUNT),
                .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = "histogram_buffer " + std::to_string(i++),
            });
        }
        task_histogram_buffer = daxa::TaskBuffer({.name = "histogram_buffer"});
        task_histogram_buffer.set_buffers({.buffers = std::array{histogram_buffers[histogram_buffer_index]}});
    }

    ~PostProcessor() {
        for (auto &histogram_buffer : histogram_buffers) {
            device.destroy_buffer(histogram_buffer);
        }
    }

    void next_frame(AutoExposureSettings const &auto_exposure_settings, float dt) {
        ++histogram_buffer_index;
        {
            auto buffer_i = (histogram_buffer_index + 0) % histogram_buffers.size();
            auto readable_buffer_i = (histogram_buffer_index + 1) % histogram_buffers.size();
            task_histogram_buffer.set_buffers({.buffers = std::array{histogram_buffers[buffer_i]}});
            auto const &histogram_buffer = histogram_buffers[readable_buffer_i];
            histogram = *device.get_host_address_as<std::array<uint32_t, LUMINANCE_HISTOGRAM_BIN_COUNT>>(histogram_buffer).value();

            // operate on histogram
            auto outlier_frac_lo = std::min<double>(auto_exposure_settings.histogram_clip_low, 1.0);
            auto outlier_frac_hi = std::min<double>(auto_exposure_settings.histogram_clip_high, 1.0 - outlier_frac_lo);

            auto total_entry_count = std::accumulate(histogram.begin(), histogram.end(), 0);
            auto reject_lo_entry_count = static_cast<uint32_t>(total_entry_count * outlier_frac_lo);
            auto entry_count_to_use = static_cast<uint32_t>(total_entry_count * (1.0 - outlier_frac_lo - outlier_frac_hi));

            auto sum = 0.0;
            auto used_count = 0u;

            auto left_to_reject = reject_lo_entry_count;
            auto left_to_use = entry_count_to_use;

            auto bin_idx = size_t{0};
            for (auto const &count : histogram) {
                auto t = (double(bin_idx) + 0.5) / double(LUMINANCE_HISTOGRAM_BIN_COUNT);

                auto count_to_use = std::min(std::max(count, left_to_reject) - left_to_reject, left_to_use);
                left_to_reject = std::max(left_to_reject, count) - count;
                left_to_use = std::max(left_to_use, count_to_use) - count_to_use;

                sum += t * double(count_to_use);
                used_count += count_to_use;
                ++bin_idx;
            }
            // AppUi::Console::s_instance->add_log(fmt::format("{}", used_count));

            auto mean = sum / std::max(used_count, 1u);
            auto image_log2_lum = float(LUMINANCE_HISTOGRAM_MIN_LOG2 + mean * (LUMINANCE_HISTOGRAM_MAX_LOG2 - LUMINANCE_HISTOGRAM_MIN_LOG2));

            dynamic_exposure.update(-image_log2_lum, dt, auto_exposure_settings.speed);

            auto ev_mult = std::exp2f(auto_exposure_settings.ev_shift + dynamic_exposure.ev_smoothed());
            exposure_state.pre_mult_prev = exposure_state.pre_mult;
            exposure_state.pre_mult = exposure_state.pre_mult * 0.9f + ev_mult * 0.1f;
            // Put the rest in post-exposure.
            exposure_state.post_mult = ev_mult / exposure_state.pre_mult;

            exposure_state.pre_mult_delta = exposure_state.pre_mult / exposure_state.pre_mult_prev;
        }
    }

    auto process(RecordContext &record_ctx, daxa::TaskImageView input_image, daxa_u32vec2 image_size) -> daxa::TaskImageView {
        record_ctx.task_graph.use_persistent_buffer(task_histogram_buffer);
        auto blur_pyramid = ::blur_pyramid(record_ctx, input_image, image_size);
        calculate_luminance_histogram(record_ctx, blur_pyramid, task_histogram_buffer, image_size);
        // auto rev_blur_pyramid = ::rev_blur_pyramid(record_ctx, blur_pyramid, image_size);
        return blur_pyramid;
    }
};

#endif
