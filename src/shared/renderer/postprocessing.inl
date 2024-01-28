#pragma once

#include <shared/core.inl>

#include <shared/renderer/blur.inl>
#include <shared/renderer/calculate_histogram.inl>

#if CompositingComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(CompositingCompute, 8)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, shadow_bitmap)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, CUBE, sky_cube)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, CUBE, ibl_cube)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, ssao_image_id)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, dst_image_id)
DAXA_DECL_TASK_HEAD_END
struct CompositingComputePush {
    DAXA_TH_BLOB(CompositingCompute, uses)
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(CompositingComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewId shadow_bitmap = push.uses.shadow_bitmap;
daxa_ImageViewId g_buffer_image_id = push.uses.g_buffer_image_id;
daxa_ImageViewId sky_cube = push.uses.sky_cube;
daxa_ImageViewId ibl_cube = push.uses.ibl_cube;
daxa_ImageViewId ssao_image_id = push.uses.ssao_image_id;
daxa_ImageViewId dst_image_id = push.uses.dst_image_id;
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

inline auto composite(RecordContext &record_ctx, GbufferDepth &gbuffer_depth, daxa::TaskImageView sky_cube, daxa::TaskImageView ibl_cube, daxa::TaskImageView ssao_image, daxa::TaskImageView shadow_bitmap) -> daxa::TaskImageView {
    auto output_image = record_ctx.task_graph.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
        .name = "composited_image",
    });

    record_ctx.add(ComputeTask<CompositingCompute, CompositingComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"postprocessing.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{CompositingCompute::gpu_input, record_ctx.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{CompositingCompute::globals, record_ctx.task_globals_buffer}},
            daxa::TaskViewVariant{std::pair{CompositingCompute::shadow_bitmap, shadow_bitmap}},
            daxa::TaskViewVariant{std::pair{CompositingCompute::g_buffer_image_id, gbuffer_depth.gbuffer}},
            daxa::TaskViewVariant{std::pair{CompositingCompute::sky_cube, sky_cube}},
            daxa::TaskViewVariant{std::pair{CompositingCompute::ibl_cube, ibl_cube}},
            daxa::TaskViewVariant{std::pair{CompositingCompute::ssao_image_id, ssao_image}},
            daxa::TaskViewVariant{std::pair{CompositingCompute::dst_image_id, output_image}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, CompositingComputePush &push, NoTaskInfo const &) {
            auto const image_info = ti.device.info_image(ti.get(CompositingCompute::dst_image_id).ids[0]).value();
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
            ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
        },
    });

    AppUi::DebugDisplay::s_instance->passes.push_back({.name = "composited_image", .task_image_id = output_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});

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
            auto outlier_frac_lo = std::min<double>(auto_exposure_settings.histogram_clip_low, 1.0f);
            auto outlier_frac_hi = std::min<double>(auto_exposure_settings.histogram_clip_high, 1.0f - outlier_frac_lo);

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
        auto rev_blur_pyramid = ::rev_blur_pyramid(record_ctx, blur_pyramid, image_size);
        return blur_pyramid;
    }
};

#endif
