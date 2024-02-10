#pragma once

#include <numeric>
#include <algorithm>
#include <fmt/format.h>

#include <renderer/kajiya/blur.inl>
#include <renderer/kajiya/calculate_histogram.inl>

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
    std::array<daxa::BufferId, FRAMES_IN_FLIGHT + 1> histogram_buffers;
    daxa::TaskBuffer task_histogram_buffer;
    size_t histogram_buffer_index = 0;

    ExposureState exposure_state{};
    DynamicExposureState dynamic_exposure{};

    std::array<uint32_t, LUMINANCE_HISTOGRAM_BIN_COUNT> histogram{};

    void create(daxa::Device &device) {
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

    void destroy(daxa::Device &device) const {
        for (auto &histogram_buffer : histogram_buffers) {
            device.destroy_buffer(histogram_buffer);
        }
    }

    void next_frame(daxa::Device &device, AutoExposureSettings const &auto_exposure_settings, float dt) {
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
            // debug_utils::Console::add_log(fmt::format("{}", used_count));

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
