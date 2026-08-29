#pragma once

#include <base/profiler.hpp>
#include <cstdint>

struct ProfilerUi {
    static constexpr int FRAME_HISTORY_COUNT = 500;
    static constexpr int MAX_DISPLAYED_THREADS = 17;

    ProfilerUi();

    Vec<ProfileTimestamp> frames[FRAME_HISTORY_COUNT][MAX_DISPLAYED_THREADS];
    uint64_t frame_thread_counts[FRAME_HISTORY_COUNT] = {};
    float frame_durations[FRAME_HISTORY_COUNT] = {};
    uint64_t frame_count = 0;

    int64_t selected_frame = -1;
    bool paused = false;
    float zoom = 1.0f;
    bool is_panning_flamegraph = false;

    void update(struct GpuContext &gpu_context);
    void ui_fullscreen();
    void ui_timeline();

  private:
    void draw_contents();
};
