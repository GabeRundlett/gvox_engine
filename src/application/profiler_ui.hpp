#pragma once

#include <base/profiler.hpp>

struct ProfilerUi {
    static constexpr int FRAME_HISTORY_COUNT = 500;
    static constexpr int MAX_DISPLAYED_THREADS = 17;

    ProfilerUi();

    Vec<ProfileTimestamp> frames[FRAME_HISTORY_COUNT][MAX_DISPLAYED_THREADS];
    uint64_t frame_thread_counts[FRAME_HISTORY_COUNT] = {};
    float frame_durations[FRAME_HISTORY_COUNT] = {};
    uint64_t frame_count = 0;

    Vec<ProfileTimestamp> startup_frames[MAX_DISPLAYED_THREADS];
    uint64_t startup_thread_count = 0;
    float startup_duration = 0.0f;
    bool startup_captured = false;

    static constexpr int64_t STARTUP_FRAME = -2;
    static constexpr int64_t LATEST_FRAME = -1;

    int64_t selected_frame = LATEST_FRAME;
    bool paused = false;
    float zoom = 1.0f;
    bool is_panning_flamegraph = false;

    void update(struct GpuContext &gpu_context);
    void ui_fullscreen();
    void ui_timeline();

  private:
    void draw_contents();
};
