#include "profiler.hpp"

#include <cstdint>
#include <Windows.h>
#include <mutex>

__forceinline uint64_t get_timestamp() {
    LARGE_INTEGER i;
    QueryPerformanceCounter(&i);
    return (uint64_t)i.QuadPart;
}

float get_timestamp_frequency() {
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    return (float)frequency.QuadPart;
}

class ProfilerImpl {
  public:
    static const int MAX_THREAD_COUNT = 128;
    struct CpuTimeStamp {
        uint64_t time;
        const char *name;
        uint64_t thread;
    };

    struct Thread {
        uint64_t thread_id = 0;
        Vec<CpuTimeStamp> time_stamps;
    };

    Vec<CpuTimeStamp> cpu_time_stamps;
    Thread threads[MAX_THREAD_COUNT];
    std::mutex mutex;

    uint64_t thread_id;
    uint64_t frame_start;
    uint64_t frame_end;
    int registered_thread_count = 1;
};

static ProfilerImpl *impl;
static thread_local int tls_thread_slot = -1;

// Finds (or registers) the calling thread's slot in impl->threads. Slot 0 is
// reserved for the thread that called profiler_init (the main thread), so
// that the default `thread = 0` argument of profiler_resolve_frame works
// without callers having to know their thread's slot index.
static int get_thread_slot() {
    if (tls_thread_slot != -1) {
        return tls_thread_slot;
    }
    uint64_t const tid = GetCurrentThreadId();
    std::lock_guard<std::mutex> lock(impl->mutex);
    for (int i = 0; i < ProfilerImpl::MAX_THREAD_COUNT; ++i) {
        if (impl->threads[i].thread_id == tid) {
            tls_thread_slot = i;
            return tls_thread_slot;
        }
    }
    for (int i = 0; i < ProfilerImpl::MAX_THREAD_COUNT; ++i) {
        if (impl->threads[i].thread_id == 0) {
            impl->threads[i].thread_id = tid;
            if (i + 1 > impl->registered_thread_count) {
                impl->registered_thread_count = i + 1;
            }
            tls_thread_slot = i;
            return tls_thread_slot;
        }
    }
    // Ran out of thread slots; fall back to the last slot so calls don't crash.
    tls_thread_slot = ProfilerImpl::MAX_THREAD_COUNT - 1;
    return tls_thread_slot;
}

void profiler_init() {
    impl = new ProfilerImpl();
    impl->thread_id = GetCurrentThreadId();
    impl->threads[0].thread_id = impl->thread_id;
}

void profiler_shutdown() {
    delete impl;
}

uint64_t profiler_get_thread_count() {
    std::lock_guard<std::mutex> lock(impl->mutex);
    return static_cast<uint64_t>(impl->registered_thread_count);
}

void profiler_begin_frame() {
    std::lock_guard<std::mutex> lock(impl->mutex);
    impl->frame_start = get_timestamp();
    for (auto &thread : impl->threads) {
        thread.time_stamps.clear();
    }
}

void profiler_end_frame() {
    impl->frame_end = get_timestamp();
}

void profiler_enter_cpu(const char *name) {
    int const slot = get_thread_slot();
    impl->threads[slot].time_stamps.push_back({get_timestamp(), name, static_cast<uint64_t>(slot)});
}

void profiler_leave_cpu() {
    int const slot = get_thread_slot();
    impl->threads[slot].time_stamps.push_back({get_timestamp(), nullptr, static_cast<uint64_t>(slot)});
}

void profiler_resolve_frame(Vec<ProfileTimestamp> &timestamps, uint64_t thread) {
    timestamps.clear();
    if (thread >= static_cast<uint64_t>(ProfilerImpl::MAX_THREAD_COUNT)) {
        return;
    }

    static float const frequency = get_timestamp_frequency();
    uint64_t const frame_start = impl->frame_start;
    auto to_ms = [frame_start](uint64_t time) {
        return static_cast<float>(time - frame_start) / frequency * 1000.0f;
    };

    // Stack of the timestamp lists currently being appended to, one per
    // nesting depth. An "enter" appends a new node to the list on top of the
    // stack and pushes that node's own children list; a "leave" pops back to
    // the parent and stamps the closing time on the node that was just closed.
    Vec<Vec<ProfileTimestamp> *> stack;
    stack.push_back(&timestamps);

    auto const &time_stamps = impl->threads[thread].time_stamps;
    for (int i = 0; i < time_stamps.size; ++i) {
        auto const &stamp = time_stamps[i];
        if (stamp.name != nullptr) {
            ProfileTimestamp node;
            node.name = stamp.name;
            node.start = to_ms(stamp.time);
            node.end = node.start;
            stack.back()->push_back(node);
            stack.push_back(&stack.back()->back().children);
        } else if (stack.size > 1) {
            stack.pop_back();
            stack.back()->back().end = to_ms(stamp.time);
        }
    }
}
