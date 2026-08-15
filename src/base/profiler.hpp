#pragma once

#include "vec.hpp"
#include <cstdint>

struct ProfileTimestamp {
    const char *name;
    float start;
    float end;
    Vec<ProfileTimestamp> children;
};

void profiler_init();
void profiler_shutdown();

void profiler_begin_frame();
void profiler_end_frame();

void profiler_enter_cpu(const char *name);
void profiler_leave_cpu();

void profiler_resolve_frame(Vec<ProfileTimestamp> &timestamps, uint64_t thread = 0);

uint64_t profiler_get_thread_count();

class ProfileScopeCpu {
  public:
    inline ProfileScopeCpu(const char *name) { profiler_enter_cpu(name); }
    inline ~ProfileScopeCpu() { profiler_leave_cpu(); }
};

#define PROFILE_SCOPE(name) ProfileScopeCpu _profile_scope(name)
#define PROFILE_FUNC() ProfileScopeCpu _profile_scope_func(__FUNCTION__)
