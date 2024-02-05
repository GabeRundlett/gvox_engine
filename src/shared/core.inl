#pragma once

#include <shared/settings.inl>

#if defined(__cplusplus)
#include <cpu/core.hpp>
#include <cpu/app_ui.hpp>
#define CPU_ONLY(x) x
#define GPU_ONLY(x)
#else
#define CPU_ONLY(x)
#define GPU_ONLY(x) x
#endif
