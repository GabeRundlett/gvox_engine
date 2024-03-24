#pragma once

#include <application/settings.inl>

#if defined(__cplusplus)
#include <utilities/debug.hpp>
#include <utilities/gpu_context.hpp>
#include <utilities/math.hpp>
#define CPU_ONLY(x) x
#define GPU_ONLY(x)
#else
#define CPU_ONLY(x)
#define GPU_ONLY(x) x
#endif
