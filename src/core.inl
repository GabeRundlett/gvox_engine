#ifndef CORE_INL
#define CORE_INL

#include <application/settings.inl>

#if defined(__cplusplus)
#include <utilities/debug.hpp>
#include <renderer/gpu_context.hpp>
#include <utilities/math.hpp>
#define CPU_ONLY(x) x
#define GPU_ONLY(x)
#else
#define CPU_ONLY(x)
#define GPU_ONLY(x) x
#endif

#endif // CORE_INL
