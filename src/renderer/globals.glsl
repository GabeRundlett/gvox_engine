#ifndef RENDERER_GLOBALS_GLSL
#define RENDERER_GLOBALS_GLSL
#include <daxa/daxa.glsl>

daxa_SamplerId g_sampler_nnc = daxa_SamplerId(2097152);
daxa_SamplerId g_sampler_lnc = daxa_SamplerId(2097153);
daxa_SamplerId g_sampler_llc = daxa_SamplerId(2097154);
daxa_SamplerId g_sampler_llr = daxa_SamplerId(2097155);

daxa_ImageViewIndex g_value_noise_tex = daxa_ImageViewIndex(1);

#endif // RENDERER_GLOBALS_GLSL
