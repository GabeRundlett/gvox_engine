#pragma once

// daxa_u32vec2 get_downscale_offset(daxa_BufferPtr(GpuInput) gpu_input) {
// #if SHADING_SCL == 1
//     return daxa_u32vec2(0);
// #elif SHADING_SCL == 2
//     daxa_u32vec2 offsets[4] = daxa_u32vec2[4](
//         daxa_u32vec2(0, 0),
//         daxa_u32vec2(0, 1),
//         daxa_u32vec2(1, 0),
//         daxa_u32vec2(1, 1));
//     return offsets[deref(gpu_input).frame_index % 4];
// #else
// #error "Unsupported SHADING_SCL"
// #endif
//     return daxa_u32vec2(0);
// }

const uvec2 hi_px_subpixels[4] = uvec2[4](
    uvec2(1, 1),
    uvec2(1, 0),
    uvec2(0, 0),
    uvec2(0, 1));

#define USE_HALFRES_SUBSAMPLE_JITTERING 1

#if USE_HALFRES_SUBSAMPLE_JITTERING
#define HALFRES_SUBSAMPLE_INDEX (deref(gpu_input).frame_index & 3)
#else
#define HALFRES_SUBSAMPLE_INDEX 0
#endif

#define HALFRES_SUBSAMPLE_OFFSET (hi_px_subpixels[HALFRES_SUBSAMPLE_INDEX])
