#include <shared/core.inl>

u32vec2 get_downscale_offset(daxa_BufferPtr(GpuInput) gpu_input) {
#if SHADING_SCL == 1
    return u32vec2(0);
#elif SHADING_SCL == 2
    u32vec2 offsets[4] = u32vec2[4](
        u32vec2(0, 0),
        u32vec2(0, 1),
        u32vec2(1, 0),
        u32vec2(1, 1));
    return offsets[deref(gpu_input).frame_index % 4];
#else
#error "Unsupported SHADING_SCL"
#endif
    return u32vec2(0);
}
