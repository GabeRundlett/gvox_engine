#include <shared/shared.inl>

// #include <virtual/postprocessing>
#define FILMIC

DAXA_USE_PUSH_CONSTANT(PostprocessingComputePush)

f32vec3 srgb_encode(f32vec3 x) {
    return mix(12.92 * x, 1.055 * pow(x, f32vec3(.41666)) - .055, step(.0031308, x));
}

f32vec3 aces_filmic(f32vec3 x) {
    f32 a = 2.51;
    f32 b = 0.03;
    f32 c = 2.43;
    f32 d = 0.59;
    f32 e = 0.14;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

f32vec3 color_correct(f32vec3 x) {
#if defined(FILMIC)
    x = max(x, f32vec3(0, 0, 0));
    x = x * 0.25;
    x = aces_filmic(x);
    x = srgb_encode(x);
#endif
    return x;
}

#define INPUT deref(daxa_push_constant.gpu_input)
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 pixel_i = gl_GlobalInvocationID.xy;
    if (pixel_i.x >= INPUT.frame_dim.x ||
        pixel_i.y >= INPUT.frame_dim.y)
        return;

    f32vec4 denoised_color = imageLoad(
        daxa_push_constant.render_col_image_id,
        i32vec2(pixel_i));

    f32vec3 col = denoised_color.rgb;

    // const f32 BAR_SIZE = 10;
    // if (pixel_i.y > INPUT.frame_dim.y - BAR_SIZE * 4) {
    //     col = f32vec3(0.05);
    //     f32 sx = INPUT.frame_dim.x - 8;
    //     f32 x = pixel_i.x - BAR_SIZE;
    //     x = x / sx;
    //     f32 max_size = (deref(daxa_push_constant.gpu_settings).gpu_heap_size / 4);
    //     f32 raw_size = f32(1u << 27) / max_size;
    //     f32 consumed = f32(deref(daxa_push_constant.gpu_allocator_state).offset) / max_size;
    //     if (raw_size > x - 1.0 / sx && raw_size < x + 1.0 / sx) {
    //         col = f32vec3(0.3, 0.1, 0.1);
    //     }
    //     if (pixel_i.x > BAR_SIZE &&
    //         pixel_i.x < INPUT.frame_dim.x - BAR_SIZE &&
    //         pixel_i.y > INPUT.frame_dim.y - BAR_SIZE * 3 &&
    //         pixel_i.y < INPUT.frame_dim.y - BAR_SIZE) {
    //         if (consumed < x) {
    //             col = f32vec3(0.02);
    //         } else {
    //             col = f32vec3(0, 1, 0);
    //         }
    //     }
    // }

    imageStore(
        daxa_push_constant.final_image_id,
        i32vec2(pixel_i),
        f32vec4(color_correct(col), 0));
}
#undef INPUT
