#include <shared/shared.inl>

#define FILMIC

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

#define INPUT deref(gpu_input)
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 pixel_i = gl_GlobalInvocationID.xy;
    if (pixel_i.x >= INPUT.frame_dim.x ||
        pixel_i.y >= INPUT.frame_dim.y)
        return;

    f32vec4 final_color = imageLoad(daxa_image2D(render_col_image_id), i32vec2(pixel_i));

    imageStore(daxa_image2D(final_image_id), i32vec2(pixel_i), f32vec4(color_correct(final_color.rgb), 0));
}
#undef INPUT
