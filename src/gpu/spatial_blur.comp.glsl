#include <shared/shared.inl>

DAXA_USE_PUSH_CONSTANT(SpatialBlurComputePush)

#define INPUT deref(daxa_push_constant.gpu_input)
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 pixel_i = gl_GlobalInvocationID.xy;
    if (pixel_i.x >= INPUT.frame_dim.x ||
        pixel_i.y >= INPUT.frame_dim.y)
        return;

    const i32 SEARCH_RADIUS = 0;
    i32 contrib = 0;
    f32vec4 denoised_color;
    for (i32 yi = -SEARCH_RADIUS; yi <= SEARCH_RADIUS; ++yi) {
        for (i32 xi = -SEARCH_RADIUS; xi <= SEARCH_RADIUS; ++xi) {
            ++contrib;
            denoised_color += imageLoad(
                daxa_push_constant.render_col_image_id,
                i32vec2(pixel_i) + i32vec2(xi, yi));
        }
    }
    denoised_color *= 1.0 / f32(contrib);

    imageStore(
        daxa_push_constant.final_image_id,
        i32vec2(pixel_i),
        denoised_color);
}
#undef INPUT
