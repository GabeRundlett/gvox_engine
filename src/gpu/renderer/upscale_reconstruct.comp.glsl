#include <shared/shared.inl>
#include <utils/downscale.glsl>
#include <utils/trace.glsl>

f32vec4 get_prev_val(i32vec2 pixel_i) {
    return texelFetch(daxa_texture2D(src_image_id), pixel_i, 0);
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 offset = get_downscale_offset(gpu_input);
    u32vec2 rounded_frame_dim = deref(gpu_input).rounded_frame_dim;
    u32vec2 half_frame_dim = rounded_frame_dim / 2;
    u32vec2 tile = u32vec2(gl_GlobalInvocationID.x >= half_frame_dim.x, gl_GlobalInvocationID.y >= half_frame_dim.y);
    u32vec2 in_tile_i = gl_GlobalInvocationID.xy - tile * half_frame_dim;
    i32vec2 output_i = i32vec2(in_tile_i * SHADING_SCL + tile);

    f32vec4 result = f32vec4(0.0);

    if (deref(gpu_input).resize_factor != 0.0) {
        f32vec4 reproj_val = texelFetch(daxa_texture2D(reprojection_image_id), output_i, 0);
        f32vec4 prev_val = get_prev_val(output_i + i32vec2(reproj_val.xy * 0.0 * deref(gpu_input).frame_dim));
        f32 validity = reproj_val.z;
        if (offset == tile) {
            f32 ssao_value = 0.0; // texelFetch(daxa_texture2D(ssao_image_id), i32vec2(in_tile_i), 0).r;
            f32vec3 direct_value = texelFetch(daxa_texture2D(scaled_shading_image), i32vec2(in_tile_i), 0).rgb;
            result = f32vec4(direct_value, ssao_value);
            // result = mix(prev_val, f32vec4(direct_value, ssao_value), 0.05);
        } else {
            result = prev_val;
        }
    }

    imageStore(daxa_image2D(dst_image_id), output_i, result);
}
