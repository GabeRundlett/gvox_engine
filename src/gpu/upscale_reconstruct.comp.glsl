#include <shared/shared.inl>
#include <utils/downscale.glsl>
#include <utils/trace.glsl>

f32vec2 prev_uv_from_pos(in out Player player, f32 aspect, f32 fov, f32vec3 pos) {
    f32vec3 dir = normalize(pos - player.cam.prev_pos) * player.cam.prev_rot_mat;
    return dir.xy / dir.z * f32vec2(0.5 / aspect, 0.5) / player.cam.prev_tan_half_fov;
}

f32vec3 view_pos;
f32vec2 frame_dim;
f32vec2 inv_frame_dim;
f32 aspect;

f32vec4 get_prev_sample(i32vec2 prev_pixel_i, f32vec3 hit) {
    if (min(i32vec2(frame_dim), prev_pixel_i) != prev_pixel_i || max(i32vec2(0, 0), prev_pixel_i) != prev_pixel_i)
        return f32vec4(f32vec3(0), MAX_SD);

    f32 prev_depth = texelFetch(daxa_texture2D(depth_image_id), i32vec2(prev_pixel_i), 0).r;
    f32vec2 pixel_p = f32vec2(prev_pixel_i) + 0.5;
    f32vec2 uv = pixel_p * inv_frame_dim;
    f32vec3 view_dir = create_view_dir(deref(globals).player, (uv * 2.0 - 1.0) * vec2(aspect, 1.0));
    f32vec3 prev_hit_pos = view_dir * prev_depth + view_pos;

    f32vec3 p0 = prev_hit_pos;
    f32vec3 p1 = hit;
    f32vec3 del = p0 - p1;
    return f32vec4(prev_hit_pos, dot(del, del));
}

f32vec4 get_prev_val(u32vec2 pixel_i) {
    // Needs to reproject!

    return texelFetch(daxa_texture2D(src_image_id), i32vec2(pixel_i), 0);

    // f32vec2 pixel_p = f32vec2(pixel_i) + 0.5;
    // frame_dim = deref(gpu_input).frame_dim;
    // inv_frame_dim = f32vec2(1.0, 1.0) / frame_dim;
    // aspect = frame_dim.x * inv_frame_dim.y;
    // f32vec2 uv = pixel_p * inv_frame_dim;

    // f32 prev_depth = texelFetch(daxa_texture2D(depth_image_id), i32vec2(pixel_i), 0).r;
    // view_pos = create_view_pos(deref(globals).player);
    // f32vec3 view_dir = create_view_dir(deref(globals).player, (uv * 2.0 - 1.0) * vec2(aspect, 1.0));
    // f32vec3 hit_pos = view_dir * prev_depth + view_pos;
    // f32vec2 prev_uv = prev_uv_from_pos(deref(globals).player, aspect, deref(settings).fov, hit_pos) + 0.5;

    // return texelFetch(daxa_texture2D(src_image_id), i32vec2(prev_uv * frame_dim), 0);

    // i32vec2 pfc, finalpfc;
    // f32 finaldist = MAX_SD;
    // f32vec3 final_pos = f32vec3(0);
    // const i32 SEARCH_RADIUS = 0;
    // i32vec2 prev_pixel_i = i32vec2(round((prev_uv + 0.5) * frame_dim));

    // f32 hit_dist = prev_depth;
    // hit_dist = ceil(pow(hit_dist * 0.1, 1) / deref(settings).fov * PI / 2);

    // f32vec4 blurred_color = f32vec4(0.0);
    // f32 blurred_samples = 0.0;

    // for (i32 x = -SEARCH_RADIUS; x <= SEARCH_RADIUS; x++) {
    //     for (i32 y = -SEARCH_RADIUS; y <= SEARCH_RADIUS; y++) {
    //         pfc = prev_pixel_i + i32vec2(x, y);
    //         f32vec4 prev_sample = get_prev_sample(pfc, hit_pos);
    //         f32 dist = prev_sample.w;
    //         f32vec3 prev_pos = prev_sample.xyz;
    //         if (dist < finaldist) {
    //             finalpfc = pfc;
    //             finaldist = dist;
    //         }
    //         bool positions_equal = true;
    //         bool is_close_enough = true;
    //         if (is_close_enough && positions_equal) {
    //             f32vec4 prev_val = texelFetch(daxa_texture2D(src_image_id), pfc, 0);
    //             blurred_color += prev_val;
    //             blurred_samples += 1.0;
    //         }
    //     }
    // }

    // return blurred_color / blurred_samples;
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 offset = get_downscale_offset(gpu_input);
    u32vec2 rounded_frame_dim = deref(gpu_input).rounded_frame_dim;
    u32vec2 half_frame_dim = rounded_frame_dim / 2;
    u32vec2 tile = u32vec2(gl_GlobalInvocationID.x >= half_frame_dim.x, gl_GlobalInvocationID.y >= half_frame_dim.y);
    u32vec2 in_tile_i = gl_GlobalInvocationID.xy - tile * half_frame_dim;
    u32vec2 output_i = in_tile_i * SHADING_SCL + tile;

    f32vec4 result = f32vec4(0.0);

    f32vec4 prev_val = get_prev_val(output_i);

    if (offset == tile) {
        f32 ssao_value = texelFetch(daxa_texture2D(ssao_image_id), i32vec2(in_tile_i), 0).r;
        f32vec3 direct_value = texelFetch(daxa_texture2D(indirect_diffuse_image_id), i32vec2(in_tile_i), 0).rgb;
        result = f32vec4(direct_value, ssao_value);
        // result = mix(f32vec4(direct_value, ssao_value), prev_val, 0.05);
    } else {
        if (deref(gpu_input).resize_factor != 0.0) {
            result = prev_val;
        }
    }

    // result.rgb = f32vec3(prev_uv, 0.0);
    // result.rgb = fract(frag_pos);

    imageStore(daxa_image2D(dst_image_id), i32vec2(output_i), result);
}
