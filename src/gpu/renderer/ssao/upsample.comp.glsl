#include <shared/shared.inl>
#include <utils/math.glsl>

float fetch_src(u32vec2 px) {
    return texelFetch(daxa_texture2D(src_image_id), i32vec2(px), 0).r;
}
float fetch_depth(u32vec2 px) {
    return texelFetch(daxa_texture2D(depth_image_id), i32vec2(px), 0).r;
}
f32vec3 fetch_nrm(u32vec2 px) {
    u32vec4 g_buffer_value = texelFetch(daxa_utexture2D(g_buffer_image_id), i32vec2(px), 0);
    return u16_to_nrm(g_buffer_value.y);
}

float process_sample(f32vec2 soffset, float ssgi, float depth, f32vec3 normal, float center_depth, f32vec3 center_normal, inout float w_sum) {
    if (depth != 0.0) {
        float depth_diff = 1.0 - (center_depth / depth);
        float depth_factor = exp2(-200.0 * abs(depth_diff));

        float normal_factor = max(0.0, dot(normal, center_normal));
        normal_factor *= normal_factor;
        normal_factor *= normal_factor;
        normal_factor *= normal_factor;

        float w = 1;
        w *= depth_factor; // TODO: differentials

        // Not super relevant when only using AO for a guide map.
        // w *= normal_factor;

        w *= exp(-dot(soffset, soffset));

        w_sum += w;
        return ssgi * w;
    } else {
        return 0.0;
    }
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 px = gl_GlobalInvocationID.xy;
    float result = 0.0;
    float w_sum = 0.0;

    float center_depth = fetch_depth(px);
    if (center_depth != 0.0) {
        f32vec3 center_normal = fetch_nrm(px);

        float center_ssgi = 0.0;
        w_sum = 0.0;
        result = center_ssgi;

        const int kernel_half_size = 1;
        for (int y = -kernel_half_size; y <= kernel_half_size; ++y) {
            for (int x = -kernel_half_size; x <= kernel_half_size; ++x) {
                i32vec2 sample_pix = i32vec2(px / 2) + i32vec2(x, y);
                float depth = fetch_depth(sample_pix * 2);
                float ssgi = fetch_src(sample_pix);
                f32vec3 normal = fetch_nrm(sample_pix * 2);
                result += process_sample(f32vec2(x, y), ssgi, depth, normal, center_depth, center_normal, w_sum);
            }
        }
    } else {
        result = 0.0;
    }

    if (w_sum > 1e-6) {
        imageStore(daxa_image2D(dst_image_id), i32vec2(px), f32vec4(result / w_sum, 0, 0, 0));
    } else {
        imageStore(daxa_image2D(dst_image_id), i32vec2(px), f32vec4(fetch_src(px / 2), 0, 0, 0));
    }
}
