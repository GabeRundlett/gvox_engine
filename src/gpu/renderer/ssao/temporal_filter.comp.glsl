#include <shared/shared.inl>
#include <utils/math.glsl>

DAXA_DECL_PUSH_CONSTANT(SsaoTemporalFilterComputePush, push)

float fetch_src(u32vec2 px) {
    return texelFetch(daxa_texture2D(src_image_id), i32vec2(px), 0).r;
}

#define LINEAR_TO_WORKING(x) x
#define WORKING_TO_LINEAR(x) x

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 px = gl_GlobalInvocationID.xy;
    f32vec4 output_tex_size;
    output_tex_size.xy = deref(gpu_input).rounded_frame_dim;
    output_tex_size.zw = f32vec2(1.0, 1.0) / output_tex_size.xy;
    f32vec2 uv = get_uv(px, output_tex_size);

    float center = WORKING_TO_LINEAR(fetch_src(px));
    f32vec4 reproj = texelFetch(daxa_texture2D(reprojection_image_id), i32vec2(px), 0);
    float history = WORKING_TO_LINEAR(textureLod(daxa_sampler2D(history_image_id, push.history_sampler), uv + reproj.xy, 0).r);

    float vsum = 0.0;
    float vsum2 = 0.0;
    float wsum = 0.0;

    const int k = 1;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float neigh = WORKING_TO_LINEAR(fetch_src(px + i32vec2(x, y) * 2));
            float w = exp(-3.0 * float(x * x + y * y) / float((k + 1.) * (k + 1.)));
            vsum += neigh * w;
            vsum2 += neigh * neigh * w;
            wsum += w;
        }
    }

    float ex = vsum / wsum;
    float ex2 = vsum2 / wsum;
    float dev = sqrt(max(0.0, ex2 - ex * ex));

    float box_size = 0.5;

    const float n_deviations = 5.0;
    float nmin = mix(center, ex, box_size * box_size) - dev * box_size * n_deviations;
    float nmax = mix(center, ex, box_size * box_size) + dev * box_size * n_deviations;

    float clamped_history = clamp(history, nmin, nmax);
    float res = mix(clamped_history, center, 1.0 / 8.0);

    // res = center;

    // history_output_tex[px] = LINEAR_TO_WORKING(res);
    imageStore(daxa_image2D(dst_image_id), i32vec2(px), f32vec4(res));
}
