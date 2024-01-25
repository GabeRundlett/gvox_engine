#include <shared/app.inl>

#include <utils/math.glsl>
#include <utils/safety.glsl>

#if SsaoComputeShader

#include <voxels/core.glsl>
#include <utils/downscale.glsl>

daxa_f32vec4 output_tex_size;

#define USE_SSGI_FACING_CORRECTION 1
#define USE_AO_ONLY 1

#define WORLDSPACE_SSAO 0

#if WORLDSPACE_SSAO
#define SSGI_KERNEL_RADIUS 5
#define MAX_KERNEL_RADIUS_CS 100
const uint SSGI_HALF_SAMPLE_COUNT = 6;
#else
#define SSGI_KERNEL_RADIUS (60.0 * output_tex_size.w)
#define MAX_KERNEL_RADIUS_CS 0.4
const uint SSGI_HALF_SAMPLE_COUNT = 6;
#endif

const float temporal_rotations[] = {60.0, 300.0, 180.0, 240.0, 120.0, 0.0};
const float temporal_offsets[] = {0.0, 0.5, 0.25, 0.75};

float fetch_depth(daxa_u32vec2 px) {
    return safeTexelFetch(depth_image_id, daxa_i32vec2(px), 0).r;
}

daxa_f32vec3 fetch_normal_vs(daxa_f32vec2 uv) {
    daxa_i32vec2 px = daxa_i32vec2(output_tex_size.xy * uv);
    daxa_f32vec3 normal_vs = safeTexelFetch(vs_normal_image_id, px, 0).xyz;
    return normal_vs;
}

float integrate_half_arc(float h1, float n) {
    float a = -cos(2.0 * h1 - n) + cos(n) + 2.0 * h1 * sin(n);
    return 0.25 * a;
}

float integrate_arc(float h1, float h2, float n) {
    float a = -cos(2.0 * h1 - n) + cos(n) + 2.0 * h1 * sin(n);
    float b = -cos(2.0 * h2 - n) + cos(n) + 2.0 * h2 * sin(n);
    return 0.25 * (a + b);
}

float update_horizion_angle(float prev, float cur, float blend) {
    return cur > prev ? mix(prev, cur, blend) : prev;
}

float intersect_dir_plane_onesided(daxa_f32vec3 dir, daxa_f32vec3 normal, daxa_f32vec3 pt) {
    float d = -dot(pt, normal);
    float t = d / max(1e-5, -dot(dir, normal));
    return t;
}

daxa_f32vec3 project_point_on_plane(daxa_f32vec3 pt, daxa_f32vec3 normal) {
    return pt - normal * dot(pt, normal);
}

float process_sample(uint i, float intsgn, float n_angle, inout daxa_f32vec3 prev_sample_vs, daxa_f32vec4 sample_cs, daxa_f32vec3 center_vs, daxa_f32vec3 normal_vs, daxa_f32vec3 v_vs, float kernel_radius_ws, float theta_cos_max) {
    if (sample_cs.z > 0) {
        daxa_f32vec4 sample_vs4 = (deref(globals).player.cam.sample_to_view * sample_cs);
        daxa_f32vec3 sample_vs = sample_vs4.xyz / sample_vs4.w;
        daxa_f32vec3 sample_vs_offset = sample_vs - center_vs;
        float sample_vs_offset_len = length(sample_vs_offset);

        float sample_theta_cos = dot(sample_vs_offset, v_vs) / sample_vs_offset_len;
        const float sample_distance_normalized = sample_vs_offset_len / kernel_radius_ws;

        if (sample_distance_normalized < 1.0) {
            const float sample_influence = smoothstep(1, 0, sample_distance_normalized);
            theta_cos_max = update_horizion_angle(theta_cos_max, sample_theta_cos, sample_influence);
        }

        prev_sample_vs = sample_vs;
    } else {
        // Sky; assume no occlusion
        theta_cos_max = update_horizion_angle(theta_cos_max, -1, 1);
    }

    return theta_cos_max;
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    daxa_u32vec2 px = gl_GlobalInvocationID.xy;
    output_tex_size.xy = deref(gpu_input).frame_dim;
    output_tex_size.zw = daxa_f32vec2(1.0, 1.0) / output_tex_size.xy;
    daxa_u32vec2 offset = get_downscale_offset(gpu_input);
    daxa_f32vec2 uv = get_uv(px * SHADING_SCL + offset, output_tex_size);
    output_tex_size *= vec4((1.0 / SHADING_SCL).xx, SHADING_SCL.xx);

    daxa_f32 depth = fetch_depth(px);
    daxa_f32vec3 normal_vs = safeTexelFetch(vs_normal_image_id, daxa_i32vec2(px), 0).xyz;

    if (depth == 0.0 || dot(normal_vs, normal_vs) == 0.0) {
        safeImageStore(ssao_image_id, daxa_i32vec2(px), daxa_f32vec4(1, 0, 0, 0));
        return;
    }

    const ViewRayContext view_ray_context = vrc_from_uv_and_depth(globals, uv, depth);
    daxa_f32vec3 v_vs = -normalize(ray_dir_vs(view_ray_context));

    daxa_f32vec4 ray_hit_cs = view_ray_context.ray_hit_cs;
    daxa_f32vec3 ray_hit_vs = ray_hit_vs(view_ray_context);

    float spatial_direction_noise = 1.0 / 16.0 * ((((px.x + px.y) & 3) << 2) + (px.x & 3));
    float temporal_direction_noise = temporal_rotations[deref(gpu_input).frame_index % 6] / 360.0;
    float spatial_offset_noise = (1.0 / 4.0) * ((px.y - px.x) & 3);
    float temporal_offset_noise = temporal_offsets[deref(gpu_input).frame_index / 6 % 4];

    float ss_angle = fract(spatial_direction_noise + temporal_direction_noise) * PI;
    float rand_offset = fract(spatial_offset_noise + temporal_offset_noise);

    daxa_f32vec2 cs_slice_dir = daxa_f32vec2(cos(ss_angle) * daxa_f32(deref(gpu_input).frame_dim.y) / daxa_f32(deref(gpu_input).frame_dim.x), sin(ss_angle));

    float kernel_radius_ws;
    float kernel_radius_shrinkage = 1;
    {
        const float ws_to_cs = 0.5 / -ray_hit_vs.z * deref(globals).player.cam.view_to_clip[1][1];

#if WORLDSPACE_SSAO
        kernel_radius_ws = SSGI_KERNEL_RADIUS;
        const float cs_kernel_radius_scaled = kernel_radius_ws * ws_to_cs;
#else
        const float cs_kernel_radius_scaled = SSGI_KERNEL_RADIUS;
        kernel_radius_ws = cs_kernel_radius_scaled / ws_to_cs;
#endif

        cs_slice_dir *= cs_kernel_radius_scaled;

        // Calculate AO radius shrinkage (if camera is too close to a surface)
        float max_kernel_radius_cs = MAX_KERNEL_RADIUS_CS;

        // float max_kernel_radius_cs = 100;
        kernel_radius_shrinkage = min(1.0, max_kernel_radius_cs / cs_kernel_radius_scaled);
    }

    // Shrink the AO radius
    cs_slice_dir *= kernel_radius_shrinkage;
    kernel_radius_ws *= kernel_radius_shrinkage;

    daxa_f32vec3 center_vs = ray_hit_vs.xyz;

    cs_slice_dir *= 1.0 / float(SSGI_HALF_SAMPLE_COUNT);
    daxa_f32vec2 vs_slice_dir = (daxa_f32vec4(cs_slice_dir, 0, 0) * deref(globals).player.cam.sample_to_view).xy;
    daxa_f32vec3 slice_normal_vs = normalize(cross(v_vs, daxa_f32vec3(vs_slice_dir, 0)));

    daxa_f32vec3 proj_normal_vs = normal_vs - slice_normal_vs * dot(slice_normal_vs, normal_vs);
    float slice_contrib_weight = length(proj_normal_vs);
    proj_normal_vs /= slice_contrib_weight;

    float n_angle = fast_acos(clamp(dot(proj_normal_vs, v_vs), -1.0, 1.0)) * sign(dot(vs_slice_dir, proj_normal_vs.xy - v_vs.xy));

    float theta_cos_max1 = cos(n_angle - PI * 0.5);
    float theta_cos_max2 = cos(n_angle + PI * 0.5);

    daxa_f32vec3 prev_sample0_vs = v_vs;
    daxa_f32vec3 prev_sample1_vs = v_vs;

    daxa_i32vec2 prev_sample_coord0 = daxa_i32vec2(px);
    daxa_i32vec2 prev_sample_coord1 = daxa_i32vec2(px);

    for (uint i = 0; i < SSGI_HALF_SAMPLE_COUNT; ++i) {
        {
            float t = float(i) + rand_offset;

            daxa_f32vec4 sample_cs = daxa_f32vec4(ray_hit_cs.xy - cs_slice_dir * t, 0, 1);
            daxa_i32vec2 sample_px = daxa_i32vec2(output_tex_size.xy * cs_to_uv(sample_cs.xy));

            if (any(bvec2(sample_px != prev_sample_coord0))) {
                prev_sample_coord0 = sample_px;
                sample_cs.z = fetch_depth(sample_px);
                theta_cos_max1 = process_sample(i, 1, n_angle, prev_sample0_vs, sample_cs, center_vs, normal_vs, v_vs, kernel_radius_ws, theta_cos_max1);
            }
        }

        {
            float t = float(i) + (1.0 - rand_offset);

            daxa_f32vec4 sample_cs = daxa_f32vec4(ray_hit_cs.xy + cs_slice_dir * t, 0, 1);
            daxa_i32vec2 sample_px = daxa_i32vec2(output_tex_size.xy * cs_to_uv(sample_cs.xy));

            if (any(bvec2(sample_px != prev_sample_coord1))) {
                prev_sample_coord1 = sample_px;
                sample_cs.z = fetch_depth(sample_px);
                theta_cos_max2 = process_sample(i, -1, n_angle, prev_sample1_vs, sample_cs, center_vs, normal_vs, v_vs, kernel_radius_ws, theta_cos_max2);
            }
        }
    }

    float h1 = -fast_acos(theta_cos_max1);
    float h2 = +fast_acos(theta_cos_max2);

    float h1p = n_angle + max(h1 - n_angle, -PI * 0.5);
    float h2p = n_angle + min(h2 - n_angle, PI * 0.5);

    float inv_ao = integrate_arc(h1p, h2p, n_angle);
    daxa_f32vec4 col;
    col.a = max(0.0, inv_ao);
    col.rgb = col.aaa;

    col *= slice_contrib_weight;

    safeImageStore(ssao_image_id, daxa_i32vec2(px), daxa_f32vec4(col));
}

#endif
#if SsaoSpatialFilterComputeShader

float fetch_src(daxa_u32vec2 px) {
    return safeTexelFetch(src_image_id, daxa_i32vec2(px), 0).r;
}
float fetch_depth(daxa_u32vec2 px) {
    return safeTexelFetch(depth_image_id, daxa_i32vec2(px), 0).r;
}
daxa_f32vec3 fetch_nrm(daxa_u32vec2 px) {
    return safeTexelFetch(vs_normal_image_id, daxa_i32vec2(px), 0).xyz;
}

float process_sample(float ssgi, float depth, daxa_f32vec3 normal, float center_depth, daxa_f32vec3 center_normal, inout float w_sum) {
    if (depth != 0.0) {
        float depth_diff = 1.0 - (center_depth / depth);
        float depth_factor = exp2(-200.0 * abs(depth_diff));

        float normal_factor = max(0.0, dot(normal, center_normal));
        normal_factor *= normal_factor;
        normal_factor *= normal_factor;

        float w = 1;
        w *= depth_factor;
        w *= normal_factor;

        w_sum += w;
        return ssgi * w;
    } else {
        return 0.0;
    }
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    daxa_u32vec2 px = gl_GlobalInvocationID.xy;

    float result = 0.0;
    float w_sum = 0.0;

#if 0
    result = fetch_src(px).r;
    w_sum = 1.0;
#else
    float center_depth = fetch_depth(px).x;
    if (center_depth != 0.0) {
        daxa_f32vec3 center_normal = fetch_nrm(px).xyz;

        float center_ssgi = fetch_src(px).r;
        w_sum = 1.0;
        result = center_ssgi;

        const int kernel_half_size = 1;
        for (int y = -kernel_half_size; y <= kernel_half_size; ++y) {
            for (int x = -kernel_half_size; x <= kernel_half_size; ++x) {
                if (x != 0 || y != 0) {
                    daxa_i32vec2 sample_px = daxa_i32vec2(px) + daxa_i32vec2(x, y);
                    float depth = fetch_depth(sample_px).x;
                    float ssgi = fetch_src(sample_px).r;
                    daxa_f32vec3 normal = fetch_nrm(sample_px).xyz;
                    result += process_sample(ssgi, depth, normal, center_depth, center_normal, w_sum);
                }
            }
        }
    } else {
        result = 0.0;
    }
#endif

    safeImageStore(dst_image_id, daxa_i32vec2(px), daxa_f32vec4(result / max(w_sum, 1e-5), 0, 0, 0));
}

#endif
#if SsaoUpscaleComputeShader

float fetch_src(daxa_u32vec2 px) {
    return safeTexelFetch(src_image_id, daxa_i32vec2(px), 0).r;
}
float fetch_depth(daxa_u32vec2 px) {
    return safeTexelFetch(depth_image_id, daxa_i32vec2(px), 0).r;
}
daxa_f32vec3 fetch_nrm(daxa_u32vec2 px) {
    daxa_u32vec4 g_buffer_value = safeTexelFetchU(g_buffer_image_id, daxa_i32vec2(px), 0);
    return u16_to_nrm(g_buffer_value.y);
}

float process_sample(daxa_f32vec2 soffset, float ssgi, float depth, daxa_f32vec3 normal, float center_depth, daxa_f32vec3 center_normal, inout float w_sum) {
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
    daxa_u32vec2 px = gl_GlobalInvocationID.xy;
    float result = 0.0;
    float w_sum = 0.0;

    float center_depth = fetch_depth(px);
    if (center_depth != 0.0) {
        daxa_f32vec3 center_normal = fetch_nrm(px);

        float center_ssgi = 0.0;
        w_sum = 0.0;
        result = center_ssgi;

        const int kernel_half_size = 1;
        for (int y = -kernel_half_size; y <= kernel_half_size; ++y) {
            for (int x = -kernel_half_size; x <= kernel_half_size; ++x) {
                daxa_i32vec2 sample_pix = daxa_i32vec2(px / SHADING_SCL) + daxa_i32vec2(x, y);
                float depth = fetch_depth(sample_pix * SHADING_SCL);
                float ssgi = fetch_src(sample_pix);
                daxa_f32vec3 normal = fetch_nrm(sample_pix * SHADING_SCL);
                result += process_sample(daxa_f32vec2(x, y), ssgi, depth, normal, center_depth, center_normal, w_sum);
            }
        }
    } else {
        result = 0.0;
    }

    if (w_sum > 1e-6) {
        safeImageStore(dst_image_id, daxa_i32vec2(px), daxa_f32vec4(result / w_sum, 0, 0, 0));
    } else {
        safeImageStore(dst_image_id, daxa_i32vec2(px), daxa_f32vec4(fetch_src(px / SHADING_SCL), 0, 0, 0));
    }
}

#endif
#if SsaoTemporalFilterComputeShader

float fetch_src(daxa_u32vec2 px) {
    return safeTexelFetch(src_image_id, daxa_i32vec2(px), 0).r;
}

#define LINEAR_TO_WORKING(x) x
#define WORKING_TO_LINEAR(x) x

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    daxa_u32vec2 px = gl_GlobalInvocationID.xy;
    daxa_f32vec4 output_tex_size;
    output_tex_size.xy = deref(gpu_input).rounded_frame_dim;
    output_tex_size.zw = daxa_f32vec2(1.0, 1.0) / output_tex_size.xy;
    daxa_f32vec2 uv = get_uv(px, output_tex_size);

    float center = WORKING_TO_LINEAR(fetch_src(px));
    daxa_f32vec4 reproj = safeTexelFetch(reprojection_image_id, daxa_i32vec2(px), 0);
    float history = WORKING_TO_LINEAR(textureLod(daxa_sampler2D(history_image_id, deref(gpu_input).sampler_lnc), uv + reproj.xy, 0).r);

    float vsum = 0.0;
    float vsum2 = 0.0;
    float wsum = 0.0;

    const int k = 1;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float neigh = WORKING_TO_LINEAR(fetch_src(px + daxa_i32vec2(x, y) * 2));
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
    safeImageStore(dst_image_id, daxa_i32vec2(px), daxa_f32vec4(res));
}

#endif
