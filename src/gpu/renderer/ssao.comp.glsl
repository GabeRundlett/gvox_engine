#include <shared/shared.inl>

#include <utils/trace.glsl>
#include <utils/downscale.glsl>
#include <utils/math.glsl>

f32vec4 output_tex_size;

#define USE_SSGI_FACING_CORRECTION 1
#define USE_AO_ONLY 1

const uint SSGI_HALF_SAMPLE_COUNT = 6;
#define SSGI_KERNEL_RADIUS (60.0 * output_tex_size.w)
#define MAX_KERNEL_RADIUS_CS 0.4

const float temporal_rotations[] = {60.0, 300.0, 180.0, 240.0, 120.0, 0.0};
const float temporal_offsets[] = {0.0, 0.5, 0.25, 0.75};

float fetch_depth(u32vec2 px) {
    return texelFetch(daxa_texture2D(depth_image_id), i32vec2(px), 0).r;
}

f32vec3 fetch_normal_vs(f32vec2 uv) {
    i32vec2 px = i32vec2(output_tex_size.xy * uv);
    f32vec3 normal_vs = texelFetch(daxa_texture2D(vs_normal_image_id), px, 0).xyz;
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

float intersect_dir_plane_onesided(f32vec3 dir, f32vec3 normal, f32vec3 pt) {
    float d = -dot(pt, normal);
    float t = d / max(1e-5, -dot(dir, normal));
    return t;
}

f32vec3 project_point_on_plane(f32vec3 pt, f32vec3 normal) {
    return pt - normal * dot(pt, normal);
}

float process_sample(uint i, float intsgn, float n_angle, inout f32vec3 prev_sample_vs, f32vec4 sample_cs, f32vec3 center_vs, f32vec3 normal_vs, f32vec3 v_vs, float kernel_radius_ws, float theta_cos_max) {
    if (sample_cs.z > 0) {
        f32vec4 sample_vs4 = (deref(globals).player.cam.sample_to_view * sample_cs);
        f32vec3 sample_vs = sample_vs4.xyz / sample_vs4.w;
        f32vec3 sample_vs_offset = sample_vs - center_vs;
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
    u32vec2 px = gl_GlobalInvocationID.xy;
    output_tex_size.xy = deref(gpu_input).frame_dim;
    output_tex_size.zw = f32vec2(1.0, 1.0) / output_tex_size.xy;
    u32vec2 offset = get_downscale_offset(gpu_input);
    f32vec2 uv = get_uv(px * SHADING_SCL + offset, output_tex_size);
    output_tex_size *= vec4(0.5, 0.5, 2.0, 2.0);

    f32 depth = fetch_depth(px);
    f32vec3 normal_vs = texelFetch(daxa_texture2D(vs_normal_image_id), i32vec2(px), 0).xyz;

    if (depth == 0.0 || dot(normal_vs, normal_vs) == 0.0) {
        imageStore(daxa_image2D(ssao_image_id), i32vec2(px), f32vec4(1, 0, 0, 0));
        return;
    }

    const ViewRayContext view_ray_context = vrc_from_uv_and_depth(globals, uv, depth);
    f32vec3 v_vs = -normalize(ray_dir_vs(view_ray_context));

    f32vec4 ray_hit_cs = view_ray_context.ray_hit_cs;
    f32vec3 ray_hit_vs = ray_hit_vs(view_ray_context);

    float spatial_direction_noise = 1.0 / 16.0 * ((((px.x + px.y) & 3) << 2) + (px.x & 3));
    float temporal_direction_noise = temporal_rotations[deref(gpu_input).frame_index % 6] / 360.0;
    float spatial_offset_noise = (1.0 / 4.0) * ((px.y - px.x) & 3);
    float temporal_offset_noise = temporal_offsets[deref(gpu_input).frame_index / 6 % 4];

    float ss_angle = fract(spatial_direction_noise + temporal_direction_noise) * PI;
    float rand_offset = fract(spatial_offset_noise + temporal_offset_noise);

    f32vec2 cs_slice_dir = f32vec2(cos(ss_angle) * f32(deref(gpu_input).frame_dim.y) / f32(deref(gpu_input).frame_dim.x), sin(ss_angle));

    float kernel_radius_ws;
    float kernel_radius_shrinkage = 1;
    {
        const float ws_to_cs = 0.5 / -ray_hit_vs.z * deref(globals).player.cam.view_to_clip[1][1];

        const float cs_kernel_radius_scaled = SSGI_KERNEL_RADIUS;
        kernel_radius_ws = cs_kernel_radius_scaled / ws_to_cs;

        cs_slice_dir *= cs_kernel_radius_scaled;

        // Calculate AO radius shrinkage (if camera is too close to a surface)
        float max_kernel_radius_cs = MAX_KERNEL_RADIUS_CS;

        // float max_kernel_radius_cs = 100;
        kernel_radius_shrinkage = min(1.0, max_kernel_radius_cs / cs_kernel_radius_scaled);
    }

    // Shrink the AO radius
    cs_slice_dir *= kernel_radius_shrinkage;
    kernel_radius_ws *= kernel_radius_shrinkage;

    f32vec3 center_vs = ray_hit_vs.xyz;

    cs_slice_dir *= 1.0 / float(SSGI_HALF_SAMPLE_COUNT);
    f32vec2 vs_slice_dir = (f32vec4(cs_slice_dir, 0, 0) * deref(globals).player.cam.sample_to_view).xy;
    f32vec3 slice_normal_vs = normalize(cross(v_vs, f32vec3(vs_slice_dir, 0)));

    f32vec3 proj_normal_vs = normal_vs - slice_normal_vs * dot(slice_normal_vs, normal_vs);
    float slice_contrib_weight = length(proj_normal_vs);
    proj_normal_vs /= slice_contrib_weight;

    float n_angle = fast_acos(clamp(dot(proj_normal_vs, v_vs), -1.0, 1.0)) * sign(dot(vs_slice_dir, proj_normal_vs.xy - v_vs.xy));

    float theta_cos_max1 = cos(n_angle - PI * 0.5);
    float theta_cos_max2 = cos(n_angle + PI * 0.5);

    f32vec3 prev_sample0_vs = v_vs;
    f32vec3 prev_sample1_vs = v_vs;

    i32vec2 prev_sample_coord0 = i32vec2(px);
    i32vec2 prev_sample_coord1 = i32vec2(px);

    for (uint i = 0; i < SSGI_HALF_SAMPLE_COUNT; ++i) {
        {
            float t = float(i) + rand_offset;

            f32vec4 sample_cs = f32vec4(ray_hit_cs.xy - cs_slice_dir * t, 0, 1);
            i32vec2 sample_px = i32vec2(output_tex_size.xy * cs_to_uv(sample_cs.xy));

            if (any(bvec2(sample_px != prev_sample_coord0))) {
                prev_sample_coord0 = sample_px;
                sample_cs.z = fetch_depth(sample_px);
                theta_cos_max1 = process_sample(i, 1, n_angle, prev_sample0_vs, sample_cs, center_vs, normal_vs, v_vs, kernel_radius_ws, theta_cos_max1);
            }
        }

        {
            float t = float(i) + (1.0 - rand_offset);

            f32vec4 sample_cs = f32vec4(ray_hit_cs.xy + cs_slice_dir * t, 0, 1);
            i32vec2 sample_px = i32vec2(output_tex_size.xy * cs_to_uv(sample_cs.xy));

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
    f32vec4 col;
    col.a = max(0.0, inv_ao);
    col.rgb = col.aaa;

    col *= slice_contrib_weight;

    imageStore(daxa_image2D(ssao_image_id), i32vec2(px), f32vec4(col));
}
