#include <shared/shared.inl>

#include <utils/trace.glsl>
#include <utils/downscale.glsl>

struct ViewRayContext {
    vec4 ray_dir_cs;
    vec4 ray_dir_vs_h;
    vec4 ray_dir_ws_h;
    vec4 ray_origin_cs;
    vec4 ray_origin_vs_h;
    vec4 ray_origin_ws_h;
    vec4 ray_hit_cs;
    vec4 ray_hit_vs_h;
    vec4 ray_hit_ws_h;
};

struct ViewConstants {
    f32mat4x4 sample_to_view;
    f32mat4x4 view_to_world;
};
ViewConstants view_constants;

// [Drobot2014a] Low Level Optimizations for GCN
float fast_sqrt(float x) {
    return uintBitsToFloat(0x1fbd1df5 + (floatBitsToUint(x) >> 1u));
}

// [Eberly2014] GPGPU Programming for Games and Science
float fast_acos(float inX) {
    float x = abs(inX);
    float res = -0.156583f * x + (PI * 0.5);
    res *= fast_sqrt(1.0f - x);
    return (inX >= 0) ? (res) : (PI - res);
}

ViewRayContext vrc_from_uv_and_depth(vec2 uv, float depth) {
    ViewRayContext result;
    result.ray_dir_cs = vec4(uv_to_cs(uv), 0.0, 1.0);
    result.ray_dir_vs_h = view_constants.sample_to_view * result.ray_dir_cs;
    result.ray_dir_ws_h = view_constants.view_to_world * result.ray_dir_vs_h;

    result.ray_origin_cs = vec4(uv_to_cs(uv), 1.0, 1.0);
    result.ray_origin_vs_h = view_constants.sample_to_view * result.ray_origin_cs;
    result.ray_origin_ws_h = view_constants.view_to_world * result.ray_origin_vs_h;

    result.ray_hit_cs = vec4(uv_to_cs(uv), depth, 1.0);
    result.ray_hit_vs_h = view_constants.sample_to_view * result.ray_hit_cs;
    result.ray_hit_ws_h = view_constants.view_to_world * result.ray_hit_vs_h;
    return result;
}

const float temporal_rotations[] = {60.0, 300.0, 180.0, 240.0, 120.0, 0.0};
const float temporal_offsets[] = {0.0, 0.5, 0.25, 0.75};

#define SSGI_KERNEL_RADIUS (60.0 * inv_frame_dim.y)
#define MAX_KERNEL_RADIUS_CS 0.4
const uint SSGI_HALF_SAMPLE_COUNT = 6;
#define USE_SSGI_FACING_CORRECTION 1

f32vec2 frame_dim;
f32vec2 inv_frame_dim;

float update_horizion_angle(float prev, float cur, float blend) {
    return (cur > prev) ? mix(prev, cur, blend) : (prev);
}

f32vec3 fetch_lighting(f32vec2 uv) {
    // return 0.0.xxx;
    return vec3(1.0);
    // int2 px = int2(input_tex_size.xy * uv);
    // float4 reproj = reprojection_tex[px];
    // return lerp(0.0, prev_radiance_tex[int2(input_tex_size.xy * (uv + reproj.xy))].xyz, reproj.z);
}

f32vec3 fetch_normal_vs(f32vec2 uv) {
    u32vec4 g_buffer_value = imageLoad(daxa_uimage2D(g_buffer_image_id), i32vec2(uv * imageSize(daxa_uimage2D(g_buffer_image_id))));
    f32vec3 nrm = u16_to_nrm(g_buffer_value.y);
    return normalize(nrm * deref(globals).player.cam.rot_mat);
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
float intersect_dir_plane_onesided(f32vec3 dir, f32vec3 normal, f32vec3 pt) {
    float d = -dot(pt, normal);
    float t = d / max(1e-5, -dot(dir, normal));
    return t;
}

// Kajiya
float process_sample(uint i, float intsgn, float n_angle, inout f32vec3 prev_sample_vs, f32vec4 sample_cs, f32vec3 center_vs, f32vec3 normal_vs, f32vec3 v_vs, float kernel_radius_ws, float theta_cos_max, inout f32vec4 color_accum) {
    if (sample_cs.z > 0) {
        f32vec4 sample_vs4 = view_constants.sample_to_view * sample_cs;
        f32vec3 sample_vs = sample_vs4.xyz / sample_vs4.w;
        f32vec3 sample_vs_offset = sample_vs - center_vs;
        float sample_vs_offset_len = length(sample_vs_offset);

        float sample_theta_cos = dot(sample_vs_offset, v_vs) / sample_vs_offset_len;
        const float sample_distance_normalized = sample_vs_offset_len / kernel_radius_ws;

        if (sample_distance_normalized < 1.0) {
            const float sample_influence = smoothstep(1, 0, sample_distance_normalized);

            bool sample_visible = sample_theta_cos >= theta_cos_max;
            float theta_cos_prev = theta_cos_max;
            float theta_delta = theta_cos_max;
            theta_cos_max = update_horizion_angle(theta_cos_max, sample_theta_cos, sample_influence);
            theta_delta = theta_cos_max - theta_delta;

            if (sample_visible) {
                f32vec3 lighting = fetch_lighting(cs_to_uv(sample_cs.xy));

                f32vec3 sample_normal_vs = fetch_normal_vs(cs_to_uv(sample_cs.xy));
                float theta_cos_prev_trunc = theta_cos_prev;

                if (i > 0) {
                    // Account for the sampled surface's normal, and how it's facing the center pixel
                    f32vec3 p1 = prev_sample_vs * min(intersect_dir_plane_onesided(prev_sample_vs, sample_normal_vs, sample_vs), intersect_dir_plane_onesided(prev_sample_vs, normal_vs, center_vs));
                    theta_cos_prev_trunc = clamp(dot(normalize(p1 - center_vs), v_vs), theta_cos_prev_trunc, theta_cos_max);
                }

                {
                    // Scale the lighting contribution by the cosine factor

                    n_angle *= -intsgn;

                    float h1 = fast_acos(theta_cos_prev_trunc);
                    float h2 = fast_acos(theta_cos_max);

                    float h1p = n_angle + max(h1 - n_angle, -(PI * 0.5));
                    float h2p = n_angle + min(h2 - n_angle, (PI * 0.5));

                    float inv_ao =
                        integrate_half_arc(h1p, n_angle) -
                        integrate_half_arc(h2p, n_angle);

                    lighting *= inv_ao;
                    lighting *= step(0.0, dot(-normalize(sample_vs_offset), sample_normal_vs));
                }

                color_accum += f32vec4(lighting, 1.0);
            }
        }

        prev_sample_vs = sample_vs;
    } else {
        // Sky; assume no occlusion
        theta_cos_max = update_horizion_angle(theta_cos_max, -1, 1);
    }

    return theta_cos_max;
}

u32vec2 px;
f32vec3 nrm;
f32vec2 uv;
f32 depth;
f32 aspect;

float kajiya_calc_ao() {
    const f32 znear = 0.01;
    view_constants.sample_to_view = f32mat4x4(
        f32vec4(inv_frame_dim.x, 0.0, 0.0, 0.0),
        f32vec4(0.0, inv_frame_dim.y, 0.0, 0.0),
        f32vec4(0.0, 0.0, 0.0, 1.0 / znear),
        f32vec4(0.0, 0.0, -1.0, 0.0));
    view_constants.view_to_world = f32mat4x4(
        f32vec4(deref(globals).player.cam.rot_mat[0], 0.0),
        f32vec4(deref(globals).player.cam.rot_mat[1], 0.0),
        f32vec4(deref(globals).player.cam.rot_mat[2], 0.0),
        f32vec4(deref(globals).player.cam.pos, 0.0));
    // kajiya -------------------
    float spatial_direction_noise = 1.0 / 16.0 * ((((px.x + px.y) & 3) << 2) + (px.x & 3));
    float temporal_direction_noise = temporal_rotations[deref(gpu_input).frame_index % 6] / 360.0;
    float spatial_offset_noise = (1.0 / 4.0) * ((px.y - px.x) & 3);
    float temporal_offset_noise = temporal_offsets[deref(gpu_input).frame_index / 6 % 4];

    f32vec3 nrm_viewspace = normalize(nrm * deref(globals).player.cam.rot_mat);
    const ViewRayContext vrc = vrc_from_uv_and_depth(uv, depth);

    f32vec3 v_vs = -normalize(vrc.ray_dir_vs_h.xyz);
    f32vec4 ray_hit_cs = vrc.ray_hit_cs;
    f32vec3 ray_hit_vs = vrc.ray_hit_vs_h.xyz / vrc.ray_hit_vs_h.w;

    float ss_angle = fract(spatial_direction_noise + temporal_direction_noise) * PI;
    float rand_offset = fract(spatial_offset_noise + temporal_offset_noise);

    f32vec2 cs_slice_dir = f32vec2(cos(ss_angle) * frame_dim.y * inv_frame_dim.x, sin(ss_angle));

    float kernel_radius_ws;
    float kernel_radius_shrinkage = 1;
    {
        const float ws_to_cs = 0.5 / -ray_hit_vs.z * frame_dim.y;
        const float cs_kernel_radius_scaled = SSGI_KERNEL_RADIUS;
        kernel_radius_ws = cs_kernel_radius_scaled / ws_to_cs;
        cs_slice_dir *= cs_kernel_radius_scaled;
        // Calculate AO radius shrinkage (if camera is too close to a surface)
        float max_kernel_radius_cs = MAX_KERNEL_RADIUS_CS;
        kernel_radius_shrinkage = min(1.0, max_kernel_radius_cs / cs_kernel_radius_scaled);
    }

    cs_slice_dir *= kernel_radius_shrinkage;
    kernel_radius_ws *= kernel_radius_shrinkage;

    f32vec3 center_vs = ray_hit_vs.xyz;

    cs_slice_dir *= 1.0 / float(SSGI_HALF_SAMPLE_COUNT);
    f32vec2 vs_slice_dir = (f32vec4(cs_slice_dir, 0, 0) * view_constants.sample_to_view).xy;
    f32vec3 slice_normal_vs = normalize(cross(v_vs, f32vec3(vs_slice_dir, 0)));

    f32vec3 proj_normal_vs = nrm_viewspace - slice_normal_vs * dot(slice_normal_vs, nrm_viewspace);
    float slice_contrib_weight = length(proj_normal_vs);
    proj_normal_vs /= slice_contrib_weight;

    float n_angle = fast_acos(clamp(dot(proj_normal_vs, v_vs), -1.0, 1.0)) * sign(dot(vs_slice_dir, proj_normal_vs.xy - v_vs.xy));

    float theta_cos_max1 = cos(n_angle - (PI * 0.5));
    float theta_cos_max2 = cos(n_angle + (PI * 0.5));

    f32vec4 color_accum = 0.0.xxxx;

    f32vec3 prev_sample0_vs = v_vs;
    f32vec3 prev_sample1_vs = v_vs;

    i32vec2 prev_sample_coord0 = i32vec2(px);
    i32vec2 prev_sample_coord1 = i32vec2(px);

    for (uint i = 0; i < SSGI_HALF_SAMPLE_COUNT; ++i) {
        {
            float t = float(i) + rand_offset;

            f32vec4 sample_cs = f32vec4(ray_hit_cs.xy - cs_slice_dir * t, 0, 1);
            i32vec2 sample_px = i32vec2(frame_dim.xy * cs_to_uv(sample_cs.xy));

            if (sample_px.x != prev_sample_coord0.x || sample_px.y != prev_sample_coord0.y) {
                prev_sample_coord0 = sample_px;
                sample_cs.z = imageLoad(daxa_image2D(depth_image), i32vec2(sample_px)).r;
                theta_cos_max1 = process_sample(i, 1, n_angle, prev_sample0_vs, sample_cs, center_vs, nrm_viewspace, v_vs, kernel_radius_ws, theta_cos_max1, color_accum);
            }
        }

        {
            float t = float(i) + (1.0 - rand_offset);

            f32vec4 sample_cs = f32vec4(ray_hit_cs.xy + cs_slice_dir * t, 0, 1);
            i32vec2 sample_px = i32vec2(frame_dim.xy * cs_to_uv(sample_cs.xy));

            if (sample_px.x != prev_sample_coord1.x || sample_px.y != prev_sample_coord1.y) {
                prev_sample_coord1 = sample_px;
                sample_cs.z = imageLoad(daxa_image2D(depth_image), i32vec2(sample_px)).r;
                theta_cos_max2 = process_sample(i, -1, n_angle, prev_sample1_vs, sample_cs, center_vs, nrm_viewspace, v_vs, kernel_radius_ws, theta_cos_max2, color_accum);
            }
        }
    }

    float h1 = -fast_acos(theta_cos_max1);
    float h2 = +fast_acos(theta_cos_max2);

    float h1p = n_angle + max(h1 - n_angle, -(PI * 0.5));
    float h2p = n_angle + min(h2 - n_angle, (PI * 0.5));

    float inv_ao = integrate_arc(h1p, h2p, n_angle);
    float ao = max(0.0, inv_ao);
    ao *= slice_contrib_weight;

    return ao;
    // -------------------------
}

float my_ssao() {
    rand_seed(u32(((gl_GlobalInvocationID.x * 123 ^ gl_GlobalInvocationID.y * 567) + deref(gpu_input).frame_index)));
    f32vec3 nrm_viewspace = normalize(nrm * deref(globals).player.cam.rot_mat);

    f32 ao = 0.0;
    const u32 SAMPLE_N = 6;
    const f32 bias = 0.1;
    const f32 radius = 4.0;

    vec3 view_pos = create_view_pos(deref(globals).player);
    vec3 view_dir = create_view_dir(deref(globals).player, (uv * 2.0 - 1.0) * vec2(aspect, 1.0));
    vec3 frag_pos = view_dir * depth + view_pos;

    for (u32 i = 0; i < SAMPLE_N; ++i) {
        f32 rng = rand();
        f32vec3 ao_sample = rand_hemi_dir(nrm_viewspace) * mix(0.1, 1.0, rng * rng) * radius;
        // f32vec4 sample_tex_coord = deref(globals).player.cam.proj_mat * f32vec4(ao_sample + frag_pos, 1.0);
        // sample_tex_coord.xyz = (sample_tex_coord.xyz * 0.5 / sample_tex_coord.w) * vec3(1.0) + 0.5;
        f32vec2 sample_tex_coord = f32vec2(uv) + ao_sample.xy * 0.01;
        f32 sample_depth = imageLoad(daxa_image2D(depth_image), i32vec2(sample_tex_coord.xy * frame_dim / SHADING_SCL)).r;
        ao += (depth + ao_sample.z >= sample_depth + bias ? 1.0 : 0.0);
    }

    return 1.0 - ao / SAMPLE_N;
}

float calc_ao() {
    return my_ssao();
    // return kajiya_calc_ao();
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    px = gl_GlobalInvocationID.xy;

    u32vec2 offset = get_downscale_offset(gpu_input);

    f32vec2 pixel_p = f32vec2(px * SHADING_SCL + offset) + 0.5;
    frame_dim = deref(gpu_input).frame_dim;
    inv_frame_dim = f32vec2(1.0, 1.0) / frame_dim;
    aspect = frame_dim.x * inv_frame_dim.y;

    uv = pixel_p * inv_frame_dim;

    depth = imageLoad(daxa_image2D(depth_image), i32vec2(px)).r;
    u32vec4 g_buffer_value = imageLoad(daxa_uimage2D(g_buffer_image_id), i32vec2(px * SHADING_SCL + offset));
    nrm = u16_to_nrm(g_buffer_value.y);

    if (depth == MAX_SD || dot(nrm, nrm) == 0.0) {
        imageStore(daxa_image2D(ssao_image_id), i32vec2(px), f32vec4(1, 0, 0, 0));
        return;
    }

    float ao = calc_ao();

    imageStore(daxa_image2D(ssao_image_id), i32vec2(px), f32vec4(ao, 0, 0, 0));
}
