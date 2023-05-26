#include <shared/shared.inl>

#include <utils/trace.glsl>
#include <utils/voxels.glsl>

#define LIGHTING 1

#if LIGHTING
#define AMBIENT_OCCLUSION 1
#define SHADOWS 1
#endif

#define LINEAR_FOG 0

#define DO_TEMPORAL_REPROJECTION 0

f32vec2 prev_uv_from_pos(in out Player player, f32 aspect, f32 fov, f32vec3 pos) {
    f32vec3 dir = normalize(pos - player.cam.prev_pos) * player.cam.prev_rot_mat;
    return dir.xy / dir.z * f32vec2(0.5 / aspect, 0.5) / player.cam.prev_tan_half_fov;
}

f32vec4 get_prev_sample(i32vec2 prev_pixel_i, f32vec3 hit, i32vec2 frame_dim) {
    if (min(frame_dim, prev_pixel_i) != prev_pixel_i || max(i32vec2(0, 0), prev_pixel_i) != prev_pixel_i)
        return f32vec4(f32vec3(0), MAX_SD);
    f32vec3 prev_hit_pos = imageLoad(render_prev_pos_image_id, prev_pixel_i).xyz;
    f32vec3 p0 = prev_hit_pos; // floor(prev_hit_pos * VOXEL_SCL);
    f32vec3 p1 = hit;          // floor(hit * VOXEL_SCL);
    f32vec3 del = p0 - p1;
    return f32vec4(prev_hit_pos, dot(del, del));
}

#define SKY_COL (f32vec3(0.02, 0.05, 0.90) * 4)
#define SKY_COL_B (f32vec3(0.11, 0.10, 0.54))

// #define SUN_TIME (deref(gpu_input).time)
#define SUN_TIME 1.8
#define SUN_COL (f32vec3(1, 0.85, 0.5) * 20)
#define SUN_DIR normalize(f32vec3(0.5 * abs(sin(SUN_TIME)), -cos(SUN_TIME), abs(sin(SUN_TIME))))

f32vec3 sample_sky_ambient(f32vec3 nrm) {
    f32 sun_val = dot(nrm, SUN_DIR) * 0.1 + 0.06;
    sun_val = pow(sun_val, 2) * 0.2;
    f32 sky_val = clamp(dot(nrm, f32vec3(0, 0, -1)) * 0.2 + 0.5, 0, 1);
    return mix(SKY_COL + sun_val * SUN_COL, SKY_COL_B, pow(sky_val, 2));

    // return f32vec3(0.1, 0.15, 0.9);

    // f32vec3 color = atmosphere(
    //     nrm.xzy,                        // normalized ray direction
    //     vec3(0, 6372e3, 0),             // ray origin
    //     SUN_DIR.xzy * 6372e3,           // position of the sun
    //     22.0,                           // intensity of the sun
    //     6371e3,                         // radius of the planet in meters
    //     6471e3,                         // radius of the atmosphere in meters
    //     vec3(5.5e-6, 13.0e-6, 22.4e-6), // Rayleigh scattering coefficient
    //     21e-6,                          // Mie scattering coefficient
    //     8e3,                            // Rayleigh scale height
    //     1.2e3,                          // Mie scale height
    //     0.758                           // Mie preferred scattering direction
    // );
    // return 1.0 - exp(-1.0 * color);
}

f32vec3 sample_sky(f32vec3 nrm) {
    f32vec3 light = sample_sky_ambient(nrm);
    f32 sun_val = dot(nrm, SUN_DIR) * 0.5 + 0.5;
    sun_val = sun_val * 200 - 199;
    sun_val = pow(clamp(sun_val * 1.1, 0, 1), 200);
    light += sun_val * SUN_COL;
    return light;
    // return f32vec3(0.02);
    // return f32vec3(0.1, 0.15, 0.9);
}

u32vec3 chunk_n;

bool is_hit(f32vec3 pos) {
    // f32 hit_dist2 = dot(pos, pos);
    // return hit_dist2 < MAX_SD * MAX_SD / 2;

    BoundingBox b;
    b.bound_min = f32vec3(0, 0, 0);
    b.bound_max = f32vec3(chunk_n) * (CHUNK_SIZE / VOXEL_SCL);

    return inside(pos, b);
}

struct HitInfo {
    f32vec3 diff_col;
    f32vec3 emit_col;
    f32vec3 nrm;
    f32 fresnel_fac;
    bool is_hit;
};

HitInfo get_hit_info(f32vec3 pos, f32vec3 ray_dir) {
    HitInfo result;
    result.is_hit = is_hit(pos);
    if (result.is_hit) {
        u32vec3 chunk_i = u32vec3(floor(pos * (f32(VOXEL_SCL) / CHUNK_SIZE)));
        u32 chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;
        u32vec3 voxel_i = u32vec3(pos * VOXEL_SCL);
        u32vec3 inchunk_voxel_i = voxel_i - chunk_i * CHUNK_SIZE;
        if ((chunk_i.x < chunk_n.x) && (chunk_i.y < chunk_n.y) && (chunk_i.z < chunk_n.z)) {
            u32 voxel_data = sample_voxel_chunk(voxel_malloc_global_allocator, voxel_chunks[chunk_index], inchunk_voxel_i, true);
            f32vec4 sample_col = uint_to_float4(voxel_data);
            if ((voxel_data >> 0x18) == 2) {
                result.diff_col = f32vec3(0);
                result.emit_col = sample_col.rgb * 20;
                result.is_hit = false;
            } else {
                result.diff_col = max(sample_col.rgb, f32vec3(0.01));
                result.emit_col = f32vec3(0);
            }
        }
        result.nrm = scene_nrm(voxel_malloc_global_allocator, voxel_chunks, chunk_n, pos);
    } else {
        result.diff_col = f32vec3(0.0);
        result.emit_col = sample_sky(ray_dir);
        result.nrm = -ray_dir;
    }
    result.fresnel_fac = pow(1.0 - dot(-ray_dir, result.nrm), 5.0);
    return result;
}

void reflect_ray(in out f32vec3 ray_dir, in out f32vec3 ray_col, in out HitInfo hit_info) {
    f32vec3 diff_dir = rand_lambertian_nrm(hit_info.nrm);
    f32vec3 spec_dir = reflect(ray_dir, hit_info.nrm);
    bool is_specular = false; // hit_info.fresnel_fac >= rand();
    ray_dir = mix(diff_dir, spec_dir, f32vec3(0.95) * f32(is_specular));
    if (is_specular) {
        ray_col *= 1.0;
    } else {
        ray_col *= hit_info.diff_col;
    }
}

f32vec3 color_trace(f32vec3 ray_pos, f32vec3 ray_dir) {
    f32vec3 result = f32vec3(0);
    f32vec3 ray_col = f32vec3(1);

    HitInfo hit_info = get_hit_info(ray_pos, ray_dir);
    ray_pos += hit_info.nrm * 0.01 / VOXEL_SCL;
    result += hit_info.emit_col * ray_col;
    reflect_ray(ray_dir, ray_col, hit_info);

    if (hit_info.is_hit) {
        // result = hit_info.diff_col;
        for (u32 i = 0; i < 2; ++i) {
            trace_hierarchy_traversal(voxel_malloc_global_allocator, voxel_chunks, chunk_n, ray_pos, ray_dir, 512, true);
            // trace_sparse(voxel_chunks, chunk_n, ray_pos, ray_dir, 6);
            hit_info = get_hit_info(ray_pos, ray_dir);
            result += hit_info.emit_col * ray_col;
            if (hit_info.is_hit) {
                ray_pos += hit_info.nrm * 0.01 / VOXEL_SCL;
                ray_col *= hit_info.diff_col;
                reflect_ray(ray_dir, ray_col, hit_info);
            } else {
                break;
            }
        }
    }

    return result;
}

#define SETTINGS deref(settings)
#define INPUT deref(gpu_input)
#define GLOBALS deref(globals)
#define CHUNK_PTRS(i) voxel_chunks[i]
#define CHUNKS(i) deref(voxel_chunks[i])
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    f32vec2 frame_dim = INPUT.frame_dim;
    f32vec2 inv_frame_dim = f32vec2(1.0, 1.0) / frame_dim;
    f32 aspect = frame_dim.x * inv_frame_dim.y;

    u32vec2 pixel_i = gl_GlobalInvocationID.xy;
    if (pixel_i.x >= frame_dim.x || pixel_i.y >= frame_dim.y)
        return;

    f32vec3 col = f32vec3(0);
    f32 accepted_count = 1;
    f32vec2 uv = f32vec2(pixel_i) * inv_frame_dim;
    uv = (uv - 0.5) * f32vec2(aspect, 1.0) * 2.0;
    chunk_n = u32vec3(1u << SETTINGS.log2_chunks_per_axis);
    rand_seed(pixel_i.x + pixel_i.y * INPUT.frame_dim.x + u32(INPUT.time * 719393));
    f32vec3 blue_noise = texelFetch(blue_noise_cosine_vec3, ivec3(pixel_i, INPUT.frame_index) & ivec3(127, 127, 63), 0).xyz * 2 - 1;
    f32vec3 cam_pos = create_view_pos(deref(globals).player);
    f32vec3 cam_dir = create_view_dir(deref(globals).player, uv);
    f32vec3 hit_pos = imageLoad(render_pos_image_id, i32vec2(pixel_i)).xyz;

#if 0
    f32vec3 ray_pos = cam_pos;
    trace_sphere_trace(ray_pos, cam_dir);

    f32 hit_dist2 = dot(ray_pos, ray_pos);
    bool is_hit = hit_dist2 < MAX_SD * MAX_SD / 2;
    if (is_hit) {
        f32vec3 nrm = sdmap_nrm(ray_pos);
        col = f32vec3(1) * (dot(nrm, SUN_DIR) * 0.5 + 0.5);
    }
#endif

#if 1
    HitInfo hit_info = get_hit_info(hit_pos, cam_dir);

    f32vec3 light = f32vec3(0);
    f32vec3 temp_pos;

#if SHADOWS
    temp_pos = hit_pos;

    f32mat3x3 sun_tbn = tbn_from_normal(SUN_DIR);
    f32vec3 sun_nrm = normalize(sun_tbn * blue_noise * 0.05 + SUN_DIR);

    trace_hierarchy_traversal(voxel_malloc_global_allocator, voxel_chunks, chunk_n, temp_pos, sun_nrm, 512, true);
    HitInfo sun_hit_info = get_hit_info(temp_pos, sun_nrm);
    if (!sun_hit_info.is_hit) {
        light += SUN_COL * 0.2 * dot(hit_info.nrm, sun_nrm);
    }
#endif

#if AMBIENT_OCCLUSION
    temp_pos = hit_pos;

    f32mat3x3 nrm_tbn = tbn_from_normal(hit_info.nrm);
    f32vec3 ao_ray_dir = nrm_tbn * blue_noise;

    // f32vec3 ao_ray_dir = rand_lambertian_nrm(hit_info.nrm);

    trace_hierarchy_traversal(voxel_malloc_global_allocator, voxel_chunks, chunk_n, temp_pos, ao_ray_dir, 16, false);
    trace_sparse(voxel_chunks, chunk_n, temp_pos, ao_ray_dir, 5);

    // trace_hierarchy_traversal(voxel_malloc_global_allocator, voxel_chunks, chunk_n, temp_pos, ao_ray_dir, 512, true);

    HitInfo ao_hit_info = get_hit_info(temp_pos, ao_ray_dir);
    if (!ao_hit_info.is_hit) {
        light += sample_sky_ambient(ao_ray_dir);
    }
#elif LIGHTING
    f32vec3 ao_ray_dir = hit_info.nrm;
    light += sample_sky_ambient(ao_ray_dir);
#else
    light = f32vec3(1);
#endif

    col = hit_info.diff_col * light + hit_info.emit_col;

#if LINEAR_FOG
    if (hit_info.is_hit) {
        f32 fog_factor = clamp(length(hit_pos - cam_pos) * 0.001, 0, 1);
        f32vec3 fog_col = f32vec3(0.1, 0.15, 0.9);
        col = mix(col, fog_col, fog_factor);
    }
#endif

#elif 0
    col = color_trace(hit_pos, cam_dir);

    f32 hit_dist = length(cam_pos - hit_pos);
    hit_dist = ceil(pow(hit_dist * 0.1, 1) / SETTINGS.fov * PI / 2);

    f32vec2 prev_uv = prev_uv_from_pos(GLOBALS.player, aspect, SETTINGS.fov, hit_pos);
    i32vec2 prev_pixel_i = i32vec2(round((prev_uv + 0.5) * frame_dim));

    i32vec2 pfc, finalpfc;
    f32 finaldist = MAX_SD;
    f32vec3 final_pos = f32vec3(0);
    const i32 SEARCH_RADIUS = 1;

    f32vec3 blurred_color = f32vec3(0);
    f32 blurred_samples = 0;
    f32vec3 hit_nrm = scene_nrm(voxel_malloc_global_allocator, voxel_chunks, chunk_n, hit_pos);

    // hit_dist = 0 + ceil(
    //     pow((dot(hit_nrm, cam_dir) * 0.5 + 0.5) * 2, 10)
    //     + pow(hit_dist * 0.1, 0.5) / SETTINGS.fov * PI / 2
    //     );

    for (i32 x = -SEARCH_RADIUS; x <= SEARCH_RADIUS; x++) {
        for (i32 y = -SEARCH_RADIUS; y <= SEARCH_RADIUS; y++) {
            pfc = prev_pixel_i + i32vec2(x, y);
            f32vec4 prev_sample = get_prev_sample(pfc, hit_pos, i32vec2(INPUT.frame_dim));
            f32 dist = prev_sample.w;
            f32vec3 prev_pos = prev_sample.xyz;
            if (dist < finaldist) {
                finalpfc = pfc;
                finaldist = dist;
            }
            f32vec3 prev_nrm = scene_nrm(voxel_malloc_global_allocator, voxel_chunks, chunk_n, prev_pos);
            bool normals_equal = prev_nrm == hit_nrm || dot(hit_nrm, cam_dir) < 0.5;
            bool positions_equal = floor(prev_pos * VOXEL_SCL / hit_dist) == floor(hit_pos * VOXEL_SCL / hit_dist);
            bool is_close_enough = dist < 0.01 * hit_dist;
            if (is_close_enough && positions_equal && normals_equal) {
                f32vec4 prev_col = imageLoad(render_prev_col_image_id, pfc);
                blurred_color += prev_col.rgb;
                accepted_count = max(accepted_count, min(prev_col.a + 1, 100));
                blurred_samples += 1.0;
            }
        }
    }

    bool accepted = blurred_samples > 0;

    // finaldist = clamp(finaldist, 0.01, 1.0 / VOXEL_SCL) / (1.0 / VOXEL_SCL);
    f32 alpha = mix(1.0, 1.0 / (accepted_count + 1), f32(accepted));
    col = col * alpha + (blurred_color / max(blurred_samples, 1)) * (1.0 - alpha);
    // col = blurred_color / blurred_samples;
    // col = hit_nrm;
#endif

#if 0
    // Naive frame blending:
    f32vec3 prev_col = imageLoad(render_col_image_id, i32vec2(pixel_i)).rgb;
    f32 alpha = 0.1;
    col = clamp(col, f32vec3(0), f32vec3(5));
    col = col * alpha + prev_col * (1.0 - alpha);
#endif

#if DO_TEMPORAL_REPROJECTION
    f32 hit_dist = length(cam_pos - hit_pos);
    // hit_dist = ceil(pow(hit_dist * 0.1, 1) / SETTINGS.fov * PI / 2);

    f32vec2 prev_uv = prev_uv_from_pos(GLOBALS.player, aspect, SETTINGS.fov, hit_pos);
    i32vec2 prev_pixel_i = i32vec2(round((prev_uv + 0.5) * frame_dim));

    i32vec2 pfc, finalpfc;
    f32 finaldist = MAX_SD;
    f32vec3 final_pos = f32vec3(0);
    const i32 SEARCH_RADIUS = 1;

    f32vec3 blurred_color = f32vec3(0);
    f32 blurred_samples = 0;
    f32vec3 hit_nrm = scene_nrm(voxel_malloc_global_allocator, voxel_chunks, chunk_n, hit_pos);

    for (i32 x = -SEARCH_RADIUS; x <= SEARCH_RADIUS; x++) {
        for (i32 y = -SEARCH_RADIUS; y <= SEARCH_RADIUS; y++) {
            pfc = prev_pixel_i + i32vec2(x, y);
            f32vec4 prev_sample = get_prev_sample(pfc, hit_pos, i32vec2(INPUT.frame_dim));
            f32 dist = prev_sample.w;
            f32vec3 prev_pos = prev_sample.xyz;
            f32vec3 prev_nrm = scene_nrm(voxel_malloc_global_allocator, voxel_chunks, chunk_n, prev_pos);
            bool normals_equal = prev_nrm == hit_nrm;
            bool positions_equal = true; // floor(prev_pos * VOXEL_SCL / hit_dist) == floor(hit_pos * VOXEL_SCL / hit_dist);
            bool is_close_enough = dist < 0.001 * hit_dist;
            if (dist < finaldist && is_close_enough && positions_equal && normals_equal) {
                finalpfc = pfc;
                finaldist = dist;
                blurred_samples = 1.0;
            }
        }
    }

    bool accepted = blurred_samples > 0;

    if (accepted) {
        f32vec4 prev_col = imageLoad(render_prev_col_image_id, finalpfc);
        blurred_color = prev_col.rgb;
        accepted_count = max(accepted_count, min(prev_col.a + 1, 10));
    }

    f32 alpha = mix(1.0, 1.0 / (accepted_count + 1), f32(accepted));
    col = col * alpha + (blurred_color / max(blurred_samples, 1)) * (1.0 - alpha);
#endif

#if SHOW_ALLOCATOR_DEBUG
    const u32 ALLOC_DEBUG_VIEW_SIZE = VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD * 48 * 2;
    if (pixel_i.x < ALLOC_DEBUG_VIEW_SIZE) {
        u32 index = pixel_i.x + pixel_i.y * ALLOC_DEBUG_VIEW_SIZE;

        u32 alloc_index = index / VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD;
        u32 in_page_alloc_index = index - alloc_index * VOXEL_MALLOC_MAX_ALLOCATIONS_IN_PAGE_BITFIELD;
        u32 chunk_index = alloc_index / 512;
        u32 in_chunk_index = alloc_index - chunk_index * 512;

        VoxelMalloc_PageInfo page_info = deref(voxel_chunks[chunk_index]).sub_allocator_state.page_allocation_infos[in_chunk_index];
        u32 local_consumption_bitmask = VoxelMalloc_PageInfo_extract_local_consumption_bitmask(page_info);

        if (chunk_index < chunk_n.x * chunk_n.y * chunk_n.z) {
            if (local_consumption_bitmask == 0) {
                col *= 0.04;
            } else {
                if (((local_consumption_bitmask >> in_page_alloc_index) & 1) == 1) {
                    col = f32vec3(0.7, 0, 0.1);
                } else {
                    col = f32vec3(0.8, 0.4, 0.1);
                }
            }
        }
    }
#endif

    imageStore(render_col_image_id, i32vec2(pixel_i), f32vec4(col, accepted_count));
}
#undef CHUNKS
#undef CHUNK_PTRS
#undef GLOBALS
#undef INPUT
#undef SETTINGS
