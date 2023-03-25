#include <shared/shared.inl>

#include <utils/trace.glsl>
#include <utils/voxels.glsl>

DAXA_USE_PUSH_CONSTANT(ColorSceneComputePush)

#define AMBIENT_OCCLUSION 0

#define SETTINGS deref(daxa_push_constant.gpu_settings)
f32vec2 prev_uv_from_pos(in out Player player, f32 aspect, f32vec3 pos) {
    f32vec3 dir = inverse(player.cam.prev_rot_mat) * normalize(pos - player.cam.prev_pos);
    return dir.xy / dir.z * f32vec2(0.5 / aspect, 0.5);
}
#undef SETTINGS

f32 distance_pixel(i32vec2 prev_pixel_i, f32vec3 hit, i32vec2 frame_dim) {
    if (min(frame_dim, prev_pixel_i) != prev_pixel_i || max(i32vec2(0, 0), prev_pixel_i) != prev_pixel_i)
        return MAX_SD;
    f32vec3 prev_hit_pos = imageLoad(daxa_push_constant.render_pos_image_id, prev_pixel_i).xyz;
    return length(prev_hit_pos - hit);
}

#define SETTINGS deref(daxa_push_constant.gpu_settings)
#define INPUT deref(daxa_push_constant.gpu_input)
#define GLOBALS deref(daxa_push_constant.gpu_globals)
#define CHUNK_PTRS(i) daxa_push_constant.voxel_chunks[i]
#define CHUNKS(i) deref(daxa_push_constant.voxel_chunks[i])
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    f32vec2 frame_dim = INPUT.frame_dim;
    f32vec2 inv_frame_dim = f32vec2(1.0, 1.0) / frame_dim;
    f32 aspect = frame_dim.x * inv_frame_dim.y;

    u32vec2 pixel_i = gl_GlobalInvocationID.xy;
    if (pixel_i.x >= frame_dim.x ||
        pixel_i.y >= frame_dim.y)
        return;

    f32vec3 hit_pos = imageLoad(
                          daxa_push_constant.render_pos_image_id,
                          i32vec2(pixel_i))
                          .xyz;

    f32vec3 hit_nrm = scene_nrm(hit_pos);
    f32 f = step((int(hit_pos.x) + int(hit_pos.y) + int(hit_pos.z)) % 2, 0.5);

    f32vec3 light_dir = normalize(f32vec3(1, 1.7, 2.1));
    f32vec3 light_col = f32vec3(3, 2.8, 1.2);
    f32vec3 sky_col = f32vec3(0.16, 0.19, 1.5);

    f32vec3 col;

    if (dot(hit_pos, hit_pos) < MAX_SD * MAX_SD / 2) {
        u32vec3 chunk_n = u32vec3(1u << SETTINGS.log2_chunks_per_axis);
        u32vec3 chunk_i = u32vec3(floor(hit_pos * (f32(VOXEL_SCL) / CHUNK_SIZE)));
        u32 chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;
        u32vec3 voxel_i = u32vec3(hit_pos * VOXEL_SCL);
        u32vec3 inchunk_voxel_i = voxel_i - chunk_i * CHUNK_SIZE;
        if ((chunk_i.x < chunk_n.x) && (chunk_i.y < chunk_n.y) && (chunk_i.z < chunk_n.z)) {
            // u32 lod = sample_lod(daxa_push_constant.voxel_malloc_global_allocator, CHUNK_PTRS(chunk_index), chunk_i, inchunk_voxel_i);
            // col = f32vec3(lod) / 7;

            // col = f32vec3(0.1);

            u32 voxel_data = sample_voxel_chunk(daxa_push_constant.voxel_malloc_global_allocator, CHUNK_PTRS(chunk_index), inchunk_voxel_i);
            f32vec4 sample_col = uint_to_float4(voxel_data);
            col = sample_col.rgb;

            // if (length(hit_pos - GLOBALS.pick_pos) < 31.0 / VOXEL_SCL) {
            //     col *= 0.1;
            // }

            // if (CHUNKS(chunk_index).edit_stage == 3) {
            //     col += 0.1;
            // }

            f32vec3 light_contrib = sky_col; // * (dot(hit_nrm, f32vec3(0, 0, 1)) * 0.5 + 0.5);

#if AMBIENT_OCCLUSION
            f32vec3 ray_pos = hit_pos + hit_nrm * 1 / VOXEL_SCL;
            f32vec3 ray_dir = rand_lambertian_nrm(hit_nrm, fract(rand2(pixel_i + fract(INPUT.time))));
            trace_sparse(daxa_push_constant.voxel_chunks, chunk_n, ray_pos, ray_dir, 2);
            if (dot(ray_pos, ray_pos) < MAX_SD * MAX_SD / 2) {
                light_contrib *= 0.0;
            }
#endif
            {
                f32vec3 ray_pos = hit_pos + hit_nrm * 0.01 / VOXEL_SCL;
                f32vec3 ray_dir = light_dir;
                trace_hierarchy_traversal(daxa_push_constant.voxel_malloc_global_allocator, daxa_push_constant.voxel_chunks, chunk_n, ray_pos, ray_dir, 256);
                if (dot(ray_pos, ray_pos) > MAX_SD * MAX_SD / 2) {
                    light_contrib += light_col * max(dot(hit_nrm, light_dir), 0);
                }
            }
            col *= light_contrib;

            // f32vec2 prev_uv = prev_uv_from_pos(GLOBALS.player, aspect, hit_pos);
            // i32vec2 prev_pixel_i = i32vec2(round((prev_uv + 0.5) * frame_dim));
            // col = f32vec3(prev_pixel_i / 100, 0);

            // if (min(frame_dim, prev_pixel_i) != prev_pixel_i || max(i32vec2(0, 0), prev_pixel_i) != prev_pixel_i)
            //     col = f32vec3(1);

            // i32vec2 pfc, finalpfc;
            // f32 finaldist = MAX_SD;
            // f32vec3 final_pos = f32vec3(0);
            // const i32 SEARCH_RADIUS = 0;
            // for (i32 x = -SEARCH_RADIUS; x <= SEARCH_RADIUS; x++) {
            //     for (i32 y = -SEARCH_RADIUS; y <= SEARCH_RADIUS; y++) {
            //         pfc = prev_pixel_i + i32vec2(x, y);
            //         if (min(frame_dim, prev_pixel_i) == prev_pixel_i && max(i32vec2(0, 0), prev_pixel_i) == prev_pixel_i) {
            //             f32vec3 prev_hit_pos = imageLoad(
            //                                        daxa_push_constant.render_pos_image_id,
            //                                        prev_pixel_i)
            //                                        .xyz;
            //             f32 dist = length(prev_hit_pos - hit_pos);
            //             if (dist < finaldist) {
            //                 finalpfc = pfc;
            //                 finaldist = dist;
            //                 final_pos = prev_hit_pos;
            //             }
            //         }
            //     }
            // }

            // finaldist = clamp(finaldist, 0.01, 1.0 / VOXEL_SCL) / (1.0 / VOXEL_SCL);

            // f32vec3 prev_col =
            //     imageLoad(
            //         daxa_push_constant.render_col_image_id,
            //         finalpfc)
            //         .rgb;
            // f32 alpha = 0.1; // pow(finaldist, 1);
            // col = clamp(col, f32vec3(0), f32vec3(5));
            // col = col * alpha + prev_col * (1.0 - alpha);

            // col = fract(final_pos);
            // col = fract(hit_pos);
            // col = ((hit_pos) - (final_pos));

            // Naive frame blending:
            // f32vec3 prev_col =
            //     imageLoad(
            //         daxa_push_constant.render_col_image_id,
            //         i32vec2(pixel_i))
            //         .rgb;
            // f32 alpha = 0.01; // pow(finaldist, 1);
            // col = clamp(col, f32vec3(0), f32vec3(5));
            // col = col * alpha + prev_col * (1.0 - alpha);
        }
    } else {
        col = sky_col;
    }

    imageStore(
        daxa_push_constant.render_col_image_id,
        i32vec2(pixel_i),
        f32vec4(col, 0));
}
#undef CHUNKS
#undef CHUNK_PTRS
#undef GLOBALS
#undef INPUT
#undef SETTINGS
