#include <shared/shared.inl>

DAXA_USE_PUSH_CONSTANT(DrawCompPush)

#include <utils/rand.glsl>
#include <utils/raytrace.glsl>
#include <utils/sky.glsl>

#define DO_RENDERING 1

#define VISUALIZE_COMPLEXITY 0

#define RENDER_SHADOWS 1
#define RENDER_SHADING 1
#define RENDER_FOG 1

b32 get_flag(u32 index) {
    return ((INPUT.settings.flags >> index) & 0x01) == 0x01;
}

TraceRecord trace_scene(in Ray ray, in out i32 complexity) {
    complexity = 0;

    TraceRecord trace;
    default_init(trace.intersection_record);
    trace.color = sample_sky(ray.nrm);
    trace.material = 0;
    trace.object_i = 0;

    for (u32 i = 0; i < SCENE.sphere_n; ++i) {
        Sphere s = SCENE.spheres[i];
        IntersectionRecord s_hit = intersect(ray, s);
        if (s_hit.hit && s_hit.dist < trace.intersection_record.dist) {
            trace.intersection_record = s_hit;
            trace.color = f32vec3(0.5, 0.5, 1.0);
            trace.material = 1;
            trace.object_i = i;
        }
    }

    for (u32 i = 0; i < SCENE.box_n; ++i) {
        BoundingBox b = SCENE.boxes[i];
        IntersectionRecord b_hit = intersect(ray, b);
        if (b_hit.hit && b_hit.dist < trace.intersection_record.dist) {
            trace.intersection_record = b_hit;
            trace.color = f32vec3(0.1, 0.1, 0.1);
            trace.material = 1;
            trace.object_i = i;
        }
    }

    for (u32 i = 0; i < SCENE.capsule_n; ++i) {
        CapsulePoints c = SCENE.capsules[i];
        IntersectionRecord c_hit = intersect(ray, c);
        if (c_hit.hit && c_hit.dist < trace.intersection_record.dist) {
            trace.intersection_record = c_hit;
            trace.color = f32vec3(1.0, 0.5, 0.5);
            trace.material = 2;
            trace.object_i = i;
        }
    }

    {
        IntersectionRecord b0_hit = intersect_voxels(ray, complexity);
        if (b0_hit.hit && b0_hit.dist < trace.intersection_record.dist) {
            trace.intersection_record = b0_hit;
            trace.color = f32vec3(0.5, 1.0, 0.5);
            trace.material = 3;
        }
    }

    if (INPUT.settings.tool_id == GAME_TOOL_BRUSH && GLOBALS.pick_intersection.hit) {
        i32 brush_complexity = 0;
        IntersectionRecord b0_hit = intersect_brush_voxels(ray, brush_complexity);
        complexity += brush_complexity;
        if (b0_hit.hit && b0_hit.dist < trace.intersection_record.dist + 0.001) {
            trace.intersection_record = b0_hit;
            trace.color = f32vec3(0.5, 0.5, 0.5);
            trace.material = 4;
        }
    }

    // {
    //     IntersectionRecord b0_hit = intersect_sdmap(ray);
    //     if (b0_hit.hit && b0_hit.dist < trace.intersection_record.dist) {
    //         trace.intersection_record = b0_hit;
    //         trace.color = f32vec3(0.5);
    //         trace.material = 5;
    //     }
    // }

    return trace;
}

f32 vertex_ao(f32vec2 side, f32 corner) {
    // if (side.x == 1.0 && side.y == 1.0) return 1.0;
    return (side.x + side.y + max(corner, side.x * side.y)) / 3.0;
}

f32vec4 voxel_ao(f32vec4 side, f32vec4 corner) {
    f32vec4 ao;
    ao.x = vertex_ao(side.xy, corner.x);
    ao.y = vertex_ao(side.yz, corner.y);
    ao.z = vertex_ao(side.zw, corner.z);
    ao.w = vertex_ao(side.wx, corner.w);
    return 1.0 - ao;
}

#define TONEMAPPER 2

f32vec3 filmic(f32vec3 color) {
#if TONEMAPPER == 0
    color = max(color, f32vec3(0, 0, 0));
    color = (color * (6.2 * color + 0.5)) / (color * (6.2 * color + 1.7) + 0.06);
#elif TONEMAPPER == 1
    color = pow(color, f32vec3(2.2));
#endif
    return color;
}

f32vec3 filmic_inv(f32vec3 color) {
#if TONEMAPPER == 0
    color = max(color, f32vec3(0, 0, 0));
    color = (-sqrt(5.0) * sqrt(701.0 * color * color - 106.0 * color + 125.0) - 85 * color + 25) / (620 * (color - 1));
#elif TONEMAPPER == 1
    color = pow(color, f32vec3(1.0 / 2.2));
#endif
    return color;
}

f32vec3 voxel_color(f32vec3 hit_pos, f32vec3 hit_nrm) {
    u32 temp_chunk_index;
    f32vec3 col;
    VoxelSampleInfo chunk_info = get_voxel_sample_info_WORLD(hit_pos - hit_nrm * 0.01);
    Voxel vox = unpack_voxel(sample_packed_voxel_WORLD(chunk_info.chunk_index, chunk_info.voxel_index));
    col = vox.col;
    // if ((VOXEL_WORLD.chunks_genstate[chunk_info.chunk_index].edit_stage == 3))
    //     col = f32vec3(0.2, 1.0, 0.2);

    // for (i32 zi = 0; zi < 3; ++zi) {
    //     if (VOXEL_WORLD.chunk_update_n >= 64)
    //         break;
    //     for (i32 yi = 0; yi < 3; ++yi) {
    //         if (VOXEL_WORLD.chunk_update_n >= 64)
    //             break;
    //         for (i32 xi = 0; xi < 3; ++xi) {
    //             if (VOXEL_WORLD.chunk_update_n >= 64)
    //                 break;
    //             u32 i = get_chunk_index_WORLD(get_chunk_i(get_voxel_i_WORLD(GLOBALS.brush_origin + (i32vec3(xi, yi, zi) - 1) * CHUNK_SIZE / VOXEL_SCL)));
    //             if (VOXEL_WORLD.chunks_genstate[i].edit_stage == 2 && i == chunk_info.chunk_index ) {
    //                 col = f32vec3(0.2, 1.0, 0.2);
    //             }
    //         }
    //     }
    // }

    // if ((PLAYER.view_state & 0x01) == 1) {
    //     col = f32vec3(1, 1, 1);
    // }

    // col = f32vec3(sample_lod(hit_pos, temp_chunk_index)) * 0.1;

    // col = f32vec3(VOXEL_WORLD.chunks_genstate[chunk_info.chunk_index].edit_stage) * 0.2;

    if (inside(hit_pos + hit_nrm * 0.1, VOXEL_WORLD.box)) {
        // col = f32vec3(1, 1, 1);

        f32vec3 mask = abs(hit_nrm);
        f32vec3 v_pos = hit_pos * VOXEL_SCL - hit_nrm * 0.01;
        f32vec3 b_pos = floor(v_pos + hit_nrm * 0.1) / VOXEL_SCL;
        f32vec3 d1 = mask.zxy / VOXEL_SCL;
        f32vec3 d2 = mask.yzx / VOXEL_SCL;
        f32vec4 side = f32vec4(
            sample_lod(b_pos + d1, temp_chunk_index) == 0,
            sample_lod(b_pos + d2, temp_chunk_index) == 0,
            sample_lod(b_pos - d1, temp_chunk_index) == 0,
            sample_lod(b_pos - d2, temp_chunk_index) == 0);
        f32vec4 corner = f32vec4(
            sample_lod(b_pos + d1 + d2, temp_chunk_index) == 0,
            sample_lod(b_pos - d1 + d2, temp_chunk_index) == 0,
            sample_lod(b_pos - d1 - d2, temp_chunk_index) == 0,
            sample_lod(b_pos + d1 - d2, temp_chunk_index) == 0);
        f32vec2 uv = mod(f32vec2(dot(mask * v_pos.yzx, f32vec3(1, 1, 1)), dot(mask * v_pos.zxy, f32vec3(1, 1, 1))), f32vec2(1, 1));
        f32vec4 ao = voxel_ao(side, corner);
        f32 interp_ao = mix(mix(ao.z, ao.w, uv.x), mix(ao.y, ao.x, uv.x), uv.y);
        col *= interp_ao * 0.6 + 0.4;
    }

    return col;
}

f32vec3 brush_voxel_color(f32vec3 hit_pos, f32vec3 hit_nrm) {
    u32 temp_chunk_index;
    f32vec3 col = f32vec3(0);

    VoxelSampleInfo chunk_info = get_voxel_sample_info_BRUSH(hit_pos - VOXEL_BRUSH.box.bound_min - hit_nrm * 0.01);
    Voxel vox = unpack_voxel(sample_packed_voxel_BRUSH(chunk_info.chunk_index, chunk_info.voxel_index));
    col = vox.col;

    if (get_flag(GPU_INPUT_FLAG_INDEX_BRUSH_PREVIEW_OVERLAY)) {
        f32vec3 b_pos = hit_pos - VOXEL_BRUSH.box.bound_min;
        b_pos = floor(b_pos * VOXEL_SCL) / VOXEL_SCL;
        f32 v = step(fract((b_pos.x + b_pos.y + b_pos.z + INPUT.time * 4) * 0.5), 0.5) * 0.5;
        col = mix(col, f32vec3(0.01, 0.01, 0.8), v);
    }

    if (inside(hit_pos + hit_nrm * 0.1, VOXEL_BRUSH.box)) {
        f32vec3 mask = abs(hit_nrm);
        f32vec3 v_pos = (hit_pos - VOXEL_BRUSH.box.bound_min) * VOXEL_SCL - hit_nrm * 0.01;
        f32vec3 b_pos = floor(v_pos + hit_nrm * 0.1) / VOXEL_SCL;
        f32vec3 d1 = mask.zxy / VOXEL_SCL;
        f32vec3 d2 = mask.yzx / VOXEL_SCL;
        f32vec4 side = f32vec4(
            sample_brush_lod(b_pos + d1, temp_chunk_index) == 0,
            sample_brush_lod(b_pos + d2, temp_chunk_index) == 0,
            sample_brush_lod(b_pos - d1, temp_chunk_index) == 0,
            sample_brush_lod(b_pos - d2, temp_chunk_index) == 0);
        f32vec4 corner = f32vec4(
            sample_brush_lod(b_pos + d1 + d2, temp_chunk_index) == 0,
            sample_brush_lod(b_pos - d1 + d2, temp_chunk_index) == 0,
            sample_brush_lod(b_pos - d1 - d2, temp_chunk_index) == 0,
            sample_brush_lod(b_pos + d1 - d2, temp_chunk_index) == 0);
        f32vec2 uv = mod(f32vec2(dot(mask * v_pos.yzx, f32vec3(1, 1, 1)), dot(mask * v_pos.zxy, f32vec3(1, 1, 1))), f32vec2(1, 1));
        f32vec4 ao = voxel_ao(side, corner);
        f32 interp_ao = mix(mix(ao.z, ao.w, uv.x), mix(ao.y, ao.x, uv.x), uv.y);
        col *= interp_ao * 0.6 + 0.4;
    }

    return col;
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
#if DO_RENDERING
    u32vec3 pixel_i = gl_GlobalInvocationID.xyz;
    if (pixel_i.x >= INPUT.frame_dim.x ||
        pixel_i.y >= INPUT.frame_dim.y)
        return;

    f32vec2 pixel_p = pixel_i.xy;
    f32vec2 frame_dim = INPUT.frame_dim;
    f32vec2 inv_frame_dim = f32vec2(1.0, 1.0) / frame_dim;
    f32 aspect = frame_dim.x * inv_frame_dim.y;

    f32 uv_rand_offset = INPUT.time;
    f32vec2 uv_offset = f32vec2(rand(pixel_p + uv_rand_offset + 10), rand(pixel_p + uv_rand_offset)) * 1.0 - 0.5;
    pixel_p += uv_offset * INPUT.settings.jitter_scl;

    f32vec2 uv = pixel_p * inv_frame_dim;
    uv = (uv - 0.5) * f32vec2(aspect, 1.0) * 2.0;

    f32vec3 col = f32vec3(0, 0, 0);

    Ray view_ray = create_view_ray(uv);
    view_ray.inv_nrm = 1.0 / view_ray.nrm;

    i32 complexity;
    TraceRecord view_trace_record = trace_scene(view_ray, complexity);

    col = view_trace_record.color;

    f32 hit_dist = MAX_SD;

#if VISUALIZE_COMPLEXITY
    col = filmic_inv(hsv2rgb(f32vec3(complexity * (0.5 / (WORLD_BLOCK_NX + WORLD_BLOCK_NY + WORLD_BLOCK_NZ)) * 16 + 0.5, 1.0, 0.9)));
#else

    if (false) {
        f32vec2 tex_uv = pixel_p * inv_frame_dim;

        Ray optical_ray = view_ray;
        optical_ray.o = f32vec3(0, 0, PLANET_RADIUS + 0.01);

        f32 o_depth;
        o_depth = optical_depth(optical_ray, calc_atmosphere_depth(optical_ray));
        // o_depth = abs(o_depth - (optical_depth_baked(optical_ray)));
        o_depth = optical_depth_baked(optical_ray);

        col = filmic_inv(f32vec3(o_depth));
    } else if (view_trace_record.intersection_record.hit) {
        f32vec3 hit_pos = view_ray.o + view_ray.nrm * view_trace_record.intersection_record.dist;
        f32vec3 hit_nrm = view_trace_record.intersection_record.nrm;
        hit_dist = view_trace_record.intersection_record.dist;
        Ray bounce_ray;
        bounce_ray.o = hit_pos;

        // Voxel hit_voxel;
        // hit_voxel.nrm = hit_nrm;

        switch (view_trace_record.material) {
        default: {
            bounce_ray.nrm = SUN_DIR;
            bounce_ray.o += hit_nrm * 0.001;
            // bounce_ray.o = floor(bounce_ray.o * VOXEL_SCL) / VOXEL_SCL;
            // bounce_ray.o += hit_nrm * 0.001;

            // VoxelSampleInfo chunk_info = get_voxel_sample_info_WORLD(hit_pos - hit_nrm * 0.01);
            // hit_voxel = unpack_voxel(sample_packed_voxel_WORLD(chunk_info.chunk_index, chunk_info.voxel_index));
            // hit_voxel.nrm = hit_nrm;
            // hit_voxel.nrm *= -1;
        } break;
        }
        bounce_ray.inv_nrm = 1.0 / bounce_ray.nrm;
#if RENDER_SHADING
        f32 shade = max(dot(bounce_ray.nrm, hit_nrm), 0.0);
#else
        f32 shade = max(dot(f32vec3(0, 0, 1), hit_nrm) * 0.5 + 0.5, 0.0);
#endif
        i32 temp_i32;
#if RENDER_SHADOWS
        TraceRecord bounce_trace_record = trace_scene(bounce_ray, temp_i32);
#endif
        f32vec3 voxel_p = f32vec3(i32vec3(hit_pos * VOXEL_SCL)) / VOXEL_SCL;
        switch (view_trace_record.material) {
        case 0:
        case 1: {
#if RENDER_SHADOWS
            shade *= max(f32(!bounce_trace_record.intersection_record.hit), 0.0);
#endif
        } break;
        case 2: {
            CapsulePoints c = SCENE.capsules[view_trace_record.object_i];
            f32mat2x2 rot_mat = f32mat2x2(f32vec2(-PLAYER.forward.y, PLAYER.forward.x), PLAYER.forward);
            f32vec3 local_pos = (hit_pos - c.p1) * 1;
            local_pos.xy = rot_mat * local_pos.xy;
            // f32vec3 result = f32vec3(0.3, 0.3, 0.3);
            f32vec3 result = f32vec3(0.50, 0.33, 0.30);
            f32vec2 uv = local_pos.xz;
            f32vec2 e1_uv = uv - f32vec2(0.1, 0.0);
            f32 e1 = f32(dot(e1_uv, e1_uv) > 0.001);
            f32vec2 e2_uv = uv + f32vec2(0.1, 0.0);
            f32 e2 = f32(dot(e2_uv, e2_uv) > 0.001);
            f32vec2 m_uv = uv + f32vec2(0.0, 0.04);
            f32 m = clamp(f32(dot(m_uv, m_uv) > 0.02) + f32(m_uv.y > -0.05), 0, 1);
            f32 face_fac = clamp(e1 * e2 * m + f32(local_pos.y < 0) * 10, 0, 1);
            f32 pants_fac = f32(local_pos.z > -0.6);
            f32 radial = atan(local_pos.y, local_pos.x) / 6.28 + 0.5;
            f32vec2 b_pocket_pos = f32vec2(abs(abs(radial - 0.5) - 0.25) - 0.075, (local_pos.z + 0.7) * 0.5);
            f32vec2 f_pocket_pos = f32vec2(abs(abs(radial - 0.5) - 0.25) - 0.200, (local_pos.z + 0.7) * 0.5);
            f32 b_pockets_fac = f32(b_pocket_pos.y < 0.0 && b_pocket_pos.x < 0.04 && b_pocket_pos.x > -0.04 && dot(b_pocket_pos, b_pocket_pos) < 0.003 && local_pos.y < 0);
            f32 f_pockets_fac = f32(f_pocket_pos.y < 0.0 && f_pocket_pos.x < 0.02 && f_pocket_pos.x > -0.06 && dot(f_pocket_pos, f_pocket_pos) < 0.003 && local_pos.y > 0);
            f32 belt_fac = f32(fract(radial * 20) < 0.8 && local_pos.z > -0.64 && local_pos.z < -0.62);
            f32vec2 shirt_uv = f32vec2(local_pos.x, (local_pos.z + 0.40) * 4);
            f32 shirt_fac = f32(shirt_uv.y > 0 || (dot(shirt_uv, shirt_uv) < 0.1 + local_pos.y * 0.1));
            result = mix(f32vec3(0.4, 0.08, 0.07), result, shirt_fac);
            result = mix(f32vec3(0.04, 0.04, 0.12), result, pants_fac);
            result = mix(result, f32vec3(0.03, 0.03, 0.10), f32(b_pockets_fac != 0.0 || f_pockets_fac != 0.0 || (f_pocket_pos.x > 0.045 && local_pos.z < -0.68 && local_pos.z > -1.5)));
            result = mix(f32vec3(0.0, 0.0, 0.0), result, face_fac);
            result = mix(result, f32vec3(0.04, 0.02, 0.01), belt_fac);
            col = result;
#if RENDER_SHADOWS
            shade *= max(f32(!bounce_trace_record.intersection_record.hit), 0.0);
#endif
        } break;
        case 3: {
            col = voxel_color(hit_pos, hit_nrm);
#if RENDER_SHADOWS
            shade *= max(f32(!bounce_trace_record.intersection_record.hit), 0.0);
#endif
        } break;
        case 4: {
            col = brush_voxel_color(hit_pos, hit_nrm);
#if RENDER_SHADOWS
            shade *= max(f32(!bounce_trace_record.intersection_record.hit), 0.0);
#endif
        } break;
        case 5: {
#if RENDER_SHADOWS
            shade *= max(f32(!bounce_trace_record.intersection_record.hit), 0.0);
#endif
        } break;
        }

        f32vec3 surface_col = col * (shade * SUN_COL * SUN_FACTOR + sample_sky_ambient(hit_nrm) * 1);
#if RENDER_FOG
        f32 fog_factor = clamp(exp(view_trace_record.intersection_record.dist * 0.01) * 0.02, 0, 1);
        f32vec3 fog_col = sample_sky_ambient(view_ray.nrm);
        col = mix(surface_col, fog_col, fog_factor);
#else
        col = surface_col;
#endif

        // col = hit_pos;
        // col = hit_nrm;
        // col = reflect(view_ray.nrm, hit_nrm);
        // col = refract(view_ray.nrm, hit_nrm, 1.0 / 1.4);
        // col = f32vec3(hit_dist);
        // col = view_ray.nrm;
        // col = bounce_ray.nrm;

        // Ray sun_ray;
        // sun_ray.o = hit_pos + view_trace_record.intersection_record.nrm * 0.001;
        // sun_ray.nrm = normalize(f32vec3(1, -2, 3));
        // sun_ray.inv_nrm = 1.0 / sun_ray.nrm;
        // TraceRecord sun_trace_record = trace_scene(sun_ray);
    }

    if (INPUT.settings.tool_id == GAME_TOOL_BRUSH) {
        {
            Sphere b;
            b.o = GLOBALS.edit_origin;
            b.r = 0.25;
            IntersectionRecord b_hit = intersect(view_ray, b);
            if (b_hit.hit) {
                col = mix(col, f32vec3(1, 0, 1), 0.8);
            }
        }
        if (get_flag(GPU_INPUT_FLAG_INDEX_SHOW_BRUSH_BOUNDING_BOX)) {
            BoundingBox b = VOXEL_BRUSH.box;
            IntersectionRecord b_hit = intersect(view_ray, b);
            f32vec3 b_col = f32vec3(0.5);
            b_col *= (dot(SUN_DIR, b_hit.nrm) * 0.5 + 0.5) * SUN_COL * SUN_FACTOR + sample_sky_ambient(b_hit.nrm);
            if (b_hit.hit) {
                col = mix(col, b_col, b_hit.dist < hit_dist ? 0.2 : 0.04);
            }
        }
    }

#endif

    i32vec2 crosshair_uv = abs(i32vec2(pixel_i) - i32vec2(frame_dim / 2));

    if (!get_flag(GPU_INPUT_FLAG_INDEX_PAUSED)) {
        if ((crosshair_uv.x < 1 && crosshair_uv.y < 10) ||
            (crosshair_uv.y < 1 && crosshair_uv.x < 10)) {
            col = f32vec3(1, 1, 1);
        }
    }

    f32vec3 prev_col = filmic_inv(
        imageLoad(
            daxa_access_Image(image2D, push_constant.image_id),
            i32vec2(pixel_i.xy))
            .rgb);

    prev_col = clamp(prev_col, f32vec3(0), f32vec3(1));
    col = clamp(col, f32vec3(0), f32vec3(1));

    f32 blending = INPUT.settings.frame_blending;

    imageStore(
        daxa_access_Image(image2D, push_constant.image_id),
        i32vec2(pixel_i.xy),
        f32vec4(filmic((col * (1.0 - blending) + prev_col * blending)), 1));
#endif
}
