#pragma once

#include <shared/shared.inl>

#include <utils/math.glsl>
#include <utils/gvox_model.glsl>

#define PLAYER deref(globals_ptr).player
f32vec3 create_view_pos(daxa_BufferPtr(GpuGlobals) globals_ptr) {
    return PLAYER.cam.pos;
}
#undef PLAYER

#define PLAYER deref(globals_ptr).player
f32vec3 create_view_dir(daxa_BufferPtr(GpuGlobals) globals_ptr, f32vec2 uv) {
    f32vec3 nrm = normalize(f32vec3(uv.x * PLAYER.cam.tan_half_fov, uv.y * PLAYER.cam.tan_half_fov, 1));
    nrm = PLAYER.cam.rot_mat * nrm;
    return nrm;
}
#undef PLAYER

f32 sdmap(f32vec3 p) {
    f32 value = MAX_SD;
    // value = sd_union(value, +sd_plane_x(p - f32vec3(-6.4, 0.0, 0.0)));
    // value = sd_union(value, -sd_plane_x(p - f32vec3(+6.4, 0.0, 0.0)));
    // value = sd_union(value, +sd_plane_y(p - f32vec3(0.0, -6.4, 0.0)));
    // value = sd_union(value, -sd_plane_y(p - f32vec3(0.0, +6.4, 0.0)));
    // value = sd_union(value, +sd_plane_z(p - f32vec3(0.0, 0.0, -6.4)));
    // value = sd_union(value, -sd_plane_z(p - f32vec3(0.0, 0.0, +6.4)));
    value = sd_union(value, sd_sphere(p - f32vec3(0.0, 2.0, 0.0), 0.5));
    return value;
}

f32vec3 sdmap_nrm(f32vec3 pos) {
    f32vec3 n = f32vec3(0.0);
    for (u32 i = 0; i < 4; ++i) {
        f32vec3 e = 0.5773 * (2.0 * f32vec3((((i + 3) >> 1) & 1), ((i >> 1) & 1), (i & 1)) - 1.0);
        n += e * sdmap(pos + 0.001 * e);
    }
    return normalize(n);
}

void trace_sphere_trace(in out f32vec3 ray_pos, f32vec3 ray_dir) {
    f32 t = 0.0;
    f32 final_dist = MAX_SD;
    for (u32 i = 0; i < 512 && t < MAX_SD; ++i) {
        f32vec3 p = ray_pos + ray_dir * t;
        f32 d = sdmap(p);
        if (abs(d) < (0.0001 * t)) {
            final_dist = t;
            break;
        }
        t += d;
    }
    ray_pos = ray_pos + ray_dir * t;
}

#define MODEL deref(model_ptr)
void trace_hierarchy_traversal(daxa_BufferPtr(GpuGvoxModel) model_ptr, in out f32vec3 ray_pos, f32vec3 ray_dir) {
    BoundingBox b;
    b.bound_min = f32vec3(0, 0, 0);
    b.bound_max = f32vec3(MODEL.extent_x, MODEL.extent_y, MODEL.extent_z) / VOXEL_SCL;

    intersect(ray_pos, ray_dir, f32vec3(1) / ray_dir, b);
    ray_pos += ray_dir * 0.01 / VOXEL_SCL;
    if (!inside(ray_pos, b)) {
        return;
    }

    f32vec3 delta_dist = f32vec3(
        ray_dir.x == 0 ? 1 : abs(1.0 / ray_dir.x),
        ray_dir.y == 0 ? 1 : abs(1.0 / ray_dir.y),
        ray_dir.z == 0 ? 1 : abs(1.0 / ray_dir.z));
    f32vec3 unit_space_ray_pos = ray_pos * VOXEL_SCL;
    i32vec3 tile_i = i32vec3(unit_space_ray_pos);
    i32vec3 tile_steps_i = i32vec3(0);
    i32vec3 ray_step = i32vec3(sign(ray_dir));
    f32vec3 start_dist = ((unit_space_ray_pos - tile_i) * -sign(ray_dir) + step(0.0, ray_dir)) * delta_dist;
    f32vec3 to_side_dist = start_dist;
    f32 dist = MAX_SD;
    bvec3 mask = bvec3(
        to_side_dist.x < to_side_dist.y && to_side_dist.x < to_side_dist.z,
        to_side_dist.y <= to_side_dist.x && to_side_dist.y < to_side_dist.z,
        to_side_dist.z <= to_side_dist.x && to_side_dist.z <= to_side_dist.y);

    if (sample_gvox_palette_voxel(model_ptr, f32vec3(tile_i) / VOXEL_SCL + 0.001, 0) != 0) {
        return;
    }

    for (u32 total_steps = 0; total_steps < 50000; ++total_steps) {
        if (!inside(f32vec3(tile_i) / VOXEL_SCL, b)) {
            break;
        }
        to_side_dist = delta_dist * tile_steps_i + start_dist;
        if (sample_gvox_palette_voxel(model_ptr, f32vec3(tile_i) / VOXEL_SCL + 0.001, 0) != 0) {
            dist = (dot(to_side_dist - delta_dist, f32vec3(mask)) + 0.001) / VOXEL_SCL;
            break;
        }
        mask = bvec3(
            to_side_dist.x < to_side_dist.y && to_side_dist.x < to_side_dist.z,
            to_side_dist.y <= to_side_dist.x && to_side_dist.y < to_side_dist.z,
            to_side_dist.z <= to_side_dist.x && to_side_dist.z <= to_side_dist.y);
        tile_steps_i += abs(ray_step) * i32vec3(mask);
        tile_i += ray_step * i32vec3(mask);
    }
    ray_pos = ray_pos + ray_dir * dist;
}
#undef MODEL

void trace(daxa_BufferPtr(GpuGvoxModel) model_ptr, in out f32vec3 ray_pos, f32vec3 ray_dir) {
    // trace_sphere_trace(ray_pos, ray_dir);
    trace_hierarchy_traversal(model_ptr, ray_pos, ray_dir);
}

f32vec3 scene_nrm(f32vec3 pos) {
    // return -sdmap_nrm(pos);
    vec3 d = fract(pos * VOXEL_SCL) - .5;
    vec3 ad = abs(d);
    float m = max(max(ad.x, ad.y), ad.z);
    return (abs(sign(ad - m)) - 1.) * sign(d);
}
