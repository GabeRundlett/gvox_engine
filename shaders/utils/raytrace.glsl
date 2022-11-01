#pragma once

#include <utils/math.glsl>

#if !defined(RAYTRACE_NO_VOXELS)
#include <utils/voxel.glsl>
#endif

#define MAX_DIST 10000.0

struct TraceRecord {
    IntersectionRecord intersection_record;
    f32vec3 color;
    u32 material;
    u32 object_i;
};
void default_init(out IntersectionRecord result) {
    result.hit = false;
    result.internal_fac = 1.0;
    result.dist = MAX_DIST;
    result.nrm = f32vec3(0, 0, 0);
}

#if !defined(RAYTRACE_NO_VOXELS)
Ray create_view_ray(f32vec2 uv) {
    Ray result;

    result.o = PLAYER.cam.pos;
    result.nrm = normalize(f32vec3(uv.x * PLAYER.cam.tan_half_fov, 1, -uv.y * PLAYER.cam.tan_half_fov));
    result.nrm = PLAYER.cam.rot_mat * result.nrm;
    result.inv_nrm = 1.0 / result.nrm;

    return result;
}
#endif

IntersectionRecord intersect(Ray ray, Sphere s) {
    IntersectionRecord result;
    default_init(result);

    f32vec3 so_r = ray.o - s.o;
    f32 a = dot(ray.nrm, ray.nrm);
    f32 b = 2.0f * dot(ray.nrm, so_r);
    f32 c = dot(so_r, so_r) - (s.r * s.r);
    f32 f = b * b - 4.0f * a * c;
    if (f < 0.0f)
        return result;
    result.internal_fac = (c > 0) ? 1.0 : -1.0;
    result.dist = (-b - sqrt(f) * result.internal_fac) / (2.0f * a);
    result.hit = result.dist > 0.0f;
    result.nrm = normalize(ray.o + ray.nrm * result.dist - s.o) * result.internal_fac;
    return result;
}
IntersectionRecord intersect(Ray ray, Box b) {
    IntersectionRecord result;
    default_init(result);

    f32 tx1 = (b.bound_min.x - ray.o.x) * ray.inv_nrm.x;
    f32 tx2 = (b.bound_max.x - ray.o.x) * ray.inv_nrm.x;
    f32 tmin = min(tx1, tx2);
    f32 tmax = max(tx1, tx2);
    f32 ty1 = (b.bound_min.y - ray.o.y) * ray.inv_nrm.y;
    f32 ty2 = (b.bound_max.y - ray.o.y) * ray.inv_nrm.y;
    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));
    f32 tz1 = (b.bound_min.z - ray.o.z) * ray.inv_nrm.z;
    f32 tz2 = (b.bound_max.z - ray.o.z) * ray.inv_nrm.z;
    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));

    result.hit = false;
    if (tmax >= tmin) {
        if (tmin > 0) {
            result.dist = tmin;
            result.hit = true;
        } else if (tmax > 0) {
            result.dist = tmax;
            result.hit = true;
            result.internal_fac = -1.0;
            tmin = tmax;
        }
    }

    b32 is_x = tmin == tx1 || tmin == tx2;
    b32 is_y = tmin == ty1 || tmin == ty2;
    b32 is_z = tmin == tz1 || tmin == tz2;

    if (is_z) {
        if (ray.nrm.z < 0) {
            result.nrm = f32vec3(0, 0, 1);
        } else {
            result.nrm = f32vec3(0, 0, -1);
        }
    } else if (is_y) {
        if (ray.nrm.y < 0) {
            result.nrm = f32vec3(0, 1, 0);
        } else {
            result.nrm = f32vec3(0, -1, 0);
        }
    } else {
        if (ray.nrm.x < 0) {
            result.nrm = f32vec3(1, 0, 0);
        } else {
            result.nrm = f32vec3(-1, 0, 0);
        }
    }
    result.nrm *= result.internal_fac;

    return result;
}
IntersectionRecord intersect(Ray ray, Capsule cap) {
    IntersectionRecord result;
    default_init(result);

    f32vec3 ba = cap.p1 - cap.p0;
    f32vec3 oa = ray.o - cap.p0;
    f32 baba = dot(ba, ba);
    f32 bard = dot(ba, ray.nrm);
    f32 baoa = dot(ba, oa);
    f32 rdoa = dot(ray.nrm, oa);
    f32 oaoa = dot(oa, oa);
    f32 a = baba - bard * bard;
    f32 b = baba * rdoa - baoa * bard;
    f32 c = baba * oaoa - baoa * baoa - cap.r * cap.r * baba;
    f32 h = b * b - a * c;
    if (h >= 0.) {
        f32 t = (-b - sqrt(h)) / a;
        f32 d = MAX_DIST;
        f32 y = baoa + t * bard;
        if (y > 0. && y < baba) {
            d = t;
        } else {
            f32vec3 oc = (y <= 0.) ? oa : ray.o - cap.p1;
            b = dot(ray.nrm, oc);
            c = dot(oc, oc) - cap.r * cap.r;
            h = b * b - c;
            if (h > 0.0) {
                d = -b - sqrt(h);
            }
        }
        // if (d >= distBound.x && d <= distBound.y) {
        cap.p0 = ray.o + ray.nrm * d - cap.p0;
        f32 h = clamp(dot(cap.p0, ba) / dot(ba, ba), 0.0, 1.0);
        result.nrm = (cap.p0 - h * ba) / cap.r;
        result.dist = d;
        result.hit = d != MAX_DIST && d >= 0;
        // return d;
        // }
    }
    return result;
}

#if !defined(RAYTRACE_NO_VOXELS)
u32 sample_lod(f32vec3 p, in out u32 chunk_index) {
    VoxelWorldSampleInfo chunk_info = get_voxel_world_sample_info(p);
    chunk_index = chunk_info.chunk_index;

    u32 lod_index_x2 = uniformity_lod_index(2)(chunk_info.inchunk_voxel_i / 2);
    u32 lod_mask_x2 = uniformity_lod_mask(chunk_info.inchunk_voxel_i / 2);
    u32 lod_index_x4 = uniformity_lod_index(4)(chunk_info.inchunk_voxel_i / 4);
    u32 lod_mask_x4 = uniformity_lod_mask(chunk_info.inchunk_voxel_i / 4);
    u32 lod_index_x8 = uniformity_lod_index(8)(chunk_info.inchunk_voxel_i / 8);
    u32 lod_mask_x8 = uniformity_lod_mask(chunk_info.inchunk_voxel_i / 8);
    u32 lod_index_x16 = uniformity_lod_index(16)(chunk_info.inchunk_voxel_i / 16);
    u32 lod_mask_x16 = uniformity_lod_mask(chunk_info.inchunk_voxel_i / 16);
    u32 lod_index_x32 = uniformity_lod_index(32)(chunk_info.inchunk_voxel_i / 32);
    u32 lod_mask_x32 = uniformity_lod_mask(chunk_info.inchunk_voxel_i / 32);
    u32 lod_index_x64 = uniformity_lod_index(64)(chunk_info.inchunk_voxel_i / 64);
    u32 lod_mask_x64 = uniformity_lod_mask(chunk_info.inchunk_voxel_i / 64);

    u32 chunk_edit_stage = VOXEL_WORLD.chunks_genstate[chunk_index].edit_stage;
    if (chunk_edit_stage != 2 && chunk_edit_stage != 3)
        return 7;
    u32 id = sample_voxel_id(chunk_index, chunk_info.inchunk_voxel_i);
    if (id != BlockID_Air && id != BlockID_Debug)
        return 0;
    if (voxel_uniformity_lod_nonuniform(2)(chunk_index, lod_index_x2, lod_mask_x2))
        return 1;
    if (voxel_uniformity_lod_nonuniform(4)(chunk_index, lod_index_x4, lod_mask_x4))
        return 2;
    if (voxel_uniformity_lod_nonuniform(8)(chunk_index, lod_index_x8, lod_mask_x8))
        return 3;
    if (voxel_uniformity_lod_nonuniform(16)(chunk_index, lod_index_x16, lod_mask_x16))
        return 4;
    if (voxel_uniformity_lod_nonuniform(32)(chunk_index, lod_index_x32, lod_mask_x32))
        return 5;
    if (voxel_uniformity_lod_nonuniform(64)(chunk_index, lod_index_x64, lod_mask_x64))
        return 6;

    return 7;
}
IntersectionRecord dda(Ray ray, in out u32 chunk_index, in out i32 x1_steps) {
    IntersectionRecord result;
    default_init(result);
    result.dist = 0;

    const u32 max_steps = WORLD_BLOCK_NX + WORLD_BLOCK_NY + WORLD_BLOCK_NZ;
    f32vec3 delta = f32vec3(
        ray.nrm.x == 0.0 ? 3.0 * max_steps : abs(ray.inv_nrm.x),
        ray.nrm.y == 0.0 ? 3.0 * max_steps : abs(ray.inv_nrm.y),
        ray.nrm.z == 0.0 ? 3.0 * max_steps : abs(ray.inv_nrm.z));
    u32 lod = sample_lod(ray.o, chunk_index);
    if (lod == 0) {
        result.hit = true;
        return result;
    }
    f32 cell_size = f32(1l << (lod - 1)) / VOXEL_SCL;
    f32vec3 t_start;
    if (ray.nrm.x < 0) {
        t_start.x = (ray.o.x / cell_size - floor(ray.o.x / cell_size)) * cell_size * delta.x;
    } else {
        t_start.x = (ceil(ray.o.x / cell_size) - ray.o.x / cell_size) * cell_size * delta.x;
    }
    if (ray.nrm.y < 0) {
        t_start.y = (ray.o.y / cell_size - floor(ray.o.y / cell_size)) * cell_size * delta.y;
    } else {
        t_start.y = (ceil(ray.o.y / cell_size) - ray.o.y / cell_size) * cell_size * delta.y;
    }
    if (ray.nrm.z < 0) {
        t_start.z = (ray.o.z / cell_size - floor(ray.o.z / cell_size)) * cell_size * delta.z;
    } else {
        t_start.z = (ceil(ray.o.z / cell_size) - ray.o.z / cell_size) * cell_size * delta.z;
    }
    f32 t_curr = min(min(t_start.x, t_start.y), t_start.z);
    f32vec3 current_pos;
    f32vec3 t_next = t_start;
    b32 outside_bounds = false;
    u32 side = 0;
    for (x1_steps = 0; x1_steps < max_steps; ++x1_steps) {
        current_pos = ray.o + ray.nrm * t_curr;
        if (inside(current_pos + ray.nrm * 0.001, VOXEL_WORLD.box) == false) {
            outside_bounds = true;
            result.hit = false;
            break;
        }
        lod = sample_lod(current_pos, chunk_index);
        if (lod == 0) {
            result.hit = true;
            if (t_next.x < t_next.y) {
                if (t_next.x < t_next.z) {
                    side = 0;
                } else {
                    side = 2;
                }
            } else {
                if (t_next.y < t_next.z) {
                    side = 1;
                } else {
                    side = 2;
                }
            }
            break;
        }
        cell_size = f32(1l << (lod - 1)) / VOXEL_SCL;

        t_next = (0.5 + sign(ray.nrm) * (0.5 - fract(current_pos / cell_size))) * cell_size * delta;
        t_curr += (min(min(t_next.x, t_next.y), t_next.z) + 0.001 / 8);
    }
    result.dist = t_curr;
    switch (side) {
    case 0: result.nrm = f32vec3(ray.nrm.x < 0 ? 1 : -1, 0, 0); break;
    case 1: result.nrm = f32vec3(0, ray.nrm.y < 0 ? 1 : -1, 0); break;
    case 2: result.nrm = f32vec3(0, 0, ray.nrm.z < 0 ? 1 : -1); break;
    }
    return result;
}

IntersectionRecord intersect_voxels(Ray ray) {
    IntersectionRecord result;
    default_init(result);
    result = intersect(ray, VOXEL_WORLD.box);

    if (inside(ray.o, VOXEL_WORLD.box)) {
        result.dist = 0;
        result.hit = true;
    } else {
        result = intersect(ray, VOXEL_WORLD.box);
        result.dist += 0.0001;
    }

    Ray dda_ray = ray;
    dda_ray.o = ray.o + ray.nrm * result.dist;
    u32 chunk_index;

    if (result.hit && sample_lod(dda_ray.o, chunk_index) != 0) {
        f32 prev_dist = result.dist;
        i32 x1_steps = 0;
        result = dda(dda_ray, chunk_index, x1_steps);
        result.dist += prev_dist;
    }

    return result;
}
IntersectionRecord intersect_voxels(Ray ray, in out i32 x1_steps) {
    IntersectionRecord result;
    default_init(result);
    result = intersect(ray, VOXEL_WORLD.box);

    if (inside(ray.o, VOXEL_WORLD.box)) {
        result.dist = 0;
        result.hit = true;
    } else {
        result = intersect(ray, VOXEL_WORLD.box);
        result.dist += 0.0001;
    }

    Ray dda_ray = ray;
    dda_ray.o = ray.o + ray.nrm * result.dist;
    u32 chunk_index;

    if (result.hit && sample_lod(dda_ray.o, chunk_index) != 0) {
        f32 prev_dist = result.dist;
        result = dda(dda_ray, chunk_index, x1_steps);
        result.dist += prev_dist;
    }

    return result;
}

u32 sample_brush_lod(f32vec3 p, in out u32 chunk_index) {
    VoxelWorldSampleInfo chunk_info = get_voxel_brush_sample_info(p);
    chunk_index = chunk_info.chunk_index;

    u32 lod_index_x2 = uniformity_lod_index(2)(chunk_info.inchunk_voxel_i / 2);
    u32 lod_mask_x2 = uniformity_lod_mask(chunk_info.inchunk_voxel_i / 2);
    u32 lod_index_x4 = uniformity_lod_index(4)(chunk_info.inchunk_voxel_i / 4);
    u32 lod_mask_x4 = uniformity_lod_mask(chunk_info.inchunk_voxel_i / 4);
    u32 lod_index_x8 = uniformity_lod_index(8)(chunk_info.inchunk_voxel_i / 8);
    u32 lod_mask_x8 = uniformity_lod_mask(chunk_info.inchunk_voxel_i / 8);
    u32 lod_index_x16 = uniformity_lod_index(16)(chunk_info.inchunk_voxel_i / 16);
    u32 lod_mask_x16 = uniformity_lod_mask(chunk_info.inchunk_voxel_i / 16);
    u32 lod_index_x32 = uniformity_lod_index(32)(chunk_info.inchunk_voxel_i / 32);
    u32 lod_mask_x32 = uniformity_lod_mask(chunk_info.inchunk_voxel_i / 32);
    u32 lod_index_x64 = uniformity_lod_index(64)(chunk_info.inchunk_voxel_i / 64);
    u32 lod_mask_x64 = uniformity_lod_mask(chunk_info.inchunk_voxel_i / 64);

    // u32 chunk_edit_stage = VOXEL_WORLD.chunks_genstate[chunk_index].edit_stage;
    // if (chunk_edit_stage != 2 && chunk_edit_stage != 3)
    //     return 7;
    u32 id = sample_brush_voxel_id(chunk_index, chunk_info.inchunk_voxel_i);
    if (id != BlockID_Air && id != BlockID_Debug)
        return 0;
    if (brush_voxel_uniformity_lod_nonuniform(2)(chunk_index, lod_index_x2, lod_mask_x2))
        return 1;
    if (brush_voxel_uniformity_lod_nonuniform(4)(chunk_index, lod_index_x4, lod_mask_x4))
        return 2;
    if (brush_voxel_uniformity_lod_nonuniform(8)(chunk_index, lod_index_x8, lod_mask_x8))
        return 3;
    if (brush_voxel_uniformity_lod_nonuniform(16)(chunk_index, lod_index_x16, lod_mask_x16))
        return 4;
    if (brush_voxel_uniformity_lod_nonuniform(32)(chunk_index, lod_index_x32, lod_mask_x32))
        return 5;
    if (brush_voxel_uniformity_lod_nonuniform(64)(chunk_index, lod_index_x64, lod_mask_x64))
        return 6;

    return 7;
}

IntersectionRecord brush_dda(Ray ray, in out u32 chunk_index, in out i32 x1_steps) {
    Box brush_box = VOXEL_BRUSH.box;
    brush_box.bound_min -= VOXEL_BRUSH.box.bound_min;
    brush_box.bound_max -= VOXEL_BRUSH.box.bound_min;

    IntersectionRecord result;
    default_init(result);
    result.dist = 0;

    const u32 max_steps = BRUSH_BLOCK_NX + BRUSH_BLOCK_NY + BRUSH_BLOCK_NZ;
    f32vec3 delta = f32vec3(
        ray.nrm.x == 0.0 ? 3.0 * max_steps : abs(ray.inv_nrm.x),
        ray.nrm.y == 0.0 ? 3.0 * max_steps : abs(ray.inv_nrm.y),
        ray.nrm.z == 0.0 ? 3.0 * max_steps : abs(ray.inv_nrm.z));
    u32 lod = sample_brush_lod(ray.o, chunk_index);
    if (lod == 0) {
        result.hit = true;
        return result;
    }
    f32 cell_size = f32(1l << (lod - 1)) / VOXEL_SCL;
    f32vec3 t_start;
    if (ray.nrm.x < 0) {
        t_start.x = (ray.o.x / cell_size - floor(ray.o.x / cell_size)) * cell_size * delta.x;
    } else {
        t_start.x = (ceil(ray.o.x / cell_size) - ray.o.x / cell_size) * cell_size * delta.x;
    }
    if (ray.nrm.y < 0) {
        t_start.y = (ray.o.y / cell_size - floor(ray.o.y / cell_size)) * cell_size * delta.y;
    } else {
        t_start.y = (ceil(ray.o.y / cell_size) - ray.o.y / cell_size) * cell_size * delta.y;
    }
    if (ray.nrm.z < 0) {
        t_start.z = (ray.o.z / cell_size - floor(ray.o.z / cell_size)) * cell_size * delta.z;
    } else {
        t_start.z = (ceil(ray.o.z / cell_size) - ray.o.z / cell_size) * cell_size * delta.z;
    }
    f32 t_curr = min(min(t_start.x, t_start.y), t_start.z);
    f32vec3 current_pos;
    f32vec3 t_next = t_start;
    b32 outside_bounds = false;
    u32 side = 0;
    for (x1_steps = 0; x1_steps < max_steps; ++x1_steps) {
        current_pos = ray.o + ray.nrm * t_curr;
        if (inside(current_pos + ray.nrm * 0.001, brush_box) == false) {
            outside_bounds = true;
            result.hit = false;
            break;
        }
        lod = sample_brush_lod(current_pos, chunk_index);
        if (lod == 0) {
            result.hit = true;
            if (t_next.x < t_next.y) {
                if (t_next.x < t_next.z) {
                    side = 0;
                } else {
                    side = 2;
                }
            } else {
                if (t_next.y < t_next.z) {
                    side = 1;
                } else {
                    side = 2;
                }
            }
            break;
        }
        cell_size = f32(1l << (lod - 1)) / VOXEL_SCL;

        t_next = (0.5 + sign(ray.nrm) * (0.5 - fract(current_pos / cell_size))) * cell_size * delta;
        t_curr += (min(min(t_next.x, t_next.y), t_next.z) + 0.001 / 8);
    }
    result.dist = t_curr;
    switch (side) {
    case 0: result.nrm = f32vec3(ray.nrm.x < 0 ? 1 : -1, 0, 0); break;
    case 1: result.nrm = f32vec3(0, ray.nrm.y < 0 ? 1 : -1, 0); break;
    case 2: result.nrm = f32vec3(0, 0, ray.nrm.z < 0 ? 1 : -1); break;
    }
    return result;
}

IntersectionRecord intersect_brush_voxels(Ray ray) {
    IntersectionRecord result;
    Box brush_box = VOXEL_BRUSH.box;

    default_init(result);
    result = intersect(ray, brush_box);

    if (inside(ray.o, brush_box)) {
        result.dist = 0;
        result.hit = true;
    } else {
        result = intersect(ray, brush_box);
        result.dist += 0.0001;
    }

    Ray dda_ray = ray;
    dda_ray.o = ray.o + ray.nrm * result.dist - VOXEL_BRUSH.box.bound_min;
    u32 chunk_index;

    if (result.hit && sample_brush_lod(dda_ray.o, chunk_index) != 0) {
        f32 prev_dist = result.dist;
        i32 x1_steps = 0;
        result = brush_dda(dda_ray, chunk_index, x1_steps);
        result.dist += prev_dist;
    }

    return result;
}
#endif
