#pragma once

#include <shared/app.inl>

#include <utils/math.glsl>

vec2 opU(vec2 d1, vec2 d2) {
    return (d1.x < d2.x) ? d1 : d2;
}

const float floor_z = -1.0;

vec2 map(in out VoxelBufferPtrs ptrs, in vec3 pos) {
    vec2 res = vec2(pos.z - floor_z, 0.0);
    res = opU(res, vec2(sd_sphere(pos - vec3(+1.0, +1.0, 0.25 + floor_z), 0.25), 1.0));
    // res = opU(res, vec2(sd_ellipsoid(pos - vec3(+1.0, +2.0, 0.25 + floor_z), f32vec3(0.1, 0.2, 0.25)), 1.0));
    res = opU(res, vec2(sd_box(apply_inv_rotation(pos - deref(ptrs.globals).box0.pos, deref(ptrs.globals).box0.rot), deref(ptrs.globals).box0.size), 2.0));
    res = opU(res, vec2(sd_box(apply_inv_rotation(pos - deref(ptrs.globals).box1.pos, deref(ptrs.globals).box1.rot), deref(ptrs.globals).box1.size), 3.0));
    // res = opU(res, vec2(sd_box_frame(pos - vec3(+3.0, +1.0, 0.25 + floor_z), f32vec3(0.1, 0.2, 0.25), 0.02), 1.0));
    // res = opU(res, vec2(sd_cylinder(pos - vec3(+2.0, +2.0, 0.25 + floor_z), 0.25, 0.25), 1.0));
    // res = opU(res, vec2(sd_torus(pos.yzx - vec3(+3.0, 0.25 + floor_z, +1.0), f32vec2(0.18, 0.07)), 1.0));
    return res;
}

vec3 map_col(in out VoxelBufferPtrs ptrs, vec3 pos, int id) {
    switch (id) {
    case 0:
#if TRACE_PRIMARY_COMPUTE
        return texture(daxa_sampler2D(debug_texture, deref(gpu_input).sampler_llr), pos.xy).rgb;
#else
        return f32vec3(0.2) + float(int(floor(pos.x) + floor(pos.y)) & 1) * 0.05;
#endif
    case 1:
    case 2:
    case 3:
        return f32vec3(0.6, 0.04, 0.12) + float(int(floor(pos.x * 8.0) + floor(pos.y * 8.0) + floor(pos.z * 8.0)) & 1) * 0.05;
    default:
        return f32vec3(0.0);
    }
}

vec3 map_nrm(in out VoxelBufferPtrs ptrs, in vec3 pos) {
    vec3 n = vec3(0.0);
    for (int i = 0; i < 4; i++) {
        vec3 e = 0.5773 * (2.0 * vec3((((i + 3) >> 1) & 1), ((i >> 1) & 1), (i & 1)) - 1.0);
        n += e * map(ptrs, pos + 0.0005 * e).x;
    }
    return normalize(n);
}

bool hit_box(f32vec3 ro, f32vec3 rd, f32vec3 box_pos, f32vec3 box_size) {
    float t = 0;

    for (int i = 0; i < MAX_STEPS && t < MAX_DIST; i++) {
        vec3 pos = ro + rd * t;
        float h = sd_box(pos - box_pos, box_size);
        if (abs(h) < (0.001 * t)) {
            return true;
        }
        t += h;
    }

    return false;
}

VoxelTraceResult voxel_trace(in VoxelTraceInfo info, in out f32vec3 ray_pos) {
    VoxelTraceResult result;

    result.dist = 0;
    result.nrm = -info.ray_dir;
    result.step_n = 0;
    result.voxel_data = 0;
    result.vel = f32vec3(deref(info.ptrs.globals).offset - deref(info.ptrs.globals).prev_offset);

    f32vec3 offset = f32vec3(deref(info.ptrs.globals).offset);
    ray_pos += offset;

#if TRACE_DEPTH_PREPASS_COMPUTE
    ray_pos -= offset;
    return result;
#endif

    if (info.extend_to_max_dist) {
        result.dist = MAX_DIST;
    }

    ray_pos += info.ray_dir * 0.001 / VOXEL_SCL;

    vec3 ro = ray_pos;
    vec3 rd = info.ray_dir;

    vec2 res = vec2(-1.0, -1.0);

    float tmin = 0.0;
    float tmax = info.max_dist;

    // raytrace floor plane
    float tp1 = (floor_z - ro.z) / rd.z;
    if (tp1 > 0.0) {
        tmax = min(tmax, tp1);
        res = vec2(tp1, 0.0);
    }

    float t = tmin;
    int i = 0;
    for (; i < info.max_steps && t < tmax; i++) {
        vec2 h = map(info.ptrs, ro + rd * t);
        if (abs(h.x) < (0.0001 * t)) {
            res = vec2(t, h.y);
            break;
        }
        t += h.x;
    }

    if (res.y != -1.0) {
        result.dist = res.x;
        ray_pos += rd * result.dist;

        vec3 obj_space = ray_pos;
        result.nrm = map_nrm(info.ptrs, ro + rd * t);
        if (res.y == 2.0) {
            result.vel += deref(info.ptrs.globals).box0.pos - deref(info.ptrs.globals).box0.prev_pos;
            obj_space -= deref(info.ptrs.globals).box0.pos;
        }
        if (res.y == 3.0) {
            result.vel += deref(info.ptrs.globals).box1.pos - deref(info.ptrs.globals).box1.prev_pos;
            obj_space -= deref(info.ptrs.globals).box1.pos;
        }
        f32vec3 col = map_col(info.ptrs, obj_space, int(res.y));
        u32 id = 1;

        result.voxel_data = f32vec4_to_uint_rgba8(f32vec4(col, 0.0)) | (id << 0x18);
    }

    ray_pos -= offset;
    return result;
}
