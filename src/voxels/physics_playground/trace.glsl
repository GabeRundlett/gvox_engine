#pragma once

#include <utilities/gpu/math.glsl>
#include <voxels/pack_unpack.glsl>

vec2 opU(vec2 d1, vec2 d2) {
    return (d1.x < d2.x) ? d1 : d2;
}

const float floor_z = -1.0;

vec2 map(in out VoxelBufferPtrs ptrs, in vec3 pos) {
    vec2 res = vec2(pos.z - floor_z, 0.0);

    uint body_n = min(RIGID_BODY_MAX_N, deref(ptrs.globals).rigid_body_n);
    RigidBody body;
    for (uint i = 0; i < body_n; ++i) {
        body = deref(ptrs.globals).rigid_bodies[i];
        float mat_id = i + 1;
        switch (body.flags & RIGID_BODY_FLAG_MASK_SHAPE_TYPE) {
        case RIGID_BODY_SHAPE_TYPE_SPHERE: res = opU(res, vec2(sd_sphere(apply_inv_rotation(pos - body.pos, body.rot), body.size.x), mat_id)); break;
        case RIGID_BODY_SHAPE_TYPE_BOX: res = opU(res, vec2(sd_box(apply_inv_rotation(pos - body.pos, body.rot), body.size * 0.5), mat_id)); break;
        }
    }

    return res;
}

vec3 map_col(in out VoxelBufferPtrs ptrs, vec3 pos, int id) {
    RigidBody body;
    switch (id) {
    case 0:
#if 0 // TracePrimaryComputeShader
        return texture(daxa_sampler2D(debug_texture, g_sampler_llr), pos.xy).rgb;
#else
        return vec3(0.2) + float(int(floor(pos.x) + floor(pos.y)) & 1) * 0.05;
#endif
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
        body = deref(ptrs.globals).rigid_bodies[id - 1];
        pos = apply_inv_rotation(pos - body.pos, body.rot);
        return vec3(0.6, 0.04, 0.12) + float(int(floor(pos.x * 8.0) + floor(pos.y * 8.0) + floor(pos.z * 8.0)) & 1) * 0.05;
    default:
        return vec3(0.0);
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

bool hit_box(vec3 ro, vec3 rd, vec3 box_pos, vec3 box_size) {
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

VoxelTraceResult voxel_trace(in VoxelTraceInfo info, in out vec3 ray_pos) {
    VoxelTraceResult result;

    result.dist = 0;
    result.nrm = -info.ray_dir;
    result.step_n = 0;
    result.voxel_data = PackedVoxel(0);
    result.vel = vec3(deref(info.ptrs.globals).offset - deref(info.ptrs.globals).prev_offset);

    vec3 offset = vec3(deref(info.ptrs.globals).offset);
    ray_pos += offset;

#if TraceDepthPrepassComputeShader
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

        Voxel voxel;
        voxel.color = map_col(info.ptrs, obj_space, int(res.y));
        voxel.material_type = 1;
        voxel.normal = map_nrm(info.ptrs, ro + rd * t);
        voxel.roughness = 0.1;
        result.nrm = voxel.normal;

        result.voxel_data = pack_voxel(voxel);
    }

    ray_pos -= offset;
    return result;
}
