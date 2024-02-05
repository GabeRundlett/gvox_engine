#pragma once

#include <shared/app.inl>

#include <utils/math.glsl>
#include <voxels/pack_unpack.glsl>

float maxcomp(in vec3 p) { return max(p.x, max(p.y, p.z)); }
float sdBox(vec3 p, vec3 b) {
    vec3 di = abs(p) - b;
    float mc = maxcomp(di);
    return min(mc, length(max(di, 0.0)));
    // return length(p + b);
}

vec2 iBox(in vec3 ro, in vec3 rd, in vec3 rad) {
    vec3 m = 1.0 / rd;
    vec3 n = m * ro;
    vec3 k = abs(m) * rad;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
    return vec2(max(max(t1.x, t1.y), t1.z),
                min(min(t2.x, t2.y), t2.z));
}

vec4 map(in vec3 p) {
    float d = sdBox(p, vec3(1.0));
    vec4 res = vec4(d, 1.0, 0.0, 0.0);

    float s = 1.0;
    for (int m = 0; m < 2; m++) {
        vec3 a = mod(p * s, 2.0) - 1.0;
        s *= 3.0;
        vec3 r = abs(1.0 - 3.0 * abs(a));
        float da = max(r.x, r.y);
        float db = max(r.y, r.z);
        float dc = max(r.z, r.x);
        float c = (min(da, min(db, dc)) - 1.0) / s;

        if (c > d) {
            d = c;
            res = vec4(d, min(res.y, 0.2 * da * db * dc), (1.0 + float(m)) / 4.0, 0.0);
        }
    }

    return res;
}

vec3 map_nrm(in vec3 pos) {
    vec3 eps = vec3(.001, 0.0, 0.0);
    return normalize(vec3(
        map(pos + eps.xyy).x - map(pos - eps.xyy).x,
        map(pos + eps.yxy).x - map(pos - eps.yxy).x,
        map(pos + eps.yyx).x - map(pos - eps.yyx).x));
}

VoxelTraceResult voxel_trace(in VoxelTraceInfo info, in out daxa_f32vec3 ray_pos) {
    VoxelTraceResult result;

    result.dist = 0;
    result.nrm = -info.ray_dir;
    result.step_n = 0;
    result.voxel_data = PackedVoxel(0);
    result.vel = daxa_f32vec3(deref(info.ptrs.globals).offset - deref(info.ptrs.globals).prev_offset);

    daxa_f32vec3 offset = daxa_f32vec3(deref(info.ptrs.globals).offset);
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

    float t = tmin;
    int i = 0;
    for (; i < info.max_steps && t < tmax; i++) {
        vec4 h = map(ro + rd * t);
        if (abs(h.x) < (0.0001 * t)) {
            res = vec2(t, h.y);
            break;
        }
        t += h.x;
    }

    if (res.y != -1.0) {
        result.dist = res.x;
        ray_pos += rd * result.dist;

        Voxel voxel;
        voxel.color = vec3(0.5);
        voxel.material_type = 1;
        voxel.normal = map_nrm(ro + rd * t);
        voxel.roughness = 0.1;
        result.nrm = voxel.normal;

        result.voxel_data = pack_voxel(voxel);
    }

    ray_pos -= offset;
    return result;
}
