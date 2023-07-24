#pragma once

#include <shared/shared.inl>

#include <utils/math.glsl>
#include <utils/voxels.glsl>

f32vec3 create_view_pos(daxa_RWBufferPtr(GpuGlobals) globals) {
    return deref(globals).player.cam.pos;
}

f32vec3 create_view_dir(daxa_RWBufferPtr(GpuGlobals) globals, f32vec2 uv) {
    f32vec3 nrm = normalize(f32vec3(uv.x * deref(globals).player.cam.tan_half_fov, uv.y * deref(globals).player.cam.tan_half_fov, 1));
    nrm = deref(globals).player.cam.rot_mat * nrm;
    return nrm;
}

f32 sdmap(f32vec3 p) {
    f32 value = MAX_DIST;
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
    f32 final_dist = MAX_DIST;
    for (u32 i = 0; i < 512 && t < MAX_DIST; ++i) {
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

struct VoxelTraceInfo {
    daxa_RWBufferPtr(VoxelMallocPageAllocator) allocator;
    daxa_BufferPtr(VoxelLeafChunk) voxel_chunks_ptr;
    u32vec3 chunk_n;
    f32vec3 ray_dir;
    u32 max_steps;
    f32 max_dist;
    f32 angular_coverage;
    bool extend_to_max_dist;
};

struct VoxelTraceResult {
    f32 dist;
    f32vec3 nrm;
    u32 step_n;
    u32 voxel_data;
};

VoxelTraceResult trace_hierarchy_traversal(in VoxelTraceInfo info, in out f32vec3 ray_pos) {
    VoxelTraceResult result;

    BoundingBox b;
    b.bound_min = f32vec3(0, 0, 0);
    b.bound_max = f32vec3(info.chunk_n) * CHUNK_WORLDSPACE_SIZE;

    intersect(ray_pos, info.ray_dir, f32vec3(1) / info.ray_dir, b);
    ray_pos += info.ray_dir * 0.01 / VOXEL_SCL;
    if (!inside(ray_pos, b)) {
        if (info.extend_to_max_dist) {
            result.dist = info.max_dist;
        } else {
            result.dist = 0.0;
        }
        return result;
    }

    f32vec3 delta = f32vec3(
        info.ray_dir.x == 0 ? 3.0 * info.max_steps : abs(1.0 / info.ray_dir.x),
        info.ray_dir.y == 0 ? 3.0 * info.max_steps : abs(1.0 / info.ray_dir.y),
        info.ray_dir.z == 0 ? 3.0 * info.max_steps : abs(1.0 / info.ray_dir.z));
    u32 lod = sample_lod(info.allocator, info.voxel_chunks_ptr, info.chunk_n, ray_pos, result.voxel_data);
    if (lod == 0) {
        if (info.extend_to_max_dist) {
#if TRACE_SECONDARY_COMPUTE
            result.dist = 0.0;
#else
            result.dist = info.max_dist;
#endif
        } else {
            result.dist = 0.0;
        }
        return result;
    }
    f32 cell_size = f32(1l << (lod - 1)) / VOXEL_SCL;
    f32vec3 t_start;
    if (info.ray_dir.x < 0) {
        t_start.x = (ray_pos.x / cell_size - floor(ray_pos.x / cell_size)) * cell_size * delta.x;
    } else {
        t_start.x = (ceil(ray_pos.x / cell_size) - ray_pos.x / cell_size) * cell_size * delta.x;
    }
    if (info.ray_dir.y < 0) {
        t_start.y = (ray_pos.y / cell_size - floor(ray_pos.y / cell_size)) * cell_size * delta.y;
    } else {
        t_start.y = (ceil(ray_pos.y / cell_size) - ray_pos.y / cell_size) * cell_size * delta.y;
    }
    if (info.ray_dir.z < 0) {
        t_start.z = (ray_pos.z / cell_size - floor(ray_pos.z / cell_size)) * cell_size * delta.z;
    } else {
        t_start.z = (ceil(ray_pos.z / cell_size) - ray_pos.z / cell_size) * cell_size * delta.z;
    }
    f32 t_curr = min(min(t_start.x, t_start.y), t_start.z);
    f32vec3 current_pos = ray_pos;
    f32vec3 t_next = t_start;
    result.dist = info.max_dist;
    for (result.step_n = 0; result.step_n < info.max_steps; ++result.step_n) {
        current_pos = ray_pos + info.ray_dir * t_curr;
        if (!inside(current_pos + info.ray_dir * 0.001, b) || t_curr > info.max_dist) {
            break;
        }
        lod = sample_lod(info.allocator, info.voxel_chunks_ptr, info.chunk_n, current_pos, result.voxel_data);
#if defined(TRACE_DEPTH_PREPASS_COMPUTE)
        bool hit_surface = lod < clamp(sqrt(t_curr * info.angular_coverage), 1, 7);
#else
        bool hit_surface = lod == 0;
#endif
        if (hit_surface) {
            result.nrm = sign(info.ray_dir) * (sign(t_next - min(min(t_next.x, t_next.y), t_next.z).xxx) - 1);
            result.dist = t_curr;
            break;
        }
        cell_size = f32(1l << (lod - 1)) / VOXEL_SCL;
        t_next = (0.5 + sign(info.ray_dir) * (0.5 - fract(current_pos / cell_size))) * cell_size * delta;
        // HACK to mitigate fp imprecision...
        t_next += 0.0001 * (sign(info.ray_dir) * -0.5 + 0.5);
        // t_curr += (min(min(t_next.x, t_next.y), t_next.z));
        t_curr += (min(min(t_next.x, t_next.y), t_next.z) + 0.0001 / VOXEL_SCL);
    }
    if (info.extend_to_max_dist) {
        t_curr = result.dist;
    }
    ray_pos = ray_pos + info.ray_dir * t_curr;

    result.dist = t_curr;
    return result;
}

void trace_sparse(daxa_BufferPtr(VoxelLeafChunk) voxel_chunks_ptr, u32vec3 chunk_n, in out f32vec3 ray_pos, f32vec3 ray_dir, u32 max_steps) {
    BoundingBox b;
    b.bound_min = f32vec3(0, 0, 0);
    b.bound_max = f32vec3(chunk_n) * CHUNK_WORLDSPACE_SIZE;

    intersect(ray_pos, ray_dir, f32vec3(1) / ray_dir, b);
    ray_pos += ray_dir * 0.001 / VOXEL_SCL;
    if (!inside(ray_pos, b)) {
        ray_pos = ray_pos + ray_dir * MAX_DIST;
        return;
    }

#define STEP(N)                                                                                        \
    for (u32 i = 0; i < max_steps; ++i) {                                                              \
        ray_pos += ray_dir * N / 8;                                                                    \
        if (!inside(f32vec3(ray_pos), b)) {                                                            \
            ray_pos += ray_dir * MAX_DIST;                                                             \
            return;                                                                                    \
        }                                                                                              \
        b32 present = sample_lod_presence(N)(voxel_chunks_ptr, chunk_n, u32vec3(ray_pos * VOXEL_SCL)); \
        if (present) {                                                                                 \
            return;                                                                                    \
        }                                                                                              \
    }
    STEP(2)
    STEP(4)
    STEP(8)
    STEP(16)
    STEP(32)
    STEP(64)
    ray_pos += ray_dir * MAX_DIST;
}

f32vec3 scene_nrm(daxa_RWBufferPtr(VoxelMallocPageAllocator) allocator, daxa_BufferPtr(VoxelLeafChunk) voxel_chunks_ptr, u32vec3 chunk_n, f32vec3 pos) {
#if 0
    const i32 RANGE = 1;
    f32vec3 result = f32vec3(0);
    for (i32 zi = -RANGE; zi <= RANGE; ++zi) {
        for (i32 yi = -RANGE; yi <= RANGE; ++yi) {
            for (i32 xi = -RANGE; xi <= RANGE; ++xi) {
                u32 voxel = sample_voxel_chunk(allocator, voxel_chunks_ptr, chunk_n, u32vec3(pos * VOXEL_SCL + f32vec3(xi, yi, zi)));
                u32 id = voxel >> 0x18;
                if (voxel != 0) {
                    result += f32vec3(xi, yi, zi);
                }
            }
        }
    }
    return -normalize(result);
#else
    // return -sdmap_nrm(pos);
    f32vec3 d = fract(pos * VOXEL_SCL) - .5;
    f32vec3 ad = abs(d);
    f32 m = max(max(ad.x, ad.y), ad.z);
    return -(abs(sign(ad - m)) - 1.) * sign(d);
#endif
}
