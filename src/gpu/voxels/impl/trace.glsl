#pragma once

#include <shared/app.inl>

#include <utils/math.glsl>
#include <voxels/impl/voxels.glsl>

VoxelTraceResult voxel_trace(in VoxelTraceInfo info, in out f32vec3 ray_pos) {
    VoxelTraceResult result;

    u32vec3 chunk_n = u32vec3(1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);

    BoundingBox b;
    b.bound_min = f32vec3(0, 0, 0);
    b.bound_max = f32vec3(chunk_n) * CHUNK_WORLDSPACE_SIZE;

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
    u32 lod = sample_lod(info.ptrs.allocator, info.ptrs.voxel_chunks_ptr, chunk_n, ray_pos, result.voxel_data);
    if (lod == 0) {
        result.dist = 0.0;
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
        lod = sample_lod(info.ptrs.allocator, info.ptrs.voxel_chunks_ptr, chunk_n, current_pos, result.voxel_data);
#if TRACE_DEPTH_PREPASS_COMPUTE
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
