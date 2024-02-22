#pragma once

#include <utilities/gpu/math.glsl>
#include <voxels/impl/voxels.glsl>

VoxelTraceResult voxel_trace(in VoxelTraceInfo info, in out vec3 ray_pos) {
    // const uint lod_index = 7;

    VoxelTraceResult result;
    result.dist = info.max_dist;

    result.vel = vec3(deref(info.ptrs.globals).offset - deref(info.ptrs.globals).prev_offset);

    uvec3 chunk_n = uvec3(1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);

    for (uint lod_index = 0; lod_index < CHUNK_LOD_LEVELS; ++lod_index) {
        if (lod_index != 0) {
            // intersect(ray_pos, info.ray_dir, vec3(1) / info.ray_dir, b);
        }

        daxa_BufferPtr(VoxelLeafChunk) voxel_chunks_ptr = advance(info.ptrs.voxel_chunks_ptr, TOTAL_CHUNKS_PER_LOD * lod_index);
        float voxel_scl = float(VOXEL_SCL) / float(1 << lod_index);
        vec3 offset = vec3((deref(info.ptrs.globals).offset) & ((1 << (lod_index + 3)) - 1)) + vec3(chunk_n) * CHUNK_WORLDSPACE_SIZE * 0.5;
        ray_pos += offset;
        BoundingBox b;
        b.bound_min = vec3(0.0);
        b.bound_max = b.bound_min + vec3(chunk_n) * CHUNK_WORLDSPACE_SIZE;

        if (ENABLE_CHUNK_WRAPPING == 0) {
            intersect(ray_pos, info.ray_dir, vec3(1) / info.ray_dir, b);
        }

        ray_pos += info.ray_dir * 0.01 / voxel_scl;

        if (false) {
            result.dist = 0.0;
            if (!inside(ray_pos, b)) {
                if (info.extend_to_max_dist) {
                    result.dist = info.max_dist;
                }
            } else {
                PackedVoxel dontcare;
                uint lod = sample_lod(info.ptrs.globals, info.ptrs.allocator, voxel_chunks_ptr, chunk_n, ray_pos, lod_index, dontcare);
                Voxel voxel = Voxel(0, 0, vec3(0), vec3(0));
                voxel.color = vec3(0);
                switch (lod) {
                case 0: voxel.color = vec3(0.0, 0.0, 0.0); break;
                case 1: voxel.color = vec3(1.0, 0.0, 0.0); break;
                case 2: voxel.color = vec3(0.0, 1.0, 0.0); break;
                case 3: voxel.color = vec3(1.0, 1.0, 0.0); break;
                case 4: voxel.color = vec3(0.0, 0.0, 1.0); break;
                case 5: voxel.color = vec3(1.0, 0.0, 1.0); break;
                case 6: voxel.color = vec3(0.0, 1.0, 1.0); break;
                case 7: voxel.color = vec3(1.0, 1.0, 1.0); break;
                }
                // if (abs(fract(ray_pos.x * voxel_scl) - 0.05) < 0.1 ||
                //     abs(fract(ray_pos.y * voxel_scl) - 0.05) < 0.1 ||
                //     abs(fract(ray_pos.z * voxel_scl) - 0.05) < 0.1) {
                //     col = col * 0.5 + 0.25;
                // }
                result.voxel_data = pack_voxel(voxel);
                result.nrm = vec3(0, 0, 1);
            }
            ray_pos -= offset;
            return result;
        }

        if (!inside(ray_pos, b)) {
            if (info.extend_to_max_dist) {
                result.dist = info.max_dist;
            } else {
                result.dist = 0.0;
            }
            ray_pos -= offset;
            return result;
        }

        vec3 delta = vec3(
            info.ray_dir.x == 0 ? 3.0 * info.max_steps : abs(1.0 / info.ray_dir.x),
            info.ray_dir.y == 0 ? 3.0 * info.max_steps : abs(1.0 / info.ray_dir.y),
            info.ray_dir.z == 0 ? 3.0 * info.max_steps : abs(1.0 / info.ray_dir.z));
        uint lod = sample_lod(info.ptrs.globals, info.ptrs.allocator, voxel_chunks_ptr, chunk_n, ray_pos, lod_index, result.voxel_data);
        if (lod == 0) {
            result.dist = 0.0;
            ray_pos -= offset;
            return result;
        }
        float cell_size = float(1l << (lod - 1)) / voxel_scl;
        vec3 t_start;
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
        float t_curr = min(min(t_start.x, t_start.y), t_start.z);
        vec3 current_pos = ray_pos;
        vec3 t_next = t_start;
        bool hit_surface = false;
        for (result.step_n = 0; result.step_n < info.max_steps; ++result.step_n) {
            current_pos = ray_pos + info.ray_dir * t_curr;
            if (!inside(current_pos + info.ray_dir * 0.001, b) || t_curr > info.max_dist) {
                break;
            }
            lod = sample_lod(info.ptrs.globals, info.ptrs.allocator, voxel_chunks_ptr, chunk_n, current_pos, lod_index, result.voxel_data);
#if TraceDepthPrepassComputeShader
            hit_surface = lod < clamp(sqrt(t_curr * info.angular_coverage * voxel_scl), 1, 7);
#else
            hit_surface = lod == 0;
#endif
            if (hit_surface) {
                result.nrm = sign(info.ray_dir) * (sign(t_next - min(min(t_next.x, t_next.y), t_next.z).xxx) - 1);
                result.dist = t_curr;
                break;
            }
            cell_size = float(1l << (lod - 1)) / voxel_scl;
            t_next = (0.5 + sign(info.ray_dir) * (0.5 - fract(current_pos / cell_size))) * cell_size * delta;
            // HACK to mitigate fp imprecision...
            t_next += 0.0001 * (sign(info.ray_dir) * -0.5 + 0.5);
            // t_curr += (min(min(t_next.x, t_next.y), t_next.z));
            t_curr += (min(min(t_next.x, t_next.y), t_next.z) + 0.0001 / voxel_scl);
        }
        ray_pos -= offset;

        if (hit_surface) {

            if (info.extend_to_max_dist) {
                t_curr = result.dist;
            }
            ray_pos = ray_pos + info.ray_dir * t_curr;
            result.dist = t_curr;

#if PER_VOXEL_NORMALS
            Voxel voxel = unpack_voxel(result.voxel_data);

            // vec3 voxel_pos = floor(ray_pos * VOXEL_SCL) / VOXEL_SCL;
            // vec3 del = normalize(voxel_pos - cam_pos);
            // if (dot(voxel.normal, del) > -1.0 && dot(trace_result.nrm, voxel.normal) < 0.0) {
            //     voxel.normal *= -1;
            // }

            result.nrm = voxel.normal;
#endif

            break;
        }
    }

    return result;
}
