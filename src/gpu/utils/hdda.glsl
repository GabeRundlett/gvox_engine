#pragma once

#include <shared/shared.inl>

#include <utils/voxels.glsl>
#include <utils/math.glsl>

// bool hdda_is_active(daxa_RWBufferPtr(VoxelMalloc_GlobalAllocator) allocator, daxa_BufferPtr(VoxelLeafChunk) voxel_chunks_ptr, uvec3 chunk_n, ivec3 voxel_i) {
//     voxel_i = clamp(voxel_i, ivec3(0, 0, 0), ivec3(chunk_n) * CHUNK_SIZE - 1);
//     uvec3 chunk_i = voxel_i / CHUNK_SIZE;
//     uint chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;
//     return sample_lod(allocator, voxel_chunks_ptr[chunk_index], chunk_i, voxel_i - chunk_i * CHUNK_SIZE) != 0;
// }

struct HDDA {
    int dim;
    float t0, t1;
    ivec3 voxel, m_step;
    vec3 delta, next;
};
float hdda_time(inout HDDA self) {
    return self.t0;
}
int hdda_dim(inout HDDA self) {
    return self.dim;
}
ivec3 hdda_voxel(inout HDDA self) {
    return self.voxel;
}
ivec3 hdda_round_down(vec3 x) {
    return ivec3(floor(x));
}
void hdda_init(inout HDDA self, vec3 ray_pos, vec3 ray_dir, int dim) {
    self.dim = dim;
    self.t0 = 0.0;
    self.t1 = MAX_SD;
    self.voxel = hdda_round_down(ray_pos) & ivec3(~(dim - 1));
    vec3 inv = vec3(1.0) / ray_dir;
    for (int axis = 0; axis < 3; ++axis) {
        if (ray_dir[axis] == 0.0) {
            self.next[axis] = MAX_SD;
            self.m_step[axis] = 0;
        } else if (inv[axis] > 0.0) {
            self.m_step[axis] = 1;
            self.next[axis] = self.t0 + (float(self.voxel[axis] + dim) - ray_pos[axis]) * inv[axis];
            self.delta[axis] = inv[axis];
        } else {
            self.m_step[axis] = -1;
            self.next[axis] = self.t0 + (float(self.voxel[axis]) - ray_pos[axis]) * inv[axis];
            self.delta[axis] = -inv[axis];
        }
    }
}
bool hdda_update(inout HDDA self, vec3 ray_pos, vec3 ray_dir, int dim) {
    if (self.dim == dim)
        return false;
    self.dim = dim;
    vec3 pos = (fma(vec3(self.t0), ray_dir, ray_pos));
    vec3 inv = vec3(1.0) / ray_dir;
    self.voxel = hdda_round_down(pos) & ivec3(~(dim - 1));
    for (int axis = 0; axis < 3; ++axis) {
        if (self.m_step[axis] == 0)
            continue;
        self.next[axis] = self.t0 + (float(self.voxel[axis]) - pos[axis]) * inv[axis];
        if (self.m_step[axis] > 0)
            self.next[axis] += float(dim) * inv[axis];
    }
    return true;
}
uint hdda_min_index(vec3 v) {
    if (v.x < v.y) {
        if (v.x < v.z) {
            return 0u;
        } else {
            return 2u;
        }
    } else {
        if (v.y < v.z) {
            return 1u;
        } else {
            return 2u;
        }
    }
}
bool hdda_step(inout HDDA self) {
    uint axis = hdda_min_index(self.next);
#if 1
    if (self.next[axis] <= self.t0) {
        self.next[axis] += self.t0 - 0.999999f * self.next[axis] + 1.0e-6f;
    }
#endif
    self.t0 = self.next[axis];
    self.next[axis] += float(self.dim) * self.delta[axis];
    self.voxel[axis] += self.dim * self.m_step[axis];
    return self.t0 <= self.t1;
}