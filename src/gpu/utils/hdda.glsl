#pragma once

#include <shared/shared.inl>

#include <utils/voxels.glsl>
#include <utils/math.glsl>

bool hdda_is_active(daxa_RWBufferPtr(VoxelMalloc_GlobalAllocator) allocator, daxa_BufferPtr(VoxelLeafChunk) voxel_chunks_ptr, u32vec3 chunk_n, i32vec3 voxel_i) {
    voxel_i = clamp(voxel_i, i32vec3(0, 0, 0), i32vec3(chunk_n) * CHUNK_SIZE - 1);
    u32vec3 chunk_i = voxel_i / CHUNK_SIZE;
    u32 chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;
    return sample_lod(allocator, voxel_chunks_ptr[chunk_index], chunk_i, voxel_i - chunk_i * CHUNK_SIZE) != 0;
}

struct HDDA {
    i32 dim;
    f32 t0, t1;
    i32vec3 voxel, m_step;
    f32vec3 delta, next;
};

f32 hdda_time(in out HDDA self) {
    return self.t0;
}
i32 hdda_dim(in out HDDA self) {
    return self.dim;
}
i32vec3 hdda_voxel(in out HDDA self) {
    return self.voxel;
}

i32vec3 hdda_round_down(f32vec3 x) {
    return i32vec3(floor(x));
}

void hdda_init(in out HDDA self, f32vec3 ray_pos, f32vec3 ray_dir, i32 dim) {
    self.dim = dim;
    self.t0 = 0;
    self.t1 = MAX_SD;
    self.voxel = hdda_round_down(ray_pos) & i32vec3(~(dim - 1));
    f32vec3 inv = f32vec3(1.0) / ray_dir;
    for (i32 axis = 0; axis < 3; ++axis) {
        if (ray_dir[axis] == 0) {
            self.next[axis] = MAX_SD;
            self.m_step[axis] = 0;
        } else if (inv[axis] > 0) {
            self.m_step[axis] = 1;
            self.next[axis] = self.t0 + (self.voxel[axis] + dim - ray_pos[axis]) * inv[axis];
            self.delta[axis] = inv[axis];
        } else {
            self.m_step[axis] = -1;
            self.next[axis] = self.t0 + (self.voxel[axis] - ray_pos[axis]) * inv[axis];
            self.delta[axis] = -inv[axis];
        }
    }
}

u32 hdda_min_index(f32vec3 v) {
    if (v.x < v.y) {
        if (v.x < v.z) {
            return 0;
        } else {
            return 2;
        }
    } else {
        if (v.y < v.z) {
            return 1;
        } else {
            return 2;
        }
    }
}

bool hdda_step(in out HDDA self) {
    const u32 axis = hdda_min_index(self.next);
    self.t0 = self.next[axis];
    self.next[axis] += self.dim * self.delta[axis];
    self.voxel[axis] += self.dim * self.m_step[axis];
    return self.t0 <= self.t1;
}

bool hdda_update(in out HDDA self, f32vec3 ray_pos, f32vec3 ray_dir, i32 dim) {
    if (self.dim == dim)
        return false;
    self.dim = dim;
    f32vec3 pos = (fma(f32vec3(self.t0), ray_dir, ray_pos));
    f32vec3 inv = f32vec3(1.0) / ray_dir;
    self.voxel = hdda_round_down(pos) & i32vec3(~(dim - 1));
    for (i32 axis = 0; axis < 3; ++axis) {
        if (self.m_step[axis] == 0)
            continue;
        self.next[axis] = self.t0 + (self.voxel[axis] - pos[axis]) * inv[axis];
        if (self.m_step[axis] > 0)
            self.next[axis] += dim * inv[axis];
    }
    return true;
}
