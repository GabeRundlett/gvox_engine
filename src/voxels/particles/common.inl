#pragma once

#include <core.inl>
#include <application/input.inl>

// for PackedVoxel
#include <voxels/brushes.inl>

struct ParticleVertex {
    daxa_f32vec3 pos;
    daxa_f32vec3 prev_pos;
    PackedVoxel packed_voxel;
};

struct PackedParticleVertex {
    daxa_u32 data;
};
DAXA_DECL_BUFFER_PTR(PackedParticleVertex)

struct ParticleDrawParams {
    IndirectDrawIndexedParams cube_draw_params;
    IndirectDrawIndexedParams shadow_cube_draw_params;
    IndirectDrawParams splat_draw_params;
};

struct VoxelParticlesState {
    daxa_u32vec3 simulation_dispatch;
    daxa_u32 place_count;
    daxa_u32vec3 place_bounds_min;
    daxa_u32vec3 place_bounds_max;
    ParticleDrawParams sim_particle;
    ParticleDrawParams grass;
    ParticleDrawParams flower;
};
DAXA_DECL_BUFFER_PTR(VoxelParticlesState)
