#include <shared/shared.inl>

DAXA_USE_PUSH_CONSTANT(ChunkOptCompPush)

#include <utils/voxel.glsl>

#if defined(SUBCHUNK_X2X4)

shared u32 local_x2_copy[4][4];

void chunk_opt_x2x4(in u32 chunk_index, in u32 chunk_local_workgroup) {
    u32vec2 x2_in_group_location = u32vec2(
        (gl_LocalInvocationID.x >> 5) & 0x3,
        (gl_LocalInvocationID.x >> 7) & 0x3);
    u32vec3 x2_i = u32vec3(
        (gl_LocalInvocationID.x >> 5) & 0x3,
        (gl_LocalInvocationID.x >> 7) & 0x3,
        (gl_LocalInvocationID.x & 0x1F));
    x2_i += 4 * u32vec3(chunk_local_workgroup & 0x7, (chunk_local_workgroup >> 3) & 0x7, 0);
    u32vec3 in_chunk_i = x2_i * 2;
    b32 at_least_one_occluding = false;
    u32 base_id_x1 = sample_voxel_id(chunk_index, in_chunk_i);
    for (i32 x = 0; x < 2; ++x)
        for (i32 y = 0; y < 2; ++y)
            for (i32 z = 0; z < 2; ++z) {
                i32vec3 local_i = i32vec3(in_chunk_i) + i32vec3(x, y, z); // in x1 space
                at_least_one_occluding = at_least_one_occluding || (sample_voxel_id(chunk_index, local_i) != base_id_x1);
            }
    u32 result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(x2_i);
    }
    u32 or_result = subgroupOr(result);
    if (subgroupElect()) {
        u32 index = uniformity_lod_index(2)(x2_i);
        VOXEL_WORLD.uniformity_chunks[chunk_index].lod_x2[index] = or_result;
        local_x2_copy[x2_in_group_location.x][x2_in_group_location.y] = or_result;
    }
    subgroupBarrier();
    if (gl_LocalInvocationID.x >= 64) {
        return;
    }
    u32vec3 x4_i = u32vec3(
        (gl_LocalInvocationID.x >> 4) & 0x1,
        (gl_LocalInvocationID.x >> 5) & 0x1,
        gl_LocalInvocationID.x & 0xF);
    x4_i += 2 * u32vec3(chunk_local_workgroup & 0x7, (chunk_local_workgroup >> 3) & 0x7, 0);
    x2_i = x4_i * 2;
    u32 base_id_x2 = sample_voxel_id(chunk_index, x2_i * 2);
    at_least_one_occluding = false;
    for (i32 x = 0; x < 2; ++x)
        for (i32 y = 0; y < 2; ++y)
            for (i32 z = 0; z < 2; ++z) {
                i32vec3 local_i = i32vec3(x2_i) + i32vec3(x, y, z); // in x2 space
                u32 mask = uniformity_lod_mask(local_i);
                u32vec2 x2_in_group_index = u32vec2(
                    local_i.x & 0x3,
                    local_i.y & 0x3);
                b32 is_occluding = (local_x2_copy[x2_in_group_index.x][x2_in_group_index.y] & mask) != 0;
                at_least_one_occluding = at_least_one_occluding || is_occluding || (sample_voxel_id(chunk_index, local_i * 2) != base_id_x2);
            }
    result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(x4_i);
    }
    for (i32 i = 0; i < 2; i++) {
        if ((gl_SubgroupInvocationID >> 4) == i) {
            result = subgroupOr(result);
        }
    }
    if ((gl_SubgroupInvocationID & 0xF /* = %16 */) == 0) {
        u32 index = uniformity_lod_index(4)(x4_i);
        VOXEL_WORLD.uniformity_chunks[chunk_index].lod_x4[index] = result;
    }
}

layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
void main() {
    u32 chunk_index = VOXEL_WORLD.chunk_update_indices[gl_WorkGroupID.y / 64];
    if (VOXEL_WORLD.chunks_genstate[chunk_index].edit_stage == 2)
        return;

    chunk_opt_x2x4(chunk_index, gl_WorkGroupID.y % 64);
}

#endif

#if defined(SUBCHUNK_X8UP)

shared u32 local_x8_copy[64];
shared u32 local_x16_copy[16];

void chunk_opt_x8up(in u32 chunk_index) {
    u32vec3 x8_i = u32vec3(
        (gl_LocalInvocationID.x >> 3) & 0x7,
        (gl_LocalInvocationID.x >> 6) & 0x7,
        gl_LocalInvocationID.x & 0x7);
    u32vec3 x4_i = x8_i * 2;

    u32 base_id_x4 = sample_voxel_id(chunk_index, x4_i * 4);

    b32 at_least_one_occluding = false;
    for (i32 x = 0; x < 2; ++x)
        for (i32 y = 0; y < 2; ++y)
            for (i32 z = 0; z < 2; ++z) {
                i32vec3 local_i = i32vec3(x4_i) + i32vec3(x, y, z); // x4 space
                u32 index = uniformity_lod_index(4)(local_i);
                u32 mask = uniformity_lod_mask(local_i);
                b32 occluding = (VOXEL_WORLD.uniformity_chunks[chunk_index].lod_x4[index] & mask) != 0;
                at_least_one_occluding = at_least_one_occluding || occluding || (sample_voxel_id(chunk_index, local_i * 4) != base_id_x4);
            }

    u32 result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(x8_i);
    }
    for (i32 i = 0; i < 4; i++) {
        if ((gl_SubgroupInvocationID >> 3) == i) {
            result = subgroupOr(result);
        }
    }
    if ((gl_SubgroupInvocationID & 0x7 /* == % 8*/) == 0) {
        u32 index = uniformity_lod_index(8)(x8_i);
        VOXEL_WORLD.uniformity_chunks[chunk_index].lod_x8[index] = result;
        local_x8_copy[index] = result;
    }

    subgroupBarrier();

    if (gl_LocalInvocationID.x >= 64) {
        return;
    }

    u32vec3 x16_i = u32vec3(
        (gl_LocalInvocationID.x >> 2) & 0x3,
        (gl_LocalInvocationID.x >> 4) & 0x3,
        gl_LocalInvocationID.x & 0x3);
    x8_i = x16_i * 2;
    u32 base_id_x8 = sample_voxel_id(chunk_index, x8_i * 8);

    at_least_one_occluding = false;
    for (i32 x = 0; x < 2; ++x)
        for (i32 y = 0; y < 2; ++y)
            for (i32 z = 0; z < 2; ++z) {
                i32vec3 local_i = i32vec3(x8_i) + i32vec3(x, y, z); // x8 space
                u32 mask = uniformity_lod_mask(local_i);
                u32 index = uniformity_lod_index(8)(local_i);
                b32 is_occluding = (local_x8_copy[index] & mask) != 0;
                at_least_one_occluding = at_least_one_occluding || is_occluding || (sample_voxel_id(chunk_index, local_i * 8) != base_id_x8);
            }

    result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(x16_i);
    }
    for (i32 i = 0; i < 8; i++) {
        if ((gl_SubgroupInvocationID >> 2) == i) {
            result = subgroupOr(result);
        }
    }
    if ((gl_SubgroupInvocationID & 0x3) == 0) {
        u32 index = uniformity_lod_index(16)(x16_i);
        VOXEL_WORLD.uniformity_chunks[chunk_index].lod_x16[index] = result;
        local_x16_copy[index] = result;
    }

    subgroupBarrier();

    if (gl_LocalInvocationID.x >= 8) {
        return;
    }

    u32vec3 x32_i = u32vec3(
        (gl_LocalInvocationID.x >> 1) & 0x1,
        (gl_LocalInvocationID.x >> 2) & 0x1,
        gl_LocalInvocationID.x & 0x1);
    x16_i = x32_i * 2;
    u32 base_id_x16 = sample_voxel_id(chunk_index, x16_i * 16);

    at_least_one_occluding = false;
    for (i32 x = 0; x < 2; ++x)
        for (i32 y = 0; y < 2; ++y)
            for (i32 z = 0; z < 2; ++z) {
                i32vec3 local_i = i32vec3(x16_i) + i32vec3(x, y, z); // x16 space
                u32 mask = uniformity_lod_mask(local_i);
                u32 index = uniformity_lod_index(16)(local_i);
                b32 is_occluding = (local_x16_copy[index] & mask) != 0;
                at_least_one_occluding = at_least_one_occluding || is_occluding || (sample_voxel_id(chunk_index, local_i * 16) != base_id_x16);
            }

    result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(x32_i);
    }
    for (i32 i = 0; i < 16; i++) {
        if ((gl_SubgroupInvocationID >> 1) == i) {
            result = subgroupOr(result);
        }
    }
    if ((gl_SubgroupInvocationID & 0x1) == 0) {
        u32 index = uniformity_lod_index(32)(x32_i);
        VOXEL_WORLD.uniformity_chunks[chunk_index].lod_x32[index] = result;
    }
}

layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
void main() {
    u32 chunk_index = VOXEL_WORLD.chunk_update_indices[gl_WorkGroupID.y];
    if (VOXEL_WORLD.chunks_genstate[chunk_index].edit_stage == 2)
        return;

    chunk_opt_x8up(chunk_index);
}

#endif
