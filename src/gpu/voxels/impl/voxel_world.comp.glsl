#include <shared/app.inl>

#if PER_CHUNK_COMPUTE

#include <utils/math.glsl>
#include <voxels/impl/voxels.glsl>

#define VOXEL_WORLD deref(voxel_globals)
#define PLAYER deref(globals).player
#define CHUNKS(i) deref(voxel_chunks[i])
#define INDIRECT deref(globals).indirect_dispatch

void try_elect(in out VoxelChunkUpdateInfo work_item) {
    u32 prev_update_n = atomicAdd(VOXEL_WORLD.chunk_update_n, 1);

    // Check if the work item can be added
    if (prev_update_n < MAX_CHUNK_UPDATES_PER_FRAME) {
        // Set the chunk edit dispatch z axis (64/8, 64/8, 64 x 8 x 8 / 8 = 64 x 8) = (8, 8, 512)
        atomicAdd(INDIRECT.chunk_edit_dispatch.z, CHUNK_SIZE / 8);
        atomicAdd(INDIRECT.subchunk_x2x4_dispatch.z, 1);
        atomicAdd(INDIRECT.subchunk_x8up_dispatch.z, 1);
        // Set the chunk update info
        VOXEL_WORLD.chunk_update_infos[prev_update_n] = work_item;
    }
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    i32vec3 chunk_n = i32vec3(1 << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);

    i32vec3 offset = (VOXEL_WORLD.offset >> i32vec3(3));
    i32vec3 prev_offset = (VOXEL_WORLD.prev_offset >> i32vec3(3));

    VoxelChunkUpdateInfo terrain_work_item;
    terrain_work_item.i = i32vec3(gl_GlobalInvocationID.xyz);
    terrain_work_item.chunk_offset = offset;
    terrain_work_item.brush_flags = BRUSH_FLAGS_WORLD_BRUSH;

    // (const) number of chunks in each axis
    u32 chunk_index = calc_chunk_index_from_worldspace(terrain_work_item.i, chunk_n);

    if ((CHUNKS(chunk_index).flags & CHUNK_FLAGS_ACCEL_GENERATED) == 0) {
        try_elect(terrain_work_item);
    } 
    else if (offset != prev_offset) {
        // invalidate chunks outside the chunk_offset
        i32vec3 diff = clamp(i32vec3(offset - prev_offset), -chunk_n, chunk_n);

        i32vec3 start;
        i32vec3 end;

        start.x = diff.x < 0 ? 0 : chunk_n.x - diff.x;
        end.x = diff.x < 0 ? -diff.x : chunk_n.x;

        start.y = diff.y < 0 ? 0 : chunk_n.y - diff.y;
        end.y = diff.y < 0 ? -diff.y : chunk_n.y;

        start.z = diff.z < 0 ? 0 : chunk_n.z - diff.z;
        end.z = diff.z < 0 ? -diff.z : chunk_n.z;

        u32vec3 temp_chunk_i = u32vec3((i32vec3(terrain_work_item.i) - offset) % i32vec3(chunk_n));

        if ((temp_chunk_i.x >= start.x && temp_chunk_i.x < end.x) ||
            (temp_chunk_i.y >= start.y && temp_chunk_i.y < end.y) ||
            (temp_chunk_i.z >= start.z && temp_chunk_i.z < end.z)) {
            CHUNKS(chunk_index).flags &= ~CHUNK_FLAGS_ACCEL_GENERATED;
            try_elect(terrain_work_item);
        }
    }
}

#undef INDIRECT
#undef CHUNKS
#undef PLAYER
#undef VOXEL_WORLD

#endif

#if CHUNK_EDIT_COMPUTE

#include <utils/math.glsl>
#include <utils/noise.glsl>
#include <voxels/impl/voxels.glsl>

u32vec3 chunk_n;
u32 temp_chunk_index;
i32vec3 chunk_i;
u32 chunk_index;
daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr;
daxa_BufferPtr(VoxelLeafChunk) voxel_chunk_ptr;
u32vec3 inchunk_voxel_i;
i32vec3 voxel_i;
i32vec3 world_voxel;
f32vec3 voxel_pos;
BrushInput brush_input;

#include "../brushes.glsl"

#define VOXEL_WORLD deref(voxel_globals)
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    // (const) number of chunks in each axis
    chunk_n = u32vec3(1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);
    // Index in chunk_update_infos buffer
    temp_chunk_index = gl_GlobalInvocationID.z / CHUNK_SIZE;
    // Chunk 3D index in leaf chunk space (0^3 - 31^3)
    chunk_i = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].i;

    // Here we check whether the chunk update that we're handling is an update
    // for a chunk that has already been submitted. This is a bit inefficient,
    // since we'd hopefully like to queue a separate work item into the queue
    // instead, but this is tricky.
    if (chunk_i == INVALID_CHUNK_I) {
        return;
    }

    // Player chunk offset
    i32vec3 chunk_offset = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].chunk_offset;
    // Brush informations
    brush_input = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].brush_input;
    // Brush flags
    u32 brush_flags = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].brush_flags;
    // Chunk u32 index in voxel_chunks buffer
    chunk_index = calc_chunk_index_from_worldspace(chunk_i, chunk_n);
    // Pointer to the previous chunk
    temp_voxel_chunk_ptr = temp_voxel_chunks + temp_chunk_index;
    // Pointer to the new chunk
    voxel_chunk_ptr = voxel_chunks + chunk_index;
    // Voxel offset in chunk
    inchunk_voxel_i = gl_GlobalInvocationID.xyz - u32vec3(0, 0, temp_chunk_index * CHUNK_SIZE);
    // Voxel 3D position (in voxel buffer)
    voxel_i = chunk_i * CHUNK_SIZE + i32vec3(inchunk_voxel_i);

    // Wrapped chunk index in leaf chunk space (0^3 - 31^3)
    i32vec3 wrapped_chunk_i = imod3(chunk_i - imod3(chunk_offset - i32vec3(chunk_n), i32vec3(chunk_n)), i32vec3(chunk_n));
    // Leaf chunk position in world space
    i32vec3 world_chunk = chunk_offset + wrapped_chunk_i - i32vec3(chunk_n / 2);

    // Voxel position in world space (voxels)
    world_voxel = world_chunk * CHUNK_SIZE + i32vec3(inchunk_voxel_i);
    // Voxel position in world space (meters)
    voxel_pos = f32vec3(world_voxel) / VOXEL_SCL;

    rand_seed(voxel_i.x + voxel_i.y * 1000 + voxel_i.z * 1000 * 1000);

    f32vec3 col = f32vec3(0.0);
    u32 id = 0;

    if ((brush_flags & BRUSH_FLAGS_WORLD_BRUSH) != 0) {
        brushgen_world(col, id);
    }
    if ((brush_flags & BRUSH_FLAGS_USER_BRUSH_A) != 0) {
        brushgen_a(col, id);
    }
    if ((brush_flags & BRUSH_FLAGS_USER_BRUSH_B) != 0) {
        brushgen_b(col, id);
    }
    // if ((brush_flags & BRUSH_FLAGS_PARTICLE_BRUSH) != 0) {
    //     brushgen_particles(col, id);
    // }

    TempVoxel result;
    result.col_and_id = f32vec4_to_uint_rgba8(f32vec4(col, 0.0)) | (id << 0x18);
    deref(temp_voxel_chunk_ptr).voxels[inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE] = result;
}
#undef VOXEL_WORLD

#endif

#if CHUNK_OPT_COMPUTE

#include <voxels/impl/voxels.glsl>

#define WAVE_SIZE gl_SubgroupSize
#define WAVE_SIZE_MUL (WAVE_SIZE / 32)

u32 sample_temp_voxel_id(daxa_BufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr, in u32vec3 in_chunk_i) {
    u32 in_chunk_index = in_chunk_i.x + in_chunk_i.y * CHUNK_SIZE + in_chunk_i.z * CHUNK_SIZE * CHUNK_SIZE;
    return deref(temp_voxel_chunk_ptr).voxels[in_chunk_index].col_and_id >> 24;
}

// For now, I'm testing with using non-zero as the accel structure, instead of uniformity.
u32 sample_base_voxel_id(daxa_BufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr, in u32vec3 in_chunk_i) {
#if VOXEL_ACCEL_UNIFORMITY
    return sample_temp_voxel_id(temp_voxel_chunk_ptr, in_chunk_i);
#else
    return 0;
#endif
}

#if CHUNK_OPT_STAGE == 0

shared u32 local_x2_copy[4][4];

#define VOXEL_CHUNK deref(voxel_chunk_ptr)
void chunk_opt_x2x4(daxa_BufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr, daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunk_ptr, in u32 chunk_local_workgroup) {
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
    u32 base_id_x1 = sample_base_voxel_id(temp_voxel_chunk_ptr, in_chunk_i);
    for (i32 x = 0; x < 2; ++x)
        for (i32 y = 0; y < 2; ++y)
            for (i32 z = 0; z < 2; ++z) {
                i32vec3 local_i = i32vec3(in_chunk_i) + i32vec3(x, y, z); // in x1 space
                at_least_one_occluding = at_least_one_occluding || (sample_temp_voxel_id(temp_voxel_chunk_ptr, local_i) != base_id_x1);
            }
    u32 result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(x2_i);
    }
    for (i32 i = 0; i < 1 * WAVE_SIZE_MUL; i++) {
        if ((gl_SubgroupInvocationID >> 5) == i) {
            result = subgroupOr(result);
        }
    }
    if ((gl_SubgroupInvocationID & 0x1F /* = %32 */) == 0) {
        u32 index = uniformity_lod_index(2)(x2_i);
        VOXEL_CHUNK.uniformity.lod_x2[index] = result;
        local_x2_copy[x2_in_group_location.x][x2_in_group_location.y] = result;
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
    u32 base_id_x2 = sample_base_voxel_id(temp_voxel_chunk_ptr, x2_i * 2);
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
                at_least_one_occluding = at_least_one_occluding || is_occluding || (sample_temp_voxel_id(temp_voxel_chunk_ptr, local_i * 2) != base_id_x2);
            }
    result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(x4_i);
    }
    for (i32 i = 0; i < 2 * WAVE_SIZE_MUL; i++) {
        if ((gl_SubgroupInvocationID >> 4) == i) {
            result = subgroupOr(result);
        }
    }
    if ((gl_SubgroupInvocationID & 0xF /* = %16 */) == 0) {
        u32 index = uniformity_lod_index(4)(x4_i);
        VOXEL_CHUNK.uniformity.lod_x4[index] = result;
    }
}
#undef VOXEL_CHUNK

#define VOXEL_WORLD deref(voxel_globals)
layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
void main() {
    i32vec3 chunk_i = VOXEL_WORLD.chunk_update_infos[gl_WorkGroupID.z].i;
    if (chunk_i == INVALID_CHUNK_I) {
        return;
    }
    u32vec3 chunk_n;
    chunk_n.x = 1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS;
    chunk_n.y = chunk_n.x;
    chunk_n.z = chunk_n.x;
    u32 chunk_index = calc_chunk_index_from_worldspace(chunk_i, chunk_n);
    daxa_BufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr = temp_voxel_chunks + gl_WorkGroupID.z;
    daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunk_ptr = voxel_chunks + chunk_index;
    chunk_opt_x2x4(temp_voxel_chunk_ptr, voxel_chunk_ptr, gl_WorkGroupID.y);
}
#undef VOXEL_WORLD

#endif

#if CHUNK_OPT_STAGE == 1

shared u32 local_x8_copy[64];
shared u32 local_x16_copy[16];

#define VOXEL_CHUNK deref(voxel_chunk_ptr)
void chunk_opt_x8up(daxa_BufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr, daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunk_ptr) {
    u32vec3 x8_i = u32vec3(
        (gl_LocalInvocationID.x >> 3) & 0x7,
        (gl_LocalInvocationID.x >> 6) & 0x7,
        gl_LocalInvocationID.x & 0x7);
    u32vec3 x4_i = x8_i * 2;

    u32 base_id_x4 = sample_base_voxel_id(temp_voxel_chunk_ptr, x4_i * 4);

    b32 at_least_one_occluding = false;
    for (i32 x = 0; x < 2; ++x)
        for (i32 y = 0; y < 2; ++y)
            for (i32 z = 0; z < 2; ++z) {
                i32vec3 local_i = i32vec3(x4_i) + i32vec3(x, y, z); // x4 space
                u32 index = uniformity_lod_index(4)(local_i);
                u32 mask = uniformity_lod_mask(local_i);
                b32 occluding = (VOXEL_CHUNK.uniformity.lod_x4[index] & mask) != 0;
                at_least_one_occluding = at_least_one_occluding || occluding || (sample_temp_voxel_id(temp_voxel_chunk_ptr, local_i * 4) != base_id_x4);
            }

    u32 result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(x8_i);
    }
    for (i32 i = 0; i < 4 * WAVE_SIZE_MUL; i++) {
        if ((gl_SubgroupInvocationID >> 3) == i) {
            result = subgroupOr(result);
        }
    }
    if ((gl_SubgroupInvocationID & 0x7 /* == % 8*/) == 0) {
        u32 index = uniformity_lod_index(8)(x8_i);
        VOXEL_CHUNK.uniformity.lod_x8[index] = result;
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
    u32 base_id_x8 = sample_base_voxel_id(temp_voxel_chunk_ptr, x8_i * 8);

    at_least_one_occluding = false;
    for (i32 x = 0; x < 2; ++x)
        for (i32 y = 0; y < 2; ++y)
            for (i32 z = 0; z < 2; ++z) {
                i32vec3 local_i = i32vec3(x8_i) + i32vec3(x, y, z); // x8 space
                u32 mask = uniformity_lod_mask(local_i);
                u32 index = uniformity_lod_index(8)(local_i);
                b32 is_occluding = (local_x8_copy[index] & mask) != 0;
                at_least_one_occluding = at_least_one_occluding || is_occluding || (sample_temp_voxel_id(temp_voxel_chunk_ptr, local_i * 8) != base_id_x8);
            }

    result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(x16_i);
    }
    for (i32 i = 0; i < 8 * WAVE_SIZE_MUL; i++) {
        if ((gl_SubgroupInvocationID >> 2) == i) {
            result = subgroupOr(result);
        }
    }
    if ((gl_SubgroupInvocationID & 0x3) == 0) {
        u32 index = uniformity_lod_index(16)(x16_i);
        VOXEL_CHUNK.uniformity.lod_x16[index] = result;
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
    u32 base_id_x16 = sample_base_voxel_id(temp_voxel_chunk_ptr, x16_i * 16);

    at_least_one_occluding = false;
    for (i32 x = 0; x < 2; ++x)
        for (i32 y = 0; y < 2; ++y)
            for (i32 z = 0; z < 2; ++z) {
                i32vec3 local_i = i32vec3(x16_i) + i32vec3(x, y, z); // x16 space
                u32 mask = uniformity_lod_mask(local_i);
                u32 index = uniformity_lod_index(16)(local_i);
                b32 is_occluding = (local_x16_copy[index] & mask) != 0;
                at_least_one_occluding = at_least_one_occluding || is_occluding || (sample_temp_voxel_id(temp_voxel_chunk_ptr, local_i * 16) != base_id_x16);
            }

    result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(x32_i);
    }
    for (i32 i = 0; i < 16 * WAVE_SIZE_MUL; i++) {
        if ((gl_SubgroupInvocationID >> 1) == i) {
            result = subgroupOr(result);
        }
    }
    if ((gl_SubgroupInvocationID & 0x1) == 0) {
        u32 index = uniformity_lod_index(32)(x32_i);
        VOXEL_CHUNK.uniformity.lod_x32[index] = result;
    }
}
#undef VOXEL_CHUNK

#define VOXEL_WORLD deref(voxel_globals)
layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
void main() {
    i32vec3 chunk_i = VOXEL_WORLD.chunk_update_infos[gl_WorkGroupID.z].i;
    if (chunk_i == INVALID_CHUNK_I) {
        return;
    }
    u32vec3 chunk_n;
    chunk_n.x = 1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS;
    chunk_n.y = chunk_n.x;
    chunk_n.z = chunk_n.x;
    u32 chunk_index = calc_chunk_index_from_worldspace(chunk_i, chunk_n);
    daxa_BufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr = temp_voxel_chunks + gl_WorkGroupID.z;
    daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunk_ptr = voxel_chunks + chunk_index;
    chunk_opt_x8up(temp_voxel_chunk_ptr, voxel_chunk_ptr);

    // Finish the chunk
    if (gl_LocalInvocationIndex == 0) {
        deref(voxel_chunk_ptr).flags = CHUNK_FLAGS_ACCEL_GENERATED;
    }
}
#undef VOXEL_WORLD

#endif

#endif

#if CHUNK_ALLOC_COMPUTE

#extension GL_EXT_shader_atomic_int64 : require

#include <utils/math.glsl>
#include <voxels/impl/voxel_malloc.glsl>
#include <voxels/impl/voxels.glsl>

shared u32 compression_result[PALETTE_REGION_TOTAL_SIZE];
shared u64 voted_results[PALETTE_REGION_TOTAL_SIZE];
shared u32 palette_size;

void process_palette_region(u32 palette_region_voxel_index, u32 my_voxel, in out u32 my_palette_index) {
    if (palette_region_voxel_index == 0) {
        palette_size = 0;
    }
    voted_results[palette_region_voxel_index] = 0;
    barrier();
    for (u32 algo_i = 0; algo_i < PALETTE_MAX_COMPRESSED_VARIANT_N + 1; ++algo_i) {
        if (my_palette_index == 0) {
            u64 vote_result = atomicCompSwap(voted_results[algo_i], 0, u64(my_voxel) | (u64(1) << u64(32)));
            if (vote_result == 0) {
                my_palette_index = algo_i + 1;
                compression_result[palette_size] = my_voxel;
                palette_size++;
            } else if (my_voxel == u32(vote_result)) {
                my_palette_index = algo_i + 1;
            }
        }
        barrier();
        memoryBarrierShared();
        if (voted_results[algo_i] == 0) {
            break;
        }
    }
}

#define VOXEL_WORLD deref(voxel_globals)
layout(local_size_x = PALETTE_REGION_SIZE, local_size_y = PALETTE_REGION_SIZE, local_size_z = PALETTE_REGION_SIZE) in;
void main() {
    u32vec3 chunk_n = u32vec3(1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);
    u32 temp_chunk_index = gl_GlobalInvocationID.z / CHUNK_SIZE;
    i32vec3 chunk_i = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].i;
    if (chunk_i == INVALID_CHUNK_I) {
        return;
    }
    u32 chunk_index = calc_chunk_index_from_worldspace(chunk_i, chunk_n);
    u32vec3 inchunk_voxel_i = gl_GlobalInvocationID.xyz - u32vec3(0, 0, temp_chunk_index * CHUNK_SIZE);
    u32 inchunk_voxel_index = inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE;
    u32 palette_region_voxel_index =
        gl_LocalInvocationID.x +
        gl_LocalInvocationID.y * PALETTE_REGION_SIZE +
        gl_LocalInvocationID.z * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE;
    u32vec3 palette_i = u32vec3(gl_WorkGroupID.xy, gl_WorkGroupID.z - temp_chunk_index * PALETTES_PER_CHUNK_AXIS);
    u32 palette_region_index =
        palette_i.x +
        palette_i.y * PALETTES_PER_CHUNK_AXIS +
        palette_i.z * PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS;

    daxa_BufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr = temp_voxel_chunks + temp_chunk_index;
    daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunk_ptr = voxel_chunks + chunk_index;

    u32 my_voxel = deref(temp_voxel_chunk_ptr).voxels[inchunk_voxel_index].col_and_id;
    u32 my_palette_index = 0;

    process_palette_region(palette_region_voxel_index, my_voxel, my_palette_index);

    u32 prev_variant_n = deref(voxel_chunk_ptr).palette_headers[palette_region_index].variant_n;
    VoxelMalloc_Pointer prev_blob_ptr = deref(voxel_chunk_ptr).palette_headers[palette_region_index].blob_ptr;

    u32 bits_per_variant = ceil_log2(palette_size);

    u32 compressed_size = 0;
    VoxelMalloc_Pointer blob_ptr = my_voxel;

    if (palette_size > PALETTE_MAX_COMPRESSED_VARIANT_N) {
        compressed_size = PALETTE_REGION_TOTAL_SIZE;
        if (prev_variant_n > 1) {
            blob_ptr = prev_blob_ptr;
            VoxelMalloc_realloc(voxel_malloc_page_allocator, voxel_chunk_ptr, blob_ptr, compressed_size);
        } else {
            blob_ptr = VoxelMalloc_malloc(voxel_malloc_page_allocator, voxel_chunk_ptr, compressed_size);
        }
        if (palette_region_voxel_index == 0) {
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].variant_n = palette_size;
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].blob_ptr = blob_ptr;
        }

        compression_result[palette_region_voxel_index] = my_voxel;
    } else if (palette_size > 1) {
        compressed_size = palette_size + (bits_per_variant * PALETTE_REGION_TOTAL_SIZE + 31) / 32;
        if (prev_variant_n > 1) {
            blob_ptr = prev_blob_ptr;
            VoxelMalloc_realloc(voxel_malloc_page_allocator, voxel_chunk_ptr, blob_ptr, compressed_size);
        } else {
            blob_ptr = VoxelMalloc_malloc(voxel_malloc_page_allocator, voxel_chunk_ptr, compressed_size);
        }
        if (palette_region_voxel_index == 0) {
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].variant_n = palette_size;
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].blob_ptr = blob_ptr;
        }

        u32 mask = (~0u) >> (32 - bits_per_variant);
        u32 bit_index = palette_region_voxel_index * bits_per_variant;
        u32 data_index = bit_index / 32;
        u32 data_offset = bit_index - data_index * 32;
        u32 data = (my_palette_index - 1) & mask;
        u32 address = palette_size + data_index;
        // clang-format off
        atomicAnd(compression_result[address + 0], ~(mask << data_offset));
        atomicOr (compression_result[address + 0],   data << data_offset);
        if (data_offset + bits_per_variant > 32) {
            u32 shift = bits_per_variant - ((data_offset + bits_per_variant) & 0x1f);
            atomicAnd(compression_result[address + 1], ~(mask >> shift));
            atomicOr (compression_result[address + 1],   data >> shift);
        }
        // clang-format on
    } else {
        if (palette_region_voxel_index == 0) {
            if (prev_variant_n > 1) {
                VoxelMalloc_free(voxel_malloc_page_allocator, voxel_chunk_ptr, prev_blob_ptr);
            }
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].variant_n = palette_size;
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].blob_ptr = my_voxel;
        }
    }

    barrier();
    memoryBarrierShared();

    if (palette_region_voxel_index < compressed_size) {
        daxa_RWBufferPtr(daxa_u32) blob_u32s;
        voxel_malloc_address_to_u32_ptr(daxa_BufferPtr(VoxelMallocPageAllocator)(voxel_malloc_page_allocator), blob_ptr, blob_u32s);
        deref(blob_u32s[palette_region_voxel_index]) = compression_result[palette_region_voxel_index];
    }
}
#undef VOXEL_WORLD

#endif
