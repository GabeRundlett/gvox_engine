#include <voxels/impl/voxel_world.inl>

#if PerChunkComputeShader

DAXA_DECL_PUSH_CONSTANT(PerChunkComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(GpuGvoxModel) gvox_model = push.uses.gvox_model;
daxa_RWBufferPtr(VoxelWorldGlobals) voxel_globals = push.uses.voxel_globals;
daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunks = push.uses.voxel_chunks;
daxa_ImageViewIndex value_noise_texture = push.uses.value_noise_texture;

#include <utilities/gpu/math.glsl>
#include <voxels/impl/voxels.glsl>

#define VOXEL_WORLD deref(voxel_globals)
#define PLAYER deref(gpu_input).player
#define CHUNKS(i) deref(advance(voxel_chunks, i))
#define INDIRECT deref(voxel_globals).indirect_dispatch

void try_elect(in out VoxelChunkUpdateInfo work_item, in out uint update_index) {
    uint prev_update_n = atomicAdd(VOXEL_WORLD.chunk_update_n, 1);

    // Check if the work item can be added
    if (prev_update_n < MAX_CHUNK_UPDATES_PER_FRAME) {
        // Set the chunk edit dispatch z axis (64/8, 64/8, 64 x 8 x 8 / 8 = 64 x 8) = (8, 8, 512)
        atomicAdd(INDIRECT.chunk_edit_dispatch.z, CHUNK_SIZE / 8);
        atomicAdd(INDIRECT.subchunk_x2x4_dispatch.z, 1);
        atomicAdd(INDIRECT.subchunk_x8up_dispatch.z, 1);
        // Set the chunk update info
        VOXEL_WORLD.chunk_update_infos[prev_update_n] = work_item;
        update_index = prev_update_n + 1;
    }
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    ivec3 chunk_n = ivec3(1 << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);

    VoxelChunkUpdateInfo terrain_work_item;
    terrain_work_item.i = ivec3(gl_GlobalInvocationID.xyz) & (chunk_n - 1);

    ivec3 offset = (VOXEL_WORLD.offset >> ivec3(6 + LOG2_VOXEL_SIZE));
    ivec3 prev_offset = (VOXEL_WORLD.prev_offset >> ivec3(6 + LOG2_VOXEL_SIZE));

    terrain_work_item.chunk_offset = offset;
    terrain_work_item.brush_flags = BRUSH_FLAGS_WORLD_BRUSH;

    // (const) number of chunks in each axis
    uint chunk_index = calc_chunk_index_from_worldspace(terrain_work_item.i, chunk_n);

    uint update_index = 0;

    if ((CHUNKS(chunk_index).flags & CHUNK_FLAGS_ACCEL_GENERATED) == 0) {
        try_elect(terrain_work_item, update_index);
    } else if (offset != prev_offset) {
        // invalidate chunks outside the chunk_offset
        ivec3 diff = clamp(ivec3(offset - prev_offset), -chunk_n, chunk_n);

        ivec3 start;
        ivec3 end;

        start.x = diff.x < 0 ? 0 : chunk_n.x - diff.x;
        end.x = diff.x < 0 ? -diff.x : chunk_n.x;

        start.y = diff.y < 0 ? 0 : chunk_n.y - diff.y;
        end.y = diff.y < 0 ? -diff.y : chunk_n.y;

        start.z = diff.z < 0 ? 0 : chunk_n.z - diff.z;
        end.z = diff.z < 0 ? -diff.z : chunk_n.z;

        uvec3 temp_chunk_i = uvec3((ivec3(terrain_work_item.i) - offset) % ivec3(chunk_n));

        if ((temp_chunk_i.x >= start.x && temp_chunk_i.x < end.x) ||
            (temp_chunk_i.y >= start.y && temp_chunk_i.y < end.y) ||
            (temp_chunk_i.z >= start.z && temp_chunk_i.z < end.z)) {
            CHUNKS(chunk_index).flags &= ~CHUNK_FLAGS_ACCEL_GENERATED;
            try_elect(terrain_work_item, update_index);
        }
    } else {
        // Wrapped chunk index in leaf chunk space (0^3 - 31^3)
        ivec3 wrapped_chunk_i = imod3(terrain_work_item.i - imod3(terrain_work_item.chunk_offset - ivec3(chunk_n), ivec3(chunk_n)), ivec3(chunk_n));
        // Leaf chunk position in world space
        ivec3 world_chunk = terrain_work_item.chunk_offset + wrapped_chunk_i - ivec3(chunk_n / 2);

        terrain_work_item.brush_input = VOXEL_WORLD.brush_input;

        ivec3 brush_chunk = (ivec3(floor(VOXEL_WORLD.brush_input.pos)) + VOXEL_WORLD.brush_input.pos_offset) >> (6 + LOG2_VOXEL_SIZE);
        bool is_near_brush = all(greaterThanEqual(world_chunk, brush_chunk - 1)) && all(lessThanEqual(world_chunk, brush_chunk + 1));

        if (is_near_brush && deref(gpu_input).actions[GAME_ACTION_BRUSH_A] != 0) {
            terrain_work_item.brush_flags = BRUSH_FLAGS_USER_BRUSH_A;
            try_elect(terrain_work_item, update_index);
        } else if (is_near_brush && deref(gpu_input).actions[GAME_ACTION_BRUSH_B] != 0) {
            terrain_work_item.brush_flags = BRUSH_FLAGS_USER_BRUSH_B;
            try_elect(terrain_work_item, update_index);
        }
    }

    CHUNKS(chunk_index).update_index = update_index;
}

#undef INDIRECT
#undef CHUNKS
#undef PLAYER
#undef VOXEL_WORLD

#endif

#if ChunkEditComputeShader

DAXA_DECL_PUSH_CONSTANT(ChunkEditComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(GpuGvoxModel) gvox_model = push.uses.gvox_model;
daxa_BufferPtr(VoxelWorldGlobals) voxel_globals = push.uses.voxel_globals;
daxa_BufferPtr(VoxelLeafChunk) voxel_chunks = push.uses.voxel_chunks;
daxa_BufferPtr(VoxelMallocPageAllocator) voxel_malloc_page_allocator = push.uses.voxel_malloc_page_allocator;
daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunks = push.uses.temp_voxel_chunks;
daxa_ImageViewIndex value_noise_texture = push.uses.value_noise_texture;
daxa_ImageViewIndex test_texture = push.uses.test_texture;
daxa_ImageViewIndex test_texture2 = push.uses.test_texture2;

#include <utilities/gpu/math.glsl>
#include <utilities/gpu/noise.glsl>
#include <voxels/impl/voxels.glsl>

uvec3 chunk_n;
uint temp_chunk_index;
ivec3 chunk_i;
uint chunk_index;
daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr;
daxa_BufferPtr(VoxelLeafChunk) voxel_chunk_ptr;
uvec3 inchunk_voxel_i;
ivec3 voxel_i;
ivec3 world_voxel;
vec3 voxel_pos;
BrushInput brush_input;

#include "../brushes.glsl"

#define VOXEL_WORLD deref(voxel_globals)
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    // (const) number of chunks in each axis
    chunk_n = uvec3(1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);
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
    ivec3 chunk_offset = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].chunk_offset;
    // Brush informations
    brush_input = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].brush_input;
    // Brush flags
    uint brush_flags = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].brush_flags;
    // Chunk uint index in voxel_chunks buffer
    chunk_index = calc_chunk_index_from_worldspace(chunk_i, chunk_n);
    // Pointer to the previous chunk
    temp_voxel_chunk_ptr = advance(temp_voxel_chunks, temp_chunk_index);
    // Pointer to the new chunk
    voxel_chunk_ptr = advance(voxel_chunks, chunk_index);
    // Voxel offset in chunk
    inchunk_voxel_i = gl_GlobalInvocationID.xyz - uvec3(0, 0, temp_chunk_index * CHUNK_SIZE);
    // Voxel 3D position (in voxel buffer)
    voxel_i = chunk_i * CHUNK_SIZE + ivec3(inchunk_voxel_i);

    // Wrapped chunk index in leaf chunk space (0^3 - 31^3)
    ivec3 wrapped_chunk_i = imod3(chunk_i - imod3(chunk_offset - ivec3(chunk_n), ivec3(chunk_n)), ivec3(chunk_n));
    // Leaf chunk position in world space
    ivec3 world_chunk = chunk_offset + wrapped_chunk_i - ivec3(chunk_n / 2);

    // Voxel position in world space (voxels)
    world_voxel = world_chunk * CHUNK_SIZE + ivec3(inchunk_voxel_i);
    // Voxel position in world space (meters)
    voxel_pos = vec3(world_voxel) * VOXEL_SIZE;

    rand_seed(voxel_i.x + voxel_i.y * 1000 + voxel_i.z * 1000 * 1000);

    Voxel result = Voxel(0, 0, vec3(0, 0, 1), vec3(0));

    if ((brush_flags & BRUSH_FLAGS_WORLD_BRUSH) != 0) {
        brushgen_world(result);
    }
    if ((brush_flags & BRUSH_FLAGS_USER_BRUSH_A) != 0) {
        brushgen_a(result);
    }
    if ((brush_flags & BRUSH_FLAGS_USER_BRUSH_B) != 0) {
        brushgen_b(result);
    }
    // if ((brush_flags & BRUSH_FLAGS_PARTICLE_BRUSH) != 0) {
    //     brushgen_particles(col, id);
    // }

    PackedVoxel packed_result = pack_voxel(result);
    // result.col_and_id = vec4_to_uint_rgba8(vec4(col, 0.0)) | (id << 0x18);
    deref(temp_voxel_chunk_ptr).voxels[inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE] = packed_result;
}
#undef VOXEL_WORLD

#endif

#if ChunkEditPostProcessComputeShader

DAXA_DECL_PUSH_CONSTANT(ChunkEditPostProcessComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(GpuGvoxModel) gvox_model = push.uses.gvox_model;
daxa_BufferPtr(VoxelWorldGlobals) voxel_globals = push.uses.voxel_globals;
daxa_BufferPtr(VoxelLeafChunk) voxel_chunks = push.uses.voxel_chunks;
daxa_BufferPtr(VoxelMallocPageAllocator) voxel_malloc_page_allocator = push.uses.voxel_malloc_page_allocator;
daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunks = push.uses.temp_voxel_chunks;
daxa_ImageViewIndex value_noise_texture = push.uses.value_noise_texture;

#include <utilities/gpu/math.glsl>
#include <utilities/gpu/noise.glsl>
#include <voxels/impl/voxels.glsl>

uvec3 chunk_n;
uint temp_chunk_index;
ivec3 chunk_i;
uint chunk_index;
daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr;
daxa_BufferPtr(VoxelLeafChunk) voxel_chunk_ptr;
uvec3 inchunk_voxel_i;
ivec3 voxel_i;
ivec3 world_voxel;
vec3 voxel_pos;
BrushInput brush_input;

Voxel get_temp_voxel(ivec3 offset_i) {
    // TODO: Simplify this, and improve precision
    vec3 i = vec3(world_voxel + offset_i) * VOXEL_SIZE - deref(gpu_input).player.player_unit_offset;
    vec3 offset = vec3((deref(voxel_globals).offset) & ((1 << (6 + LOG2_VOXEL_SIZE)) - 1)) + vec3(chunk_n) * CHUNK_WORLDSPACE_SIZE * 0.5;
    uvec3 voxel_i = uvec3(floor((i + offset) * VOXEL_SCL));
    Voxel default_value = Voxel(0, 0, vec3(0), vec3(0));
    if (any(greaterThanEqual(voxel_i, uvec3(CHUNK_SIZE * chunk_n)))) {
        return default_value;
    }
    return unpack_voxel(sample_temp_voxel_chunk(
        voxel_globals,
        voxel_malloc_page_allocator,
        voxel_chunks,
        temp_voxel_chunks,
        chunk_n, voxel_i));
}

bool has_air_neighbor() {
    bool result = false;

    {
        Voxel v = get_temp_voxel(ivec3(-1, 0, 0));
        if (v.material_type == 0) {
            result = true;
        }
    }
    {
        Voxel v = get_temp_voxel(ivec3(+1, 0, 0));
        if (v.material_type == 0) {
            result = true;
        }
    }
    {
        Voxel v = get_temp_voxel(ivec3(0, -1, 0));
        if (v.material_type == 0) {
            result = true;
        }
    }
    {
        Voxel v = get_temp_voxel(ivec3(0, +1, 0));
        if (v.material_type == 0) {
            result = true;
        }
    }
    {
        Voxel v = get_temp_voxel(ivec3(0, 0, -1));
        if (v.material_type == 0) {
            result = true;
        }
    }
    {
        Voxel v = get_temp_voxel(ivec3(0, 0, +1));
        if (v.material_type == 0) {
            result = true;
        }
    }

    return result;
}

vec3 generate_normal_from_geometry() {
    vec3 density_n = vec3(0);
    vec3 density_p = vec3(0);
    const int RADIUS = 2;
    for (int zi = -RADIUS; zi <= RADIUS; ++zi) {
        for (int yi = -RADIUS; yi <= RADIUS; ++yi) {
            for (int xi = -RADIUS; xi <= RADIUS; ++xi) {
                Voxel v = get_temp_voxel(ivec3(xi, yi, zi));
                if (v.material_type == 0) {
                    vec3 dir = vec3(xi, yi, zi);
                    density_n.x += max(0.0, dot(dir, vec3(-1, 0, 0)));
                    density_p.x += max(0.0, dot(dir, vec3(+1, 0, 0)));
                    density_n.y += max(0.0, dot(dir, vec3(0, -1, 0)));
                    density_p.y += max(0.0, dot(dir, vec3(0, +1, 0)));
                    density_n.z += max(0.0, dot(dir, vec3(0, 0, -1)));
                    density_p.z += max(0.0, dot(dir, vec3(0, 0, +1)));
                }
            }
        }
    }

    vec3 d = density_p - density_n;
    if (dot(d, d) < 0.1) {
        // Hack to fix flat sides. TODO: Generalize
        vec3 v = density_p + density_n;
        float min_v = min(v.x, min(v.y, v.z));
        float max_v = max(v.x, max(v.y, v.z));
        if (min_v == v.x) {
            if (max_v == v.z) {
                d = vec3(0, 0, 1);
            } else {
                d = vec3(0, 1, 0);
            }
        } else if (min_v == v.y) {
            if (max_v == v.z) {
                d = vec3(0, 0, 1);
            } else {
                d = vec3(1, 0, 0);
            }
        } else {
            if (max_v == v.x) {
                d = vec3(1, 0, 0);
            } else {
                d = vec3(0, 1, 0);
            }
        }
    }

    return normalize(d);
}

#define VOXEL_WORLD deref(voxel_globals)
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    // (const) number of chunks in each axis
    chunk_n = uvec3(1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);
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
    ivec3 chunk_offset = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].chunk_offset;
    // Brush informations
    brush_input = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].brush_input;
    // Brush flags
    uint brush_flags = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].brush_flags;
    // Chunk uint index in voxel_chunks buffer
    chunk_index = calc_chunk_index_from_worldspace(chunk_i, chunk_n);
    // Pointer to the previous chunk
    temp_voxel_chunk_ptr = advance(temp_voxel_chunks, temp_chunk_index);
    // Pointer to the new chunk
    voxel_chunk_ptr = advance(voxel_chunks, chunk_index);
    // Voxel offset in chunk
    inchunk_voxel_i = gl_GlobalInvocationID.xyz - uvec3(0, 0, temp_chunk_index * CHUNK_SIZE);
    // Voxel 3D position (in voxel buffer)
    voxel_i = chunk_i * CHUNK_SIZE + ivec3(inchunk_voxel_i);

    // Wrapped chunk index in leaf chunk space (0^3 - 31^3)
    ivec3 wrapped_chunk_i = imod3(chunk_i - imod3(chunk_offset - ivec3(chunk_n), ivec3(chunk_n)), ivec3(chunk_n));
    // Leaf chunk position in world space
    ivec3 world_chunk = chunk_offset + wrapped_chunk_i - ivec3(chunk_n / 2);

    // Voxel position in world space (voxels)
    world_voxel = world_chunk * CHUNK_SIZE + ivec3(inchunk_voxel_i);
    // Voxel position in world space (meters)
    voxel_pos = vec3(world_voxel) * VOXEL_SIZE;

    rand_seed(voxel_i.x + voxel_i.y * 1000 + voxel_i.z * 1000 * 1000);

    PackedVoxel packed_result = deref(temp_voxel_chunk_ptr).voxels[inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE];
    Voxel result = unpack_voxel(packed_result);

    if (result.material_type == 0) {
        result.normal = vec3(0, 0, 1);
        result.color = vec3(0.0);
        result.roughness = 0;
    } else {
        bool is_occluded = !has_air_neighbor();
        if (is_occluded) {
            // nullify normal
            result.normal = vec3(0, 0, 1);
            // result.color = vec3(0.9);
        } else {
            // potentially generate a normal
            // if the voxel normal is the "null" normal AKA up
            // bool generate_normal = true;
            bool generate_normal = (result.normal == unpack_voxel(pack_voxel(Voxel(0, 0, vec3(0, 0, 1), vec3(0)))).normal);
            if (generate_normal) {
                result.normal = generate_normal_from_geometry();
            }
            result.normal = normalize(result.normal);
        }
    }

    packed_result = pack_voxel(result);
    // result.col_and_id = vec4_to_uint_rgba8(vec4(col, 0.0)) | (id << 0x18);
    deref(temp_voxel_chunk_ptr).voxels[inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE] = packed_result;
}
#undef VOXEL_WORLD

#endif

#if ChunkOptComputeShader

DAXA_DECL_PUSH_CONSTANT(ChunkOptComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(VoxelWorldGlobals) voxel_globals = push.uses.voxel_globals;
daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunks = push.uses.temp_voxel_chunks;
daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunks = push.uses.voxel_chunks;

#include <voxels/impl/voxels.glsl>

#define WAVE_SIZE gl_SubgroupSize
#define WAVE_SIZE_MUL (WAVE_SIZE / 32)

uint sample_temp_voxel_id(daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr, in uvec3 in_chunk_i) {
    uint in_chunk_index = in_chunk_i.x + in_chunk_i.y * CHUNK_SIZE + in_chunk_i.z * CHUNK_SIZE * CHUNK_SIZE;
    Voxel voxel = unpack_voxel(deref(temp_voxel_chunk_ptr).voxels[in_chunk_index]);
    return voxel.material_type;
}

// For now, I'm testing with using non-zero as the accel structure, instead of uniformity.
uint sample_base_voxel_id(daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr, in uvec3 in_chunk_i) {
#if VOXEL_ACCEL_UNIFORMITY
    return sample_temp_voxel_id(temp_voxel_chunk_ptr, in_chunk_i);
#else
    return 0;
#endif
}

#if CHUNK_OPT_STAGE == 0

shared uint local_x2_copy[4][4];

#define VOXEL_CHUNK deref(temp_voxel_chunk_ptr)
void chunk_opt_x2x4(daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr, daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunk_ptr, in uint chunk_local_workgroup) {
    uvec2 x2_in_group_location = uvec2(
        (gl_LocalInvocationID.x >> 5) & 0x3,
        (gl_LocalInvocationID.x >> 7) & 0x3);
    uvec3 x2_i = uvec3(
        (gl_LocalInvocationID.x >> 5) & 0x3,
        (gl_LocalInvocationID.x >> 7) & 0x3,
        (gl_LocalInvocationID.x & 0x1F));
    x2_i += 4 * uvec3(chunk_local_workgroup & 0x7, (chunk_local_workgroup >> 3) & 0x7, 0);
    uvec3 in_chunk_i = x2_i * 2;
    bool at_least_one_occluding = false;
    uint base_id_x1 = sample_base_voxel_id(temp_voxel_chunk_ptr, in_chunk_i);
    for (uint x = 0; x < 2; ++x)
        for (uint y = 0; y < 2; ++y)
            for (uint z = 0; z < 2; ++z) {
                uvec3 local_i = in_chunk_i + uvec3(x, y, z); // in x1 space
                at_least_one_occluding = at_least_one_occluding || (sample_temp_voxel_id(temp_voxel_chunk_ptr, local_i) != base_id_x1);
            }
    uint result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(2)(x2_i);
    }
    for (uint i = 0; i < 1 * WAVE_SIZE_MUL; i++) {
        if ((gl_SubgroupInvocationID >> 5) == i) {
            result = subgroupOr(result);
        }
    }
    if ((gl_SubgroupInvocationID & 0x1F /* = %32 */) == 0) {
        uint index = uniformity_lod_index(2)(x2_i);
        VOXEL_CHUNK.uniformity.lod_x2[index] = result;
        local_x2_copy[x2_in_group_location.x][x2_in_group_location.y] = result;
    }
    subgroupBarrier();
    if (gl_LocalInvocationID.x >= 64) {
        return;
    }
    uvec3 x4_i = uvec3(
        (gl_LocalInvocationID.x >> 4) & 0x1,
        (gl_LocalInvocationID.x >> 5) & 0x1,
        gl_LocalInvocationID.x & 0xF);
    x4_i += 2 * uvec3(chunk_local_workgroup & 0x7, (chunk_local_workgroup >> 3) & 0x7, 0);
    x2_i = x4_i * 2;
    uint base_id_x2 = sample_base_voxel_id(temp_voxel_chunk_ptr, x2_i * 2);
    at_least_one_occluding = false;
    for (uint x = 0; x < 2; ++x)
        for (uint y = 0; y < 2; ++y)
            for (uint z = 0; z < 2; ++z) {
                uvec3 local_i = x2_i + uvec3(x, y, z); // in x2 space
                uint mask = uniformity_lod_mask(2)(local_i);
                uvec2 x2_in_group_index = uvec2(
                    local_i.x & 0x3,
                    local_i.y & 0x3);
                bool is_occluding = (local_x2_copy[x2_in_group_index.x][x2_in_group_index.y] & mask) != 0;
                at_least_one_occluding = at_least_one_occluding || is_occluding || (sample_temp_voxel_id(temp_voxel_chunk_ptr, local_i * 2) != base_id_x2);
            }
    result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(4)(x4_i);
    }
    for (uint i = 0; i < 2 * WAVE_SIZE_MUL; i++) {
        if ((gl_SubgroupInvocationID >> 4) == i) {
            result = subgroupOr(result);
        }
    }
    if ((gl_SubgroupInvocationID & 0xF /* = %16 */) == 0) {
        uint index = uniformity_lod_index(4)(x4_i);
        VOXEL_CHUNK.uniformity.lod_x4[index] = result;
    }
}
#undef VOXEL_CHUNK

#define VOXEL_WORLD deref(voxel_globals)
layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
void main() {
    ivec3 chunk_i = VOXEL_WORLD.chunk_update_infos[gl_WorkGroupID.z].i;
    if (chunk_i == INVALID_CHUNK_I) {
        return;
    }
    uvec3 chunk_n;
    chunk_n.x = 1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS;
    chunk_n.y = chunk_n.x;
    chunk_n.z = chunk_n.x;
    uint chunk_index = calc_chunk_index_from_worldspace(chunk_i, chunk_n);
    daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr = advance(temp_voxel_chunks, gl_WorkGroupID.z);
    daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunk_ptr = advance(voxel_chunks, chunk_index);
    chunk_opt_x2x4(temp_voxel_chunk_ptr, voxel_chunk_ptr, gl_WorkGroupID.y);
}
#undef VOXEL_WORLD

#endif

#if CHUNK_OPT_STAGE == 1

shared uint local_x8_copy[64];
shared uint local_x16_copy[16];
shared uint local_x32_copy[4];

#define VOXEL_CHUNK deref(temp_voxel_chunk_ptr)
void chunk_opt_x8up(daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr, daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunk_ptr) {
    uvec3 x8_i = uvec3(
        (gl_LocalInvocationID.x >> 3) & 0x7,
        (gl_LocalInvocationID.x >> 6) & 0x7,
        gl_LocalInvocationID.x & 0x7);
    uvec3 x4_i = x8_i * 2;

    uint base_id_x4 = sample_base_voxel_id(temp_voxel_chunk_ptr, x4_i * 4);

    bool at_least_one_occluding = false;
    for (uint x = 0; x < 2; ++x)
        for (uint y = 0; y < 2; ++y)
            for (uint z = 0; z < 2; ++z) {
                uvec3 local_i = x4_i + uvec3(x, y, z); // x4 space
                uint index = uniformity_lod_index(4)(local_i);
                uint mask = uniformity_lod_mask(4)(local_i);
                bool occluding = (VOXEL_CHUNK.uniformity.lod_x4[index] & mask) != 0;
                at_least_one_occluding = at_least_one_occluding || occluding || (sample_temp_voxel_id(temp_voxel_chunk_ptr, local_i * 4) != base_id_x4);
            }

    uint result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(8)(x8_i);
    }
    for (int i = 0; i < 4 * WAVE_SIZE_MUL; i++) {
        if ((gl_SubgroupInvocationID >> 3) == i) {
            result = subgroupOr(result);
        }
    }
    if ((gl_SubgroupInvocationID & 0x7 /* == % 8*/) == 0) {
        uint index = uniformity_lod_index(8)(x8_i);
        VOXEL_CHUNK.uniformity.lod_x8[index] = result;
        local_x8_copy[index] = result;
    }

    subgroupBarrier();

    if (gl_LocalInvocationID.x >= 64) {
        return;
    }

    uvec3 x16_i = uvec3(
        (gl_LocalInvocationID.x >> 2) & 0x3,
        (gl_LocalInvocationID.x >> 4) & 0x3,
        gl_LocalInvocationID.x & 0x3);
    x8_i = x16_i * 2;
    uint base_id_x8 = sample_base_voxel_id(temp_voxel_chunk_ptr, x8_i * 8);

    at_least_one_occluding = false;
    for (uint x = 0; x < 2; ++x)
        for (uint y = 0; y < 2; ++y)
            for (uint z = 0; z < 2; ++z) {
                uvec3 local_i = x8_i + uvec3(x, y, z); // x8 space
                uint mask = uniformity_lod_mask(8)(local_i);
                uint index = uniformity_lod_index(8)(local_i);
                bool is_occluding = (local_x8_copy[index] & mask) != 0;
                at_least_one_occluding = at_least_one_occluding || is_occluding || (sample_temp_voxel_id(temp_voxel_chunk_ptr, local_i * 8) != base_id_x8);
            }

    result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(16)(x16_i);
    }
    for (int i = 0; i < 8 * WAVE_SIZE_MUL; i++) {
        if ((gl_SubgroupInvocationID >> 2) == i) {
            result = subgroupOr(result);
        }
    }
    if ((gl_SubgroupInvocationID & 0x3) == 0) {
        uint index = uniformity_lod_index(16)(x16_i);
        VOXEL_CHUNK.uniformity.lod_x16[index] = result;
        local_x16_copy[index] = result;
    }

    subgroupBarrier();

    if (gl_LocalInvocationID.x >= 8) {
        return;
    }

    uvec3 x32_i = uvec3(
        (gl_LocalInvocationID.x >> 1) & 0x1,
        (gl_LocalInvocationID.x >> 2) & 0x1,
        gl_LocalInvocationID.x & 0x1);
    x16_i = x32_i * 2;
    uint base_id_x16 = sample_base_voxel_id(temp_voxel_chunk_ptr, x16_i * 16);

    at_least_one_occluding = false;
    for (uint x = 0; x < 2; ++x)
        for (uint y = 0; y < 2; ++y)
            for (uint z = 0; z < 2; ++z) {
                uvec3 local_i = x16_i + uvec3(x, y, z); // x16 space
                uint mask = uniformity_lod_mask(16)(local_i);
                uint index = uniformity_lod_index(16)(local_i);
                bool is_occluding = (local_x16_copy[index] & mask) != 0;
                at_least_one_occluding = at_least_one_occluding || is_occluding || (sample_temp_voxel_id(temp_voxel_chunk_ptr, local_i * 16) != base_id_x16);
            }

    result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(32)(x32_i);
    }
    for (int i = 0; i < 16 * WAVE_SIZE_MUL; i++) {
        if ((gl_SubgroupInvocationID >> 1) == i) {
            result = subgroupOr(result);
        }
    }
    if ((gl_SubgroupInvocationID & 0x1) == 0) {
        uint index = uniformity_lod_index(32)(x32_i);
        VOXEL_CHUNK.uniformity.lod_x32[index] = result;
        local_x32_copy[index] = result;
    }

    subgroupBarrier();

    if (gl_LocalInvocationID.x >= 1) {
        return;
    }

    uvec3 x64_i = uvec3(0);
    x32_i = x64_i * 2;
    uint base_id_x32 = sample_base_voxel_id(temp_voxel_chunk_ptr, x32_i * 32);

    at_least_one_occluding = false;
    for (uint x = 0; x < 2; ++x)
        for (uint y = 0; y < 2; ++y)
            for (uint z = 0; z < 2; ++z) {
                uvec3 local_i = x32_i + uvec3(x, y, z); // x32 space
                uint mask = uniformity_lod_mask(32)(local_i);
                uint index = uniformity_lod_index(32)(local_i);
                bool is_occluding = (local_x32_copy[index] & mask) != 0;
                at_least_one_occluding = at_least_one_occluding || is_occluding || (sample_temp_voxel_id(temp_voxel_chunk_ptr, local_i * 32) != base_id_x32);
            }

    // TODO, remove for a parallel option above instead

    uint uniformity_bits[3];
    uniformity_bits[0] = 0x0;
    uniformity_bits[1] = 0x0;
    uniformity_bits[2] = 0x0;

    for (int x = 0; x < 1; ++x)
        for (int y = 0; y < 1; ++y)
            for (int z = 0; z < 1; ++z) {
                bool has_occluding = at_least_one_occluding;
                uint new_index = new_uniformity_lod_index(64)(uvec3(x, y, z));
                uint new_mask = new_uniformity_lod_mask(64)(uvec3(x, y, z));
                if (has_occluding) {
                    uniformity_bits[new_index] |= new_mask;
                }
            }
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z) {
                uint index = uniformity_lod_index(32)(uvec3(x, y, z));
                uint mask = uniformity_lod_mask(32)(uvec3(x, y, z));
                bool has_occluding = (local_x32_copy[index] & mask) != 0;
                uint new_index = new_uniformity_lod_index(32)(uvec3(x, y, z));
                uint new_mask = new_uniformity_lod_mask(32)(uvec3(x, y, z));
                if (has_occluding) {
                    uniformity_bits[new_index] |= new_mask;
                }
            }
    for (int x = 0; x < 4; ++x)
        for (int y = 0; y < 4; ++y)
            for (int z = 0; z < 4; ++z) {
                uint index = uniformity_lod_index(16)(uvec3(x, y, z));
                uint mask = uniformity_lod_mask(16)(uvec3(x, y, z));
                bool has_occluding = (local_x16_copy[index] & mask) != 0;
                uint new_index = new_uniformity_lod_index(16)(uvec3(x, y, z));
                uint new_mask = new_uniformity_lod_mask(16)(uvec3(x, y, z));
                if (has_occluding) {
                    uniformity_bits[new_index] |= new_mask;
                }
            }

    deref(voxel_chunk_ptr).uniformity_bits[0] = uniformity_bits[0];
    deref(voxel_chunk_ptr).uniformity_bits[1] = uniformity_bits[1];
    deref(voxel_chunk_ptr).uniformity_bits[2] = uniformity_bits[2];
}
#undef VOXEL_CHUNK

#define VOXEL_WORLD deref(voxel_globals)
layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
void main() {
    ivec3 chunk_i = VOXEL_WORLD.chunk_update_infos[gl_WorkGroupID.z].i;
    if (chunk_i == INVALID_CHUNK_I) {
        return;
    }
    uvec3 chunk_n;
    chunk_n.x = 1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS;
    chunk_n.y = chunk_n.x;
    chunk_n.z = chunk_n.x;
    uint chunk_index = calc_chunk_index_from_worldspace(chunk_i, chunk_n);
    daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr = advance(temp_voxel_chunks, gl_WorkGroupID.z);
    daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunk_ptr = advance(voxel_chunks, chunk_index);
    chunk_opt_x8up(temp_voxel_chunk_ptr, voxel_chunk_ptr);

    // Finish the chunk
    if (gl_LocalInvocationIndex == 0) {
        deref(voxel_chunk_ptr).flags = CHUNK_FLAGS_ACCEL_GENERATED;
    }
}
#undef VOXEL_WORLD

#endif

#endif

#if ChunkAllocComputeShader

DAXA_DECL_PUSH_CONSTANT(ChunkAllocComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(VoxelWorldGlobals) voxel_globals = push.uses.voxel_globals;
daxa_BufferPtr(TempVoxelChunk) temp_voxel_chunks = push.uses.temp_voxel_chunks;
daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunks = push.uses.voxel_chunks;
daxa_RWBufferPtr(VoxelMallocPageAllocator) voxel_malloc_page_allocator = push.uses.voxel_malloc_page_allocator;

#extension GL_EXT_shader_atomic_int64 : require

#include <utilities/gpu/math.glsl>
#include <voxels/impl/voxel_malloc.glsl>
#include <voxels/impl/voxels.glsl>

shared uint compression_result[PALETTE_REGION_TOTAL_SIZE];
shared daxa_u64 voted_results[PALETTE_REGION_TOTAL_SIZE];
shared uint palette_size;

void process_palette_region(uint palette_region_voxel_index, uint my_voxel, in out uint my_palette_index) {
    if (palette_region_voxel_index == 0) {
        palette_size = 0;
    }
    voted_results[palette_region_voxel_index] = 0;
    barrier();
    for (uint algo_i = 0; algo_i < PALETTE_MAX_COMPRESSED_VARIANT_N + 1; ++algo_i) {
        if (my_palette_index == 0) {
            daxa_u64 vote_result = atomicCompSwap(voted_results[algo_i], 0, daxa_u64(my_voxel) | (daxa_u64(1) << daxa_u64(32)));
            if (vote_result == 0) {
                my_palette_index = algo_i + 1;
                compression_result[palette_size] = my_voxel;
                palette_size++;
            } else if (my_voxel == uint(vote_result)) {
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
    uvec3 chunk_n = uvec3(1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);
    uint temp_chunk_index = gl_GlobalInvocationID.z / CHUNK_SIZE;
    ivec3 chunk_i = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].i;
    if (chunk_i == INVALID_CHUNK_I) {
        return;
    }
    uint chunk_index = calc_chunk_index_from_worldspace(chunk_i, chunk_n);
    uvec3 inchunk_voxel_i = gl_GlobalInvocationID.xyz - uvec3(0, 0, temp_chunk_index * CHUNK_SIZE);
    uint inchunk_voxel_index = inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE;
    uint palette_region_voxel_index =
        gl_LocalInvocationID.x +
        gl_LocalInvocationID.y * PALETTE_REGION_SIZE +
        gl_LocalInvocationID.z * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE;
    uvec3 palette_i = uvec3(gl_WorkGroupID.xy, gl_WorkGroupID.z - temp_chunk_index * PALETTES_PER_CHUNK_AXIS);
    uint palette_region_index =
        palette_i.x +
        palette_i.y * PALETTES_PER_CHUNK_AXIS +
        palette_i.z * PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS;

    daxa_BufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr = advance(temp_voxel_chunks, temp_chunk_index);
    daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunk_ptr = advance(voxel_chunks, chunk_index);

    uint my_voxel = deref(temp_voxel_chunk_ptr).voxels[inchunk_voxel_index].data;
    uint my_palette_index = 0;

    process_palette_region(palette_region_voxel_index, my_voxel, my_palette_index);

    uint prev_variant_n = deref(voxel_chunk_ptr).palette_headers[palette_region_index].variant_n;
    VoxelMalloc_Pointer prev_blob_ptr = deref(voxel_chunk_ptr).palette_headers[palette_region_index].blob_ptr;

    uint bits_per_variant = ceil_log2(palette_size);

    uint compressed_size = 0;
    VoxelMalloc_Pointer blob_ptr = my_voxel;

    if (palette_size > PALETTE_MAX_COMPRESSED_VARIANT_N) {
        compressed_size = PALETTE_REGION_TOTAL_SIZE;
        if (prev_variant_n > 1) {
            blob_ptr = prev_blob_ptr;
            VoxelMalloc_realloc(voxel_malloc_page_allocator, voxel_chunk_ptr, blob_ptr, PALETTE_ACCELERATION_STRUCTURE_SIZE_U32S + compressed_size);
        } else {
            blob_ptr = VoxelMalloc_malloc(voxel_malloc_page_allocator, voxel_chunk_ptr, PALETTE_ACCELERATION_STRUCTURE_SIZE_U32S + compressed_size);
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
            VoxelMalloc_realloc(voxel_malloc_page_allocator, voxel_chunk_ptr, blob_ptr, PALETTE_ACCELERATION_STRUCTURE_SIZE_U32S + compressed_size);
        } else {
            blob_ptr = VoxelMalloc_malloc(voxel_malloc_page_allocator, voxel_chunk_ptr, PALETTE_ACCELERATION_STRUCTURE_SIZE_U32S + compressed_size);
        }
        if (palette_region_voxel_index == 0) {
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].variant_n = palette_size;
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].blob_ptr = blob_ptr;
        }

        uint mask = (~0u) >> (32 - bits_per_variant);
        uint bit_index = palette_region_voxel_index * bits_per_variant;
        uint data_index = bit_index / 32;
        uint data_offset = bit_index - data_index * 32;
        uint data = (my_palette_index - 1) & mask;
        uint address = palette_size + data_index;
        // clang-format off
        atomicAnd(compression_result[address + 0], ~(mask << data_offset));
        atomicOr (compression_result[address + 0],   data << data_offset);
        if (data_offset + bits_per_variant > 32) {
            uint shift = bits_per_variant - ((data_offset + bits_per_variant) & 0x1f);
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
        daxa_RWBufferPtr(uint) blob_u32s;
        voxel_malloc_address_to_u32_ptr(daxa_BufferPtr(VoxelMallocPageAllocator)(as_address(voxel_malloc_page_allocator)), blob_ptr, blob_u32s);
        deref(advance(blob_u32s, PALETTE_ACCELERATION_STRUCTURE_SIZE_U32S + palette_region_voxel_index)) = compression_result[palette_region_voxel_index];
    }

    if (palette_size > 1 && palette_region_voxel_index < 1) {
        // write accel structure
        // TODO: remove for a parallel option instead
        daxa_RWBufferPtr(uint) blob_u32s;
        voxel_malloc_address_to_u32_ptr(daxa_BufferPtr(VoxelMallocPageAllocator)(as_address(voxel_malloc_page_allocator)), blob_ptr, blob_u32s);
        uint i = palette_region_voxel_index;

        deref(advance(blob_u32s, 0)) = 0x0;
        deref(advance(blob_u32s, 1)) = 0x0;
        deref(advance(blob_u32s, 2)) = 0x0;

        uvec3 x8_i = palette_i;

        for (uint x = 0; x < 2; ++x)
            for (uint y = 0; y < 2; ++y)
                for (uint z = 0; z < 2; ++z) {
                    uvec3 local_i = x8_i * 2 + uvec3(x, y, z);
                    uint index = uniformity_lod_index(4)(local_i);
                    uint mask = uniformity_lod_mask(4)(local_i);
                    bool has_occluding = (deref(temp_voxel_chunk_ptr).uniformity.lod_x4[index] & mask) != 0;
                    uint new_index = new_uniformity_lod_index(4)(local_i);
                    uint new_mask = new_uniformity_lod_mask(4)(local_i);
                    if (has_occluding) {
                        deref(advance(blob_u32s, new_index)) |= new_mask;
                    }
                }
        for (uint x = 0; x < 4; ++x)
            for (uint y = 0; y < 4; ++y)
                for (uint z = 0; z < 4; ++z) {
                    uvec3 local_i = x8_i * 4 + uvec3(x, y, z);
                    uint index = uniformity_lod_index(2)(local_i);
                    uint mask = uniformity_lod_mask(2)(local_i);
                    bool has_occluding = (deref(temp_voxel_chunk_ptr).uniformity.lod_x2[index] & mask) != 0;
                    uint new_index = new_uniformity_lod_index(2)(local_i);
                    uint new_mask = new_uniformity_lod_mask(2)(local_i);
                    if (has_occluding) {
                        deref(advance(blob_u32s, new_index)) |= new_mask;
                    }
                }
    }
}
#undef VOXEL_WORLD

#endif
