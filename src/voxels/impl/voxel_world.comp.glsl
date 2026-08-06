#include <voxels/voxel_world.inl>

#if PerChunkComputeShader

DAXA_DECL_PUSH_CONSTANT(PerChunkComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(VoxelWorldGlobals) voxel_globals = push.uses.voxel_globals;
daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunks = push.uses.voxel_chunks;

#include <utilities/gpu/math.glsl>
#include <voxels/voxels.glsl>

#define VOXEL_WORLD deref(voxel_globals)
#define PLAYER deref(gpu_input).player
#define CHUNKS(i) deref(advance(voxel_chunks, i))
#define INDIRECT deref(voxel_globals).indirect_dispatch

uint chunk_index;

void try_elect(in out VoxelChunkUpdateInfo work_item, in out uint update_index) {
    if (work_item.brush_flags == BRUSH_FLAGS_WORLD_BRUSH) {
        // reject certain worldgen chunks

        // (const) number of chunks in each axis
        uvec3 chunk_n = uvec3(CHUNK_NX, CHUNK_NY, CHUNK_NZ);
        // Chunk 3D index in leaf chunk space (0^3 - 31^3)
        ivec3 chunk_i = work_item.i;
        // Player chunk offset
        ivec3 chunk_offset = work_item.chunk_offset;
        // Wrapped chunk index in leaf chunk space (0^3 - 31^3)
        ivec3 wrapped_chunk_i = imod3(chunk_i - imod3(chunk_offset - ivec3(chunk_n), ivec3(chunk_n)), ivec3(chunk_n));
        // Leaf chunk position in world space
        ivec3 world_chunk = chunk_offset + wrapped_chunk_i - ivec3(chunk_n / 2);

        // [Chunk corner] Voxel position in world space (voxels)
        ivec3 world_voxel = world_chunk * CHUNK_SIZE;
        // [Chunk center] Voxel position in world space (meters)
        vec3 voxel_pos = (vec3(world_voxel) + CHUNK_SIZE / 2 + 0.5) * VOXEL_SIZE;

        // if (length(fract(voxel_pos / 64) - 0.5) * 64 - 10 - (CHUNK_SIZE * VOXEL_SIZE * 0.87) > 0) {
        //     CHUNKS(chunk_index).flags |= CHUNK_FLAGS_ACCEL_GENERATED;
        //     return;
        // }
    }

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
    ivec3 chunk_n = ivec3(CHUNK_NX, CHUNK_NY, CHUNK_NZ);

    VoxelChunkUpdateInfo terrain_work_item;
    terrain_work_item.i = ivec3(gl_GlobalInvocationID.xyz) % chunk_n;

    ivec3 offset = (VOXEL_WORLD.offset >> ivec3(6 + LOG2_VOXEL_SIZE));
    ivec3 prev_offset = (VOXEL_WORLD.prev_offset >> ivec3(6 + LOG2_VOXEL_SIZE));

    terrain_work_item.chunk_offset = offset;
    terrain_work_item.brush_flags = BRUSH_FLAGS_WORLD_BRUSH;

    // (const) number of chunks in each axis
    chunk_index = calc_chunk_index_from_worldspace(terrain_work_item.i, chunk_n);

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
        vec3 world_chunk_center = vec3(world_chunk << (6 + LOG2_VOXEL_SIZE)) + (32 * VOXEL_SIZE);

        terrain_work_item.brush_input = VOXEL_WORLD.brush_input;

        ivec3 brush_chunk = (ivec3(floor(VOXEL_WORLD.brush_input.pos)) + VOXEL_WORLD.brush_input.pos_offset) >> (6 + LOG2_VOXEL_SIZE);
        bool is_near_brush = all(greaterThanEqual(world_chunk, brush_chunk - 1)) && all(lessThanEqual(world_chunk, brush_chunk + 1));

        vec3 world_brush_pos = VOXEL_WORLD.brush_input.pos + VOXEL_WORLD.brush_input.pos_offset;
        bool is_near_brush_a = is_near_brush;
        bool is_near_brush_b = length(world_brush_pos.xy - world_chunk_center.xy) < 4 && abs(world_brush_pos.z + 6 - world_chunk_center.z) < 10;
        is_near_brush_b = is_near_brush_b || (length(world_brush_pos + vec3(0, 0, 9) - world_chunk_center) < 10);

        if (is_near_brush_a && deref(gpu_input).actions[GAME_ACTION_BRUSH_A] != 0) {
            terrain_work_item.brush_flags = BRUSH_FLAGS_USER_BRUSH_A;
            try_elect(terrain_work_item, update_index);
        } else if (is_near_brush_b && deref(gpu_input).actions[GAME_ACTION_BRUSH_B] != 0 && deref(voxel_globals).brush_state.initial_frame == deref(gpu_input).frame_index) {
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
daxa_BufferPtr(VoxelWorldGlobals) voxel_globals = push.uses.voxel_globals;
daxa_BufferPtr(VoxelLeafChunk) voxel_chunks = push.uses.voxel_chunks;
daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunks = push.uses.temp_voxel_chunks;
SIMPLE_STATIC_ALLOCATOR_BUFFERS_PUSH_USES(GrassStrandAllocator, grass_allocator)
SIMPLE_STATIC_ALLOCATOR_BUFFERS_PUSH_USES(FlowerAllocator, flower_allocator)
SIMPLE_STATIC_ALLOCATOR_BUFFERS_PUSH_USES(TreeParticleAllocator, tree_particle_allocator)
SIMPLE_STATIC_ALLOCATOR_BUFFERS_PUSH_USES(FireParticleAllocator, fire_particle_allocator)
daxa_ImageViewIndex test_texture = push.uses.test_texture;
daxa_ImageViewIndex test_texture2 = push.uses.test_texture2;

#include <utilities/gpu/math.glsl>
#include <utilities/gpu/noise.glsl>
#include <voxels/voxels.glsl>

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

#include <voxels/brushes.glsl>

#define VOXEL_WORLD deref(voxel_globals)
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    // (const) number of chunks in each axis
    chunk_n = uvec3(CHUNK_NX, CHUNK_NY, CHUNK_NZ);
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

    PackedVoxel packed_result = PackedVoxel(PACKED_NULL_VOXEL);
    Voxel result = unpack_voxel(packed_result);

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

    packed_result = pack_voxel(result);
    // result.col_and_id = vec4_to_uint_rgba8(vec4(col, 0.0)) | (id << 0x18);
    deref(temp_voxel_chunk_ptr).voxels[inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE] = packed_result;
}
#undef VOXEL_WORLD

#endif

#if ChunkEditPostProcessComputeShader

DAXA_DECL_PUSH_CONSTANT(ChunkEditPostProcessComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(VoxelWorldGlobals) voxel_globals = push.uses.voxel_globals;
daxa_BufferPtr(VoxelLeafChunk) voxel_chunks = push.uses.voxel_chunks;
daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunks = push.uses.temp_voxel_chunks;

#include <utilities/gpu/math.glsl>
#include <utilities/gpu/noise.glsl>
#include <voxels/voxels.glsl>

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
    chunk_n = uvec3(CHUNK_NX, CHUNK_NY, CHUNK_NZ);
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

    if ((packed_result.data & PACKED_NULL_VOXEL_MASK) != PACKED_NULL_VOXEL) {
        Voxel result = unpack_voxel(packed_result);
        if (result.material_type == 0) {
            result.normal = GENERATE_NORMAL;
            result.color = vec3(0.0);
            result.roughness = 0;
        } else {
            bool is_occluded = !has_air_neighbor();
            if (is_occluded) {
                // nullify normal
                result.normal = GENERATE_NORMAL;
                // result.color = vec3(0.9);
            } else {
                // potentially generate a normal
                // if the voxel normal is the "null" normal AKA up
                // bool generate_normal = true;
                bool generate_normal = dot(result.normal, unpack_voxel(pack_voxel(Voxel(0, 0, GENERATE_NORMAL, vec3(0)))).normal) > 0.99;
                if (generate_normal) {
                    result.normal = generate_normal_from_geometry();
                }
                result.normal = normalize(result.normal);
            }
        }
        packed_result = pack_voxel(result);
    }

    // result.col_and_id = vec4_to_uint_rgba8(vec4(col, 0.0)) | (id << 0x18);
    deref(temp_voxel_chunk_ptr).voxels[inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE] = packed_result;
}
#undef VOXEL_WORLD

#endif

#if ChunkAllocComputeShader

DAXA_DECL_PUSH_CONSTANT(ChunkAllocComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(VoxelWorldGlobals) voxel_globals = push.uses.voxel_globals;
daxa_BufferPtr(TempVoxelChunk) temp_voxel_chunks = push.uses.temp_voxel_chunks;
daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunks = push.uses.voxel_chunks;
daxa_RWBufferPtr(ChunkUpdate) chunk_updates = push.uses.chunk_updates;
daxa_RWBufferPtr(uint) chunk_update_heap = push.uses.chunk_update_heap;

#extension GL_EXT_shader_atomic_int64 : require

#include <utilities/gpu/math.glsl>
#include <voxels/voxels.glsl>

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
        daxa_u64 vote_result = atomicCompSwap(voted_results[algo_i], 0, daxa_u64(my_voxel) | (daxa_u64(1) << daxa_u64(32)));
        if (vote_result == 0) {
            my_palette_index = algo_i + 1;
            compression_result[algo_i] = my_voxel;
            break;
        } else if (my_voxel == uint(vote_result)) {
            my_palette_index = algo_i + 1;
            break;
        }
    }
    atomicMax(palette_size, my_palette_index);
    // Necessary so that all threads have the same `palette_size` value
    barrier();
}

shared uint output_offset;
void alloc_output(uint palette_region_voxel_index, uint compressed_size) {
    if (palette_region_voxel_index == 0 && compressed_size > 0) {
        output_offset = atomicAdd(deref(voxel_globals).chunk_update_heap_alloc_n, compressed_size);
    }

    barrier();
    memoryBarrierShared();
}

#define VOXEL_WORLD deref(voxel_globals)
layout(local_size_x = PALETTE_REGION_SIZE, local_size_y = PALETTE_REGION_SIZE, local_size_z = PALETTE_REGION_SIZE) in;
void main() {
    uvec3 chunk_n = uvec3(CHUNK_NX, CHUNK_NY, CHUNK_NZ);
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

    uint bits_per_variant = ceil_log2(palette_size);

    uint compressed_size = 0;

    if (palette_size > PALETTE_MAX_COMPRESSED_VARIANT_N) {
        compressed_size = PALETTE_REGION_TOTAL_SIZE;
        compression_result[palette_region_voxel_index] = my_voxel;
    } else if (palette_size > 1) {
        compressed_size = palette_size + (bits_per_variant * PALETTE_REGION_TOTAL_SIZE + 31) / 32;
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
    }

    barrier();
    memoryBarrierShared();

    alloc_output(palette_region_voxel_index, compressed_size);
    uint frame_index = deref(gpu_input).frame_index % (FRAMES_IN_FLIGHT + 1);

    if (palette_region_voxel_index < compressed_size) {
        deref(advance(chunk_update_heap, frame_index * MAX_CHUNK_UPDATES_PER_FRAME_VOXEL_COUNT + output_offset + palette_region_voxel_index)) = compression_result[palette_region_voxel_index];
    }
    if (palette_region_voxel_index == 0) {
        CpuChunkUpdateInfo chunk_update_info;
        chunk_update_info.chunk_index = chunk_index;
        chunk_update_info.flags = 1;
        deref(advance(chunk_updates, frame_index * MAX_CHUNK_UPDATES_PER_FRAME + temp_chunk_index)).info = chunk_update_info;
        PaletteHeader palette_header;
        palette_header.variant_n = palette_size;
        if (compressed_size == 0) {
            palette_header.blob_ptr = my_voxel;
        } else {
            palette_header.blob_ptr = output_offset;
        }
        deref(advance(chunk_updates, frame_index * MAX_CHUNK_UPDATES_PER_FRAME + temp_chunk_index)).palette_headers[palette_region_index] = palette_header;
    }

    // Finish the chunk
    if (gl_LocalInvocationIndex == 0) {
        deref(voxel_chunk_ptr).flags = CHUNK_FLAGS_ACCEL_GENERATED;
    }
}
#undef VOXEL_WORLD

#endif
