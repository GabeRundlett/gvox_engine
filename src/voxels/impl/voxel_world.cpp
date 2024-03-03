#include <application/settings.inl>
#include "voxel_world.inl"

#include <fmt/format.h>

#if CPU_VOXEL_GEN

using glm::ivec3;
using glm::uvec3;
using glm::uvec2;
using glm::vec3;
using glm::vec2;

void VoxelWorld::startup() {
    voxel_globals.chunk_update_n = 0;
}
void VoxelWorld::per_frame() {
    for (uint32_t i = 0; i < MAX_CHUNK_UPDATES_PER_FRAME; ++i) {
        voxel_globals.chunk_update_infos[i].brush_flags = 0;
        voxel_globals.chunk_update_infos[i].i = std::bit_cast<daxa_i32vec3>(INVALID_CHUNK_I);
    }

    voxel_globals.prev_offset = voxel_globals.offset;
    voxel_globals.offset = {}; // deref(gpu_input).player.player_unit_offset;

    voxel_globals.indirect_dispatch.chunk_edit_dispatch = daxa_u32vec3(CHUNK_SIZE / 8, CHUNK_SIZE / 8, 0);
    voxel_globals.indirect_dispatch.subchunk_x2x4_dispatch = daxa_u32vec3(1, 64, 0);
    voxel_globals.indirect_dispatch.subchunk_x8up_dispatch = daxa_u32vec3(1, 1, 0);

    voxel_globals.chunk_update_n = 0;
    buffers.voxel_malloc.per_frame();

    // TODO: Brush stuff...
}

uint32_t calc_chunk_index_from_worldspace(ivec3 chunk_i, uvec3 chunk_n) {
    chunk_i = chunk_i % ivec3(chunk_n) + ivec3(chunk_i.x < 0, chunk_i.y < 0, chunk_i.z < 0) * ivec3(chunk_n);
    uint32_t chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;
    return chunk_index;
}

template <typename T>
T atomicAdd(T &x, T val) {
    T prev = x;
    x += val;
    return prev;
}

int imod(int x, int m) {
    return x >= 0 ? x % m : m - 1 - (-x - 1) % m;
}
ivec3 imod3(ivec3 p, int m) {
    return ivec3(imod(p.x, m), imod(p.y, m), imod(p.z, m));
}
ivec3 imod3(ivec3 p, ivec3 m) {
    return ivec3(imod(p.x, m.x), imod(p.y, m.y), imod(p.z, m.z));
}
ivec3 floor(ivec3 p, ivec3 m) {
    return ivec3(imod(p.x, m.x), imod(p.y, m.y), imod(p.z, m.z));
}

uint32_t pack_unit(float x, uint32_t bit_n) {
    float scl = float(1u << bit_n) - 1.0;
    return uint32_t(round(x * scl));
}
float unpack_unit(uint32_t x, uint32_t bit_n) {
    float scl = float(1u << bit_n) - 1.0;
    return float(x) / scl;
}

vec3 unpack_rgb(uint32_t u) {
    vec3 result;
    result.r = float((u >> 0) & 0x3f) / 63.0;
    result.g = float((u >> 6) & 0x3f) / 63.0;
    result.b = float((u >> 12) & 0x3f) / 63.0;
    result = pow(result, vec3(2.2));
    // result = hsv2rgb(result);
    return result;
}
uint32_t pack_rgb(vec3 f) {
    // f = rgb2hsv(f);
    f = pow(f, vec3(1.0 / 2.2));
    uint32_t result = 0;
    result |= uint32_t(std::clamp(f.r * 63.0f, 0.0f, 63.0f)) << 0;
    result |= uint32_t(std::clamp(f.g * 63.0f, 0.0f, 63.0f)) << 6;
    result |= uint32_t(std::clamp(f.b * 63.0f, 0.0f, 63.0f)) << 12;
    return result;
}

float msign(float v) {
    return (v >= 0.0f) ? 1.0f : -1.0f;
}

uint32_t octahedral_8(vec3 nor) {
    auto fac0 = (abs(nor.x) + abs(nor.y) + abs(nor.z));
    nor.x /= fac0;
    nor.y /= fac0;
    auto fac1x = (nor.z >= 0.0f) ? nor.x : (1.0f - abs(nor.y)) * msign(nor.x);
    auto fac1y = (nor.z >= 0.0f) ? nor.y : (1.0f - abs(nor.x)) * msign(nor.y);
    nor.x = fac1x;
    nor.y = fac1y;
    uvec2 d = uvec2(round(7.5f + nor.x * 7.5f), round(7.5f + nor.y * 7.5f));
    return d.x | (d.y << 4u);
}
vec3 i_octahedral_8(uint32_t data) {
    uvec2 iv = uvec2(data, data >> 4u) & 15u;
    vec2 v = vec2(iv) / 7.5f - 1.0f;
    vec3 nor = vec3(v, 1.0f - abs(v.x) - abs(v.y)); // Rune Stubbe's version,
    float t = std::max(-nor.z, 0.0f);               // much faster than original
    nor.x += (nor.x > 0.0) ? -t : t;                // implementation of this
    nor.y += (nor.y > 0.0) ? -t : t;                // technique
    return normalize(nor);
}

PackedVoxel pack_voxel(Voxel v) {
    PackedVoxel result;

#if DITHER_NORMALS
    rand_seed(good_rand_hash(floatBitsToUint(v.normal)));
    const mat3 basis = build_orthonormal_basis(normalize(v.normal));
    v.normal = basis * uniform_sample_cone(vec2(rand(), rand()), cos(0.19 * 0.5));
#endif

    uint32_t packed_roughness = pack_unit(sqrt(v.roughness), 4);
    uint32_t packed_normal = octahedral_8(glm::normalize(std::bit_cast<vec3>(v.normal)));
    uint32_t packed_color = pack_rgb(std::bit_cast<vec3>(v.color));

    result.data = (v.material_type) | (packed_roughness << 2) | (packed_normal << 6) | (packed_color << 14);

    return result;
}
Voxel unpack_voxel(PackedVoxel v) {
    Voxel result = Voxel(0, 0, daxa_f32vec3(0), daxa_f32vec3(0));

    result.material_type = (v.data >> 0) & 3;

    uint32_t packed_roughness = (v.data >> 2) & 15;
    uint32_t packed_normal = (v.data >> 6) & ((1 << 8) - 1);
    uint32_t packed_color = (v.data >> 14);

    result.roughness = pow(unpack_unit(packed_roughness, 4), 2.0f);
    result.normal = std::bit_cast<daxa_f32vec3>(i_octahedral_8(packed_normal));
    result.color = std::bit_cast<daxa_f32vec3>(unpack_rgb(packed_color));

    return result;
}

// 3D Leaf Chunk index => uint index in buffer
uint32_t calc_chunk_index(VoxelWorldGlobals const *voxel_globals, uvec3 chunk_i, uvec3 chunk_n) {
#if ENABLE_CHUNK_WRAPPING
    // Modulate the chunk index to be wrapped around relative to the chunk offset provided.
    chunk_i = uvec3((ivec3(chunk_i) + (std::bit_cast<ivec3>(voxel_globals->offset) >> ivec3(6 + LOG2_VOXEL_SIZE))) % ivec3(chunk_n));
#endif
    uint32_t chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;
    return chunk_index;
}

uint32_t calc_temp_voxel_index(uvec3 inchunk_voxel_i) {
    uvec3 in_chunk_4x_node = inchunk_voxel_i / uvec3(4u);
    uvec3 in_4x_node_voxel = inchunk_voxel_i & uvec3(3u);
    auto in_chunk_4x_node_index = in_chunk_4x_node.x + in_chunk_4x_node.y * (CHUNK_SIZE / 4) + in_chunk_4x_node.z * (CHUNK_SIZE / 4) * (CHUNK_SIZE / 4);
    auto in_4x_node_voxel_index = in_4x_node_voxel.x + in_4x_node_voxel.y * 4 + in_4x_node_voxel.z * 4 * 4;
    return in_chunk_4x_node_index * (4 * 4 * 4) + in_4x_node_voxel_index;

    // return inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE;
}

void try_elect(VoxelWorldGlobals &VOXEL_WORLD, VoxelChunkUpdateInfo &work_item, uint32_t &update_index) {
    uint32_t prev_update_n = atomicAdd(VOXEL_WORLD.chunk_update_n, 1u);

    // Check if the work item can be added
    if (prev_update_n < MAX_CHUNK_UPDATES_PER_FRAME) {
        // Set the chunk edit dispatch z axis (64/8, 64/8, 64 x 8 x 8 / 8 = 64 x 8) = (8, 8, 512)
        atomicAdd(VOXEL_WORLD.indirect_dispatch.chunk_edit_dispatch.z, CHUNK_SIZE / 8u);
        atomicAdd(VOXEL_WORLD.indirect_dispatch.subchunk_x2x4_dispatch.z, 1u);
        atomicAdd(VOXEL_WORLD.indirect_dispatch.subchunk_x8up_dispatch.z, 1u);
        // Set the chunk update info
        VOXEL_WORLD.chunk_update_infos[prev_update_n] = work_item;
        update_index = prev_update_n + 1;
    }
}

void VoxelWorld::per_chunk(uvec3 gl_GlobalInvocationID) {
    ivec3 chunk_n = ivec3(1 << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);

    VoxelChunkUpdateInfo terrain_work_item;
    terrain_work_item.i = std::bit_cast<daxa_i32vec3>(ivec3(gl_GlobalInvocationID) & (chunk_n - 1));

    ivec3 offset = (std::bit_cast<ivec3>(voxel_globals.offset) >> ivec3(6 + LOG2_VOXEL_SIZE));
    ivec3 prev_offset = (std::bit_cast<ivec3>(voxel_globals.prev_offset) >> ivec3(6 + LOG2_VOXEL_SIZE));

    terrain_work_item.chunk_offset = std::bit_cast<daxa_i32vec3>(offset);
    terrain_work_item.brush_flags = BRUSH_FLAGS_WORLD_BRUSH;

    // (const) number of chunks in each axis
    uint32_t chunk_index = calc_chunk_index_from_worldspace(std::bit_cast<ivec3>(terrain_work_item.i), chunk_n);

    uint32_t update_index = 0;

    if ((voxel_chunks[chunk_index].flags & CHUNK_FLAGS_ACCEL_GENERATED) == 0) {
        try_elect(voxel_globals, terrain_work_item, update_index);
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

        uvec3 temp_chunk_i = uvec3((std::bit_cast<ivec3>(terrain_work_item.i) - offset) % ivec3(chunk_n));

        if ((temp_chunk_i.x >= start.x && temp_chunk_i.x < end.x) ||
            (temp_chunk_i.y >= start.y && temp_chunk_i.y < end.y) ||
            (temp_chunk_i.z >= start.z && temp_chunk_i.z < end.z)) {
            voxel_chunks[chunk_index].flags &= ~CHUNK_FLAGS_ACCEL_GENERATED;
            try_elect(voxel_globals, terrain_work_item, update_index);
        }
    } else {
        // Wrapped chunk index in leaf chunk space (0^3 - 31^3)
        ivec3 wrapped_chunk_i = imod3(std::bit_cast<ivec3>(terrain_work_item.i) - imod3(std::bit_cast<ivec3>(terrain_work_item.chunk_offset) - ivec3(chunk_n), ivec3(chunk_n)), ivec3(chunk_n));
        // Leaf chunk position in world space
        ivec3 world_chunk = std::bit_cast<ivec3>(terrain_work_item.chunk_offset) + wrapped_chunk_i - ivec3(chunk_n / 2);

        terrain_work_item.brush_input = voxel_globals.brush_input;

        ivec3 brush_chunk = (ivec3(floor(std::bit_cast<vec3>(voxel_globals.brush_input.pos))) + std::bit_cast<ivec3>(voxel_globals.brush_input.pos_offset)) >> (6 + LOG2_VOXEL_SIZE);
        bool is_near_brush = all(greaterThanEqual(world_chunk, brush_chunk - 1)) && all(lessThanEqual(world_chunk, brush_chunk + 1));

        // if (is_near_brush && deref(gpu_input).actions[GAME_ACTION_BRUSH_A] != 0) {
        //     terrain_work_item.brush_flags = BRUSH_FLAGS_USER_BRUSH_A;
        //     try_elect(voxel_globals, terrain_work_item, update_index);
        // } else if (is_near_brush && deref(gpu_input).actions[GAME_ACTION_BRUSH_B] != 0) {
        //     terrain_work_item.brush_flags = BRUSH_FLAGS_USER_BRUSH_B;
        //     try_elect(voxel_globals, terrain_work_item, update_index);
        // }
    }

    voxel_chunks[chunk_index].update_index = update_index;
}

const uvec3 chunk_n = uvec3(1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);

void VoxelWorld::edit(uvec3 gl_GlobalInvocationID) {
    uint32_t temp_chunk_index;
    ivec3 chunk_i;
    uint32_t chunk_index;
    uvec3 inchunk_voxel_i;
    ivec3 voxel_i;
    ivec3 world_voxel;
    vec3 voxel_pos;
    BrushInput brush_input;

    // (const) number of chunks in each axis
    // Index in chunk_update_infos buffer
    temp_chunk_index = gl_GlobalInvocationID.z / CHUNK_SIZE;
    // Chunk 3D index in leaf chunk space (0^3 - 31^3)
    chunk_i = std::bit_cast<ivec3>(voxel_globals.chunk_update_infos[temp_chunk_index].i);

    // Here we check whether the chunk update that we're handling is an update
    // for a chunk that has already been submitted. This is a bit inefficient,
    // since we'd hopefully like to queue a separate work item into the queue
    // instead, but this is tricky.
    if (chunk_i == INVALID_CHUNK_I) {
        return;
    }

    // Player chunk offset
    ivec3 chunk_offset = std::bit_cast<ivec3>(voxel_globals.chunk_update_infos[temp_chunk_index].chunk_offset);
    // Brush informations
    brush_input = voxel_globals.chunk_update_infos[temp_chunk_index].brush_input;
    // Brush flags
    uint32_t brush_flags = voxel_globals.chunk_update_infos[temp_chunk_index].brush_flags;
    // Chunk uint32_t index in voxel_chunks buffer
    chunk_index = calc_chunk_index_from_worldspace(chunk_i, chunk_n);
    // Pointer to the new chunk
    // voxel_chunk_ptr = advance(voxel_chunks, chunk_index);
    // Voxel offset in chunk
    inchunk_voxel_i = gl_GlobalInvocationID - uvec3(0, 0, temp_chunk_index * CHUNK_SIZE);
    // Voxel 3D position (in voxel buffer)
    voxel_i = chunk_i * CHUNK_SIZE + ivec3(inchunk_voxel_i);

    // Wrapped chunk index in leaf chunk space (0^3 - 31^3)
    ivec3 wrapped_chunk_i = imod3(chunk_i - imod3(chunk_offset - ivec3(chunk_n), ivec3(chunk_n)), ivec3(chunk_n));
    // Leaf chunk position in world space
    ivec3 world_chunk = chunk_offset + wrapped_chunk_i - ivec3(chunk_n / 2u);

    // Voxel position in world space (voxels)
    world_voxel = world_chunk * CHUNK_SIZE + ivec3(inchunk_voxel_i);
    // Voxel position in world space (meters)
    voxel_pos = vec3(world_voxel) * VOXEL_SIZE;

    // rand_seed(voxel_i.x + voxel_i.y * 1000 + voxel_i.z * 1000 * 1000);

    Voxel result = Voxel(0, 0, daxa_f32vec3(0, 0, 1), daxa_f32vec3(0, 0, 0));

    if ((brush_flags & BRUSH_FLAGS_WORLD_BRUSH) != 0) {
        // brushgen_world(result);
        result.material_type = 1;
    }
    // if ((brush_flags & BRUSH_FLAGS_USER_BRUSH_A) != 0) {
    //     brushgen_a(result);
    // }
    // if ((brush_flags & BRUSH_FLAGS_USER_BRUSH_B) != 0) {
    //     brushgen_b(result);
    // }
    // if ((brush_flags & BRUSH_FLAGS_PARTICLE_BRUSH) != 0) {
    //     brushgen_particles(col, id);
    // }

    PackedVoxel packed_result = pack_voxel(result);
    // result.col_and_id = vec4_to_uint_rgba8(vec4(col, 0.0)) | (id << 0x18);
    auto temp_voxel_index = calc_temp_voxel_index(inchunk_voxel_i);
    temp_voxel_chunks[temp_chunk_index].voxels[temp_voxel_index] = packed_result;
}

uint32_t calc_palette_region_index(uvec3 inchunk_voxel_i) {
    uvec3 palette_region_i = inchunk_voxel_i / uvec3(PALETTE_REGION_SIZE);
    return palette_region_i.x + palette_region_i.y * PALETTES_PER_CHUNK_AXIS + palette_region_i.z * PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS;
}

uint32_t calc_palette_voxel_index(uvec3 inchunk_voxel_i) {
    uvec3 palette_voxel_i = inchunk_voxel_i & uvec3(PALETTE_REGION_SIZE - 1);
    return palette_voxel_i.x + palette_voxel_i.y * PALETTE_REGION_SIZE + palette_voxel_i.z * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE;
}

// See 'VoxelMalloc_Pointer' in shared/voxel_malloc.inl
uint32_t VoxelMalloc_Pointer_extract_local_page_alloc_offset(VoxelMalloc_Pointer ptr) {
    return (ptr >> 0) & 0x1f;
}
uint32_t VoxelMalloc_Pointer_extract_global_page_index(VoxelMalloc_Pointer ptr) {
    return (ptr >> 5);
}

void voxel_malloc_address_to_base_u32_ptr(uint32_t *heap, VoxelMalloc_Pointer address, uint32_t **result) {
    uint32_t *page = heap + VoxelMalloc_Pointer_extract_global_page_index(address) * VOXEL_MALLOC_PAGE_SIZE_U32S;
    *result = page + (VoxelMalloc_Pointer_extract_local_page_alloc_offset(address) * VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT);
}

void voxel_malloc_address_to_u32_ptr(uint32_t *heap, VoxelMalloc_Pointer address, uint32_t **result) {
    uint32_t *page = heap + VoxelMalloc_Pointer_extract_global_page_index(address) * VOXEL_MALLOC_PAGE_SIZE_U32S;
    *result = page + (VoxelMalloc_Pointer_extract_local_page_alloc_offset(address) * VOXEL_MALLOC_U32S_PER_PAGE_BITFIELD_BIT + 1);
}

#include <renderer/kajiya/blur.inl>

#define deref(x) (*(x))
#define advance(x, offset) ((x) + (offset))

#define READ_FROM_HEAP 1
// This function assumes the variant_n is greater than 1.
PackedVoxel sample_palette(uint32_t *heap, PaletteHeader palette_header, uint32_t palette_voxel_index) {
#if READ_FROM_HEAP
    uint32_t *blob_u32s;
    voxel_malloc_address_to_u32_ptr(heap, palette_header.blob_ptr, &blob_u32s);
    blob_u32s = blob_u32s + PALETTE_ACCELERATION_STRUCTURE_SIZE_U32S;
#endif
    if (palette_header.variant_n > PALETTE_MAX_COMPRESSED_VARIANT_N) {
#if READ_FROM_HEAP
        return PackedVoxel(*(blob_u32s + palette_voxel_index));
#else
        return PackedVoxel(0x01ffff00);
#endif
    }
#if READ_FROM_HEAP
    uint32_t bits_per_variant = ceil_log2(palette_header.variant_n);
    uint32_t mask = (~0u) >> (32 - bits_per_variant);
    uint32_t bit_index = palette_voxel_index * bits_per_variant;
    uint32_t data_index = bit_index / 32;
    uint32_t data_offset = bit_index - data_index * 32;
    uint32_t my_palette_index = (deref(advance(blob_u32s, palette_header.variant_n + data_index + 0)) >> data_offset) & mask;
    if (data_offset + bits_per_variant > 32) {
        uint32_t shift = bits_per_variant - ((data_offset + bits_per_variant) & 0x1f);
        my_palette_index |= (deref(advance(blob_u32s, palette_header.variant_n + data_index + 1)) << shift) & mask;
    }
    uint32_t voxel_data = deref(advance(blob_u32s, my_palette_index));
    return PackedVoxel(voxel_data);
#else
    return PackedVoxel(0x01ffff00);
#endif
}

PackedVoxel sample_voxel_chunk(uint32_t *heap, VoxelLeafChunk const *voxel_chunk_ptr, uvec3 inchunk_voxel_i) {
    uint32_t palette_region_index = calc_palette_region_index(inchunk_voxel_i);
    uint32_t palette_voxel_index = calc_palette_voxel_index(inchunk_voxel_i);
    PaletteHeader palette_header = voxel_chunk_ptr->palette_headers[palette_region_index];
    if (palette_header.variant_n < 2) {
        return PackedVoxel(palette_header.blob_ptr);
    }
    return sample_palette(heap, palette_header, palette_voxel_index);
}

PackedVoxel sample_temp_voxel_chunk(
    VoxelWorldGlobals const *voxel_globals,
    uint32_t *heap,
    VoxelLeafChunk const *voxel_chunks_ptr,
    TempVoxelChunk *temp_voxel_chunks,
    uvec3 chunk_n, uvec3 voxel_i) {

    uvec3 chunk_i = voxel_i / uvec3(CHUNK_SIZE);
    uvec3 inchunk_voxel_i = voxel_i - chunk_i * uvec3(CHUNK_SIZE);
    uint32_t chunk_index = calc_chunk_index(voxel_globals, chunk_i, chunk_n);
    VoxelLeafChunk const *voxel_chunk_ptr = advance(voxel_chunks_ptr, chunk_index);
    uint32_t update_index = voxel_chunk_ptr->update_index;
    if (update_index == 0) {
        return sample_voxel_chunk(heap, voxel_chunk_ptr, inchunk_voxel_i);
    } else {
        TempVoxelChunk *temp_voxel_chunk_ptr = advance(temp_voxel_chunks, update_index - 1);
        auto temp_voxel_index = calc_temp_voxel_index(inchunk_voxel_i);
        return temp_voxel_chunk_ptr->voxels[temp_voxel_index];
    }
}

Voxel VoxelWorld::get_temp_voxel(ivec3 world_voxel, ivec3 offset_i) {
    // TODO: Simplify this, and improve precision
    vec3 i = vec3(world_voxel + offset_i) * VOXEL_SIZE; // - deref(gpu_input).player.player_unit_offset;
    vec3 offset = vec3(std::bit_cast<ivec3>(voxel_globals.offset) & ((1 << (6 + LOG2_VOXEL_SIZE)) - 1)) + vec3(chunk_n) * CHUNK_WORLDSPACE_SIZE * 0.5f;
    uvec3 voxel_i = uvec3(floor((i + offset) * float(VOXEL_SCL)));
    Voxel default_value = Voxel(0, 0, daxa_f32vec3(0), daxa_f32vec3(0));
    if (any(greaterThanEqual(voxel_i, uvec3(chunk_n * uint32_t(CHUNK_SIZE))))) {
        return default_value;
    }
    return unpack_voxel(sample_temp_voxel_chunk(
        &voxel_globals,
        buffers.voxel_malloc.elements.data(),
        voxel_chunks.data(),
        temp_voxel_chunks.data(),
        chunk_n, voxel_i));
}

bool VoxelWorld::has_air_neighbor(ivec3 world_voxel) {
    bool result = false;

    {
        Voxel v = get_temp_voxel(world_voxel, ivec3(-1, 0, 0));
        if (v.material_type == 0) {
            result = true;
        }
    }
    {
        Voxel v = get_temp_voxel(world_voxel, ivec3(+1, 0, 0));
        if (v.material_type == 0) {
            result = true;
        }
    }
    {
        Voxel v = get_temp_voxel(world_voxel, ivec3(0, -1, 0));
        if (v.material_type == 0) {
            result = true;
        }
    }
    {
        Voxel v = get_temp_voxel(world_voxel, ivec3(0, +1, 0));
        if (v.material_type == 0) {
            result = true;
        }
    }
    {
        Voxel v = get_temp_voxel(world_voxel, ivec3(0, 0, -1));
        if (v.material_type == 0) {
            result = true;
        }
    }
    {
        Voxel v = get_temp_voxel(world_voxel, ivec3(0, 0, +1));
        if (v.material_type == 0) {
            result = true;
        }
    }

    return result;
}

vec3 VoxelWorld::generate_normal_from_geometry(ivec3 world_voxel) {
    vec3 density_n = vec3(0);
    vec3 density_p = vec3(0);
    const int RADIUS = 2;
    for (int zi = -RADIUS; zi <= RADIUS; ++zi) {
        for (int yi = -RADIUS; yi <= RADIUS; ++yi) {
            for (int xi = -RADIUS; xi <= RADIUS; ++xi) {
                Voxel v = get_temp_voxel(world_voxel, ivec3(xi, yi, zi));
                if (v.material_type == 0) {
                    vec3 dir = vec3(xi, yi, zi);
                    density_n.x += std::max(0.0f, dot(dir, vec3(-1, 0, 0)));
                    density_p.x += std::max(0.0f, dot(dir, vec3(+1, 0, 0)));
                    density_n.y += std::max(0.0f, dot(dir, vec3(0, -1, 0)));
                    density_p.y += std::max(0.0f, dot(dir, vec3(0, +1, 0)));
                    density_n.z += std::max(0.0f, dot(dir, vec3(0, 0, -1)));
                    density_p.z += std::max(0.0f, dot(dir, vec3(0, 0, +1)));
                }
            }
        }
    }

    vec3 d = density_p - density_n;
    if (dot(d, d) < 0.1f) {
        // Hack to fix flat sides. TODO: Generalize
        vec3 v = density_p + density_n;
        float min_v = std::min(v.x, std::min(v.y, v.z));
        float max_v = std::max(v.x, std::max(v.y, v.z));
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

void VoxelWorld::edit_post_process(uvec3 gl_GlobalInvocationID) {
    uint32_t temp_chunk_index;
    ivec3 chunk_i;
    uint32_t chunk_index;
    uvec3 inchunk_voxel_i;
    ivec3 voxel_i;
    ivec3 world_voxel;
    vec3 voxel_pos;
    BrushInput brush_input;

    // (const) number of chunks in each axis
    // Index in chunk_update_infos buffer
    temp_chunk_index = gl_GlobalInvocationID.z / CHUNK_SIZE;
    // Chunk 3D index in leaf chunk space (0^3 - 31^3)
    chunk_i = std::bit_cast<ivec3>(voxel_globals.chunk_update_infos[temp_chunk_index].i);

    // Here we check whether the chunk update that we're handling is an update
    // for a chunk that has already been submitted. This is a bit inefficient,
    // since we'd hopefully like to queue a separate work item into the queue
    // instead, but this is tricky.
    if (chunk_i == INVALID_CHUNK_I) {
        return;
    }

    // Player chunk offset
    ivec3 chunk_offset = std::bit_cast<ivec3>(voxel_globals.chunk_update_infos[temp_chunk_index].chunk_offset);
    // Brush informations
    brush_input = voxel_globals.chunk_update_infos[temp_chunk_index].brush_input;
    // Brush flags
    uint32_t brush_flags = voxel_globals.chunk_update_infos[temp_chunk_index].brush_flags;
    // Chunk uint32_t index in voxel_chunks buffer
    chunk_index = calc_chunk_index_from_worldspace(chunk_i, chunk_n);
    // Pointer to the new chunk
    // voxel_chunk_ptr = advance(voxel_chunks, chunk_index);
    // Voxel offset in chunk
    inchunk_voxel_i = gl_GlobalInvocationID - uvec3(0, 0, temp_chunk_index * CHUNK_SIZE);
    // Voxel 3D position (in voxel buffer)
    voxel_i = chunk_i * CHUNK_SIZE + ivec3(inchunk_voxel_i);

    // Wrapped chunk index in leaf chunk space (0^3 - 31^3)
    ivec3 wrapped_chunk_i = imod3(chunk_i - imod3(chunk_offset - ivec3(chunk_n), ivec3(chunk_n)), ivec3(chunk_n));
    // Leaf chunk position in world space
    ivec3 world_chunk = chunk_offset + wrapped_chunk_i - ivec3(chunk_n / 2u);

    // Voxel position in world space (voxels)
    world_voxel = world_chunk * CHUNK_SIZE + ivec3(inchunk_voxel_i);
    // Voxel position in world space (meters)
    voxel_pos = vec3(world_voxel) * VOXEL_SIZE;

    // rand_seed(voxel_i.x + voxel_i.y * 1000 + voxel_i.z * 1000 * 1000);
    auto temp_voxel_index = calc_temp_voxel_index(inchunk_voxel_i);

    PackedVoxel packed_result = temp_voxel_chunks[temp_chunk_index].voxels[temp_voxel_index];
    Voxel result = unpack_voxel(packed_result);

    if (result.material_type == 0) {
        result.normal = daxa_f32vec3(0, 0, 1);
        result.color = daxa_f32vec3(0.0f);
        result.roughness = 0;
    } else {
        bool is_occluded = !has_air_neighbor(world_voxel);
        if (is_occluded) {
            // nullify normal
            result.normal = daxa_f32vec3(0, 0, 1);
            // result.color = vec3(0.9);
        } else {
            // potentially generate a normal
            // if the voxel normal is the "null" normal AKA up
            // bool generate_normal = true;
            auto invalid_normal = unpack_voxel(pack_voxel(Voxel(0, 0, daxa_f32vec3(0, 0, 1), daxa_f32vec3(0)))).normal;
            bool generate_normal = (std::bit_cast<vec3>(result.normal) == std::bit_cast<vec3>(invalid_normal));
            if (generate_normal) {
                result.normal = std::bit_cast<daxa_f32vec3>(generate_normal_from_geometry(world_voxel));
            }
            result.normal = std::bit_cast<daxa_f32vec3>(glm::normalize(std::bit_cast<vec3>(result.normal)));
        }
    }

    packed_result = pack_voxel(result);
    // result.col_and_id = vec4_to_uint_rgba8(vec4(col, 0.0)) | (id << 0x18);
    temp_voxel_chunks[temp_chunk_index].voxels[temp_voxel_index] = packed_result;
}
void VoxelWorld::opt_x2x4(uvec3 gl_GlobalInvocationID) {
}
void VoxelWorld::opt_x8up(uvec3 gl_WorkGroupID, uvec3 gl_GlobalInvocationID) {
    ivec3 chunk_i = std::bit_cast<ivec3>(voxel_globals.chunk_update_infos[gl_WorkGroupID.z].i);

    uvec3 chunk_n;
    chunk_n.x = 1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS;
    chunk_n.y = chunk_n.x;
    chunk_n.z = chunk_n.x;
    uint32_t chunk_index = calc_chunk_index_from_worldspace(chunk_i, chunk_n);

    voxel_chunks[chunk_index].flags = CHUNK_FLAGS_ACCEL_GENERATED;
}
void VoxelWorld::alloc(uvec3 gl_GlobalInvocationID) {
}
#endif

void VoxelWorld::record_startup(RecordContext &record_ctx) {
#if CPU_VOXEL_GEN
    startup();
#else
    use_buffers(record_ctx);

    record_ctx.task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_globals.task_resource),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_chunks.task_resource),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_malloc.task_element_buffer),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            ti.recorder.clear_buffer({
                .buffer = buffers.voxel_globals.task_resource.get_state().buffers[0],
                .offset = 0,
                .size = sizeof(VoxelWorldGlobals),
                .clear_value = 0,
            });

            auto chunk_n = (1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);
            chunk_n = chunk_n * chunk_n * chunk_n * CHUNK_LOD_LEVELS;
            ti.recorder.clear_buffer({
                .buffer = buffers.voxel_chunks.task_resource.get_state().buffers[0],
                .offset = 0,
                .size = sizeof(VoxelLeafChunk) * chunk_n,
                .clear_value = 0,
            });

            buffers.voxel_malloc.clear_buffers(ti.recorder);
        },
        .name = "clear chunk editor",
    });

    record_ctx.task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_malloc.task_allocator_buffer),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            buffers.voxel_malloc.init(ti.device, ti.recorder);
        },
        .name = "Initialize",
    });

    record_ctx.add(ComputeTask<VoxelWorldStartupCompute, VoxelWorldStartupComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/startup.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{VoxelWorldStartupCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
            VOXELS_BUFFER_USES_ASSIGN(VoxelWorldStartupCompute, buffers),
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, VoxelWorldStartupComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({1, 1, 1});
        },
    });
#endif
}

void VoxelWorld::begin_frame(daxa::Device &device, VoxelWorldOutput const &gpu_output) {
#if CPU_VOXEL_GEN
    per_frame();
    for (uint32_t zi = 0; zi < (1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS); ++zi) {
        for (uint32_t yi = 0; yi < (1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS); ++yi) {
            for (uint32_t xi = 0; xi < (1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS); ++xi) {
                per_chunk(uvec3(xi, yi, zi));
            }
        }
    }
    for (uint32_t zi = 0; zi < voxel_globals.indirect_dispatch.chunk_edit_dispatch.z; ++zi) {
        for (uint32_t yi = 0; yi < voxel_globals.indirect_dispatch.chunk_edit_dispatch.y; ++yi) {
            for (uint32_t xi = 0; xi < voxel_globals.indirect_dispatch.chunk_edit_dispatch.x; ++xi) {
                // shared mem
                for (uint32_t tzi = 0; tzi < 8; ++tzi) {
                    for (uint32_t tyi = 0; tyi < 8; ++tyi) {
                        for (uint32_t txi = 0; txi < 8; ++txi) {
                            edit(uvec3(xi * 8 + txi, yi * 8 + tyi, zi * 8 + tzi));
                        }
                    }
                }
            }
        }
    }
    // for (uint32_t zi = 0; zi < voxel_globals.indirect_dispatch.chunk_edit_dispatch.z; ++zi) {
    //     for (uint32_t yi = 0; yi < voxel_globals.indirect_dispatch.chunk_edit_dispatch.y; ++yi) {
    //         for (uint32_t xi = 0; xi < voxel_globals.indirect_dispatch.chunk_edit_dispatch.x; ++xi) {
    //             // shared mem
    //             for (uint32_t tzi = 0; tzi < 8; ++tzi) {
    //                 for (uint32_t tyi = 0; tyi < 8; ++tyi) {
    //                     for (uint32_t txi = 0; txi < 8; ++txi) {
    //                         edit_post_process(uvec3(xi * 8 + txi, yi * 8 + tyi, zi * 8 + tzi));
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    for (uint32_t zi = 0; zi < voxel_globals.indirect_dispatch.subchunk_x2x4_dispatch.z; ++zi) {
        for (uint32_t yi = 0; yi < voxel_globals.indirect_dispatch.subchunk_x2x4_dispatch.y; ++yi) {
            for (uint32_t xi = 0; xi < voxel_globals.indirect_dispatch.subchunk_x2x4_dispatch.x; ++xi) {
                // shared mem
                for (uint32_t ti = 0; ti < 512; ++ti) {
                    opt_x2x4(uvec3(xi * 512 + ti, yi, zi));
                }
            }
        }
    }
    for (uint32_t zi = 0; zi < voxel_globals.indirect_dispatch.subchunk_x2x4_dispatch.z; ++zi) {
        for (uint32_t yi = 0; yi < voxel_globals.indirect_dispatch.subchunk_x2x4_dispatch.y; ++yi) {
            for (uint32_t xi = 0; xi < voxel_globals.indirect_dispatch.subchunk_x2x4_dispatch.x; ++xi) {
                // shared mem
                for (uint32_t ti = 0; ti < 512; ++ti) {
                    opt_x8up(uvec3(xi, yi, zi), uvec3(xi * 512 + ti, yi, zi));
                }
            }
        }
    }
    for (uint32_t zi = 0; zi < voxel_globals.indirect_dispatch.chunk_edit_dispatch.z; ++zi) {
        for (uint32_t yi = 0; yi < voxel_globals.indirect_dispatch.chunk_edit_dispatch.y; ++yi) {
            for (uint32_t xi = 0; xi < voxel_globals.indirect_dispatch.chunk_edit_dispatch.x; ++xi) {
                // shared mem
                for (uint32_t tzi = 0; tzi < 8; ++tzi) {
                    for (uint32_t tyi = 0; tyi < 8; ++tyi) {
                        for (uint32_t txi = 0; txi < 8; ++txi) {
                            alloc(uvec3(xi * 8 + txi, yi * 8 + tyi, zi * 8 + tzi));
                        }
                    }
                }
            }
        }
    }
#else
    buffers.voxel_malloc.check_for_realloc(device, gpu_output.voxel_malloc_output.current_element_count);

    bool needs_realloc = false;
    needs_realloc = needs_realloc || buffers.voxel_malloc.needs_realloc();

    debug_gpu_heap_usage = gpu_output.voxel_malloc_output.current_element_count * VOXEL_MALLOC_PAGE_SIZE_BYTES;
    debug_page_count = buffers.voxel_malloc.current_element_count;
    debug_utils::DebugDisplay::set_debug_string("VoxelWorld: Page count", fmt::format("{} pages ({:.2f} MB)", debug_page_count, static_cast<double>(debug_page_count) * VOXEL_MALLOC_PAGE_SIZE_BYTES / 1'000'000.0));
    debug_utils::DebugDisplay::set_debug_string("VoxelWorld: GPU heap usage", fmt::format("{:.2f} MB", static_cast<double>(debug_gpu_heap_usage) / 1'000'000));

    if (needs_realloc) {
        auto temp_task_graph = daxa::TaskGraph({
            .device = device,
            .name = "temp_task_graph",
        });

        buffers.voxel_malloc.for_each_task_buffer([&temp_task_graph](auto &task_buffer) { temp_task_graph.use_persistent_buffer(task_buffer); });
        temp_task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_READ, buffers.voxel_malloc.task_old_element_buffer),
                daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_malloc.task_element_buffer),
            },
            .task = [this](daxa::TaskInterface const &ti) {
                if (buffers.voxel_malloc.needs_realloc()) {
                    buffers.voxel_malloc.realloc(ti.device, ti.recorder);
                }
            },
            .name = "Transfer Task",
        });

        temp_task_graph.submit({});
        temp_task_graph.complete({});
        temp_task_graph.execute({});
    }
#endif
}

void VoxelWorld::use_buffers(RecordContext &record_ctx) {
    buffers.voxel_globals = record_ctx.gpu_context->find_or_add_temporal_buffer({
        .size = sizeof(VoxelWorldGlobals),
        .name = "voxel_globals",
    });

    auto chunk_n = (1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);
    chunk_n = chunk_n * chunk_n * chunk_n * CHUNK_LOD_LEVELS;
    buffers.voxel_chunks = record_ctx.gpu_context->find_or_add_temporal_buffer({
        .size = sizeof(VoxelLeafChunk) * chunk_n,
        .name = "voxel_chunks",
    });

    if (!gpu_malloc_initialized) {
        gpu_malloc_initialized = true;
        buffers.voxel_malloc.create(*record_ctx.gpu_context);
    }

    record_ctx.task_graph.use_persistent_buffer(buffers.voxel_globals.task_resource);
    record_ctx.task_graph.use_persistent_buffer(buffers.voxel_chunks.task_resource);
    buffers.voxel_malloc.for_each_task_buffer([&record_ctx](auto &task_buffer) { record_ctx.task_graph.use_persistent_buffer(task_buffer); });

#if CPU_VOXEL_GEN
    temp_voxel_chunks.resize(MAX_CHUNK_UPDATES_PER_FRAME);
    voxel_chunks.resize(chunk_n);
#endif
}

void VoxelWorld::record_frame(RecordContext &record_ctx, daxa::TaskBufferView task_gvox_model_buffer, daxa::TaskImageView task_value_noise_image) {
    use_buffers(record_ctx);

#if CPU_VOXEL_GEN
    // upload data
#else
    record_ctx.add(ComputeTask<VoxelWorldPerframeCompute, VoxelWorldPerframeComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/perframe.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{VoxelWorldPerframeCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
            daxa::TaskViewVariant{std::pair{VoxelWorldPerframeCompute::gpu_output, record_ctx.gpu_context->task_output_buffer}},
            VOXELS_BUFFER_USES_ASSIGN(VoxelWorldPerframeCompute, buffers),
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, VoxelWorldPerframeComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({1, 1, 1});
        },
    });

    record_ctx.add(ComputeTask<PerChunkCompute, PerChunkComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{PerChunkCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
            daxa::TaskViewVariant{std::pair{PerChunkCompute::gvox_model, task_gvox_model_buffer}},
            daxa::TaskViewVariant{std::pair{PerChunkCompute::voxel_globals, buffers.voxel_globals.task_resource}},
            daxa::TaskViewVariant{std::pair{PerChunkCompute::voxel_chunks, buffers.voxel_chunks.task_resource}},
            daxa::TaskViewVariant{std::pair{PerChunkCompute::value_noise_texture, task_value_noise_image.view({.layer_count = 256})}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, PerChunkComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            auto const dispatch_size = 1 << LOG2_CHUNKS_DISPATCH_SIZE;
            ti.recorder.dispatch({dispatch_size, dispatch_size, dispatch_size * CHUNK_LOD_LEVELS});
        },
    });

    auto task_temp_voxel_chunks_buffer = record_ctx.task_graph.create_transient_buffer({
        .size = sizeof(TempVoxelChunk) * MAX_CHUNK_UPDATES_PER_FRAME,
        .name = "temp_voxel_chunks_buffer",
    });

    record_ctx.add(ComputeTask<ChunkEditCompute, ChunkEditComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::gvox_model, task_gvox_model_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::voxel_globals, buffers.voxel_globals.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::voxel_chunks, buffers.voxel_chunks.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::voxel_malloc_page_allocator, buffers.voxel_malloc.task_allocator_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::temp_voxel_chunks, task_temp_voxel_chunks_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::value_noise_texture, task_value_noise_image.view({.layer_count = 256})}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::test_texture, record_ctx.gpu_context->task_test_texture}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::test_texture2, record_ctx.gpu_context->task_test_texture2}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ChunkEditComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.get(ChunkEditCompute::voxel_globals).ids[0],
                .offset = offsetof(VoxelWorldGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, chunk_edit_dispatch),
            });
        },
    });

    record_ctx.add(ComputeTask<ChunkEditPostProcessCompute, ChunkEditPostProcessComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::gvox_model, task_gvox_model_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::voxel_globals, buffers.voxel_globals.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::voxel_chunks, buffers.voxel_chunks.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::voxel_malloc_page_allocator, buffers.voxel_malloc.task_allocator_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::temp_voxel_chunks, task_temp_voxel_chunks_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::value_noise_texture, task_value_noise_image.view({.layer_count = 256})}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ChunkEditPostProcessComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.get(ChunkEditPostProcessCompute::voxel_globals).ids[0],
                .offset = offsetof(VoxelWorldGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, chunk_edit_dispatch),
            });
        },
    });

    record_ctx.add(ComputeTask<ChunkOptCompute, ChunkOptComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
        .extra_defines = {{"CHUNK_OPT_STAGE", "0"}},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{ChunkOptCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkOptCompute::voxel_globals, buffers.voxel_globals.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkOptCompute::temp_voxel_chunks, task_temp_voxel_chunks_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkOptCompute::voxel_chunks, buffers.voxel_chunks.task_resource}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ChunkOptComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.get(ChunkOptCompute::voxel_globals).ids[0],
                .offset = offsetof(VoxelWorldGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, subchunk_x2x4_dispatch),
            });
        },
    });

    record_ctx.add(ComputeTask<ChunkOptCompute, ChunkOptComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
        .extra_defines = {{"CHUNK_OPT_STAGE", "1"}},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{ChunkOptCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkOptCompute::voxel_globals, buffers.voxel_globals.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkOptCompute::temp_voxel_chunks, task_temp_voxel_chunks_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkOptCompute::voxel_chunks, buffers.voxel_chunks.task_resource}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ChunkOptComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.get(ChunkOptCompute::voxel_globals).ids[0],
                .offset = offsetof(VoxelWorldGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, subchunk_x8up_dispatch),
            });
        },
    });

    record_ctx.add(ComputeTask<ChunkAllocCompute, ChunkAllocComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{ChunkAllocCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkAllocCompute::voxel_globals, buffers.voxel_globals.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkAllocCompute::temp_voxel_chunks, task_temp_voxel_chunks_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkAllocCompute::voxel_chunks, buffers.voxel_chunks.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkAllocCompute::voxel_malloc_page_allocator, buffers.voxel_malloc.task_allocator_buffer}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ChunkAllocComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.get(ChunkAllocCompute::voxel_globals).ids[0],
                // NOTE: This should always have the same value as the chunk edit dispatch, so we're re-using it here
                .offset = offsetof(VoxelWorldGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, chunk_edit_dispatch),
            });
        },
    });
#endif
}
