#pragma once

#include <utilities/gpu/math.glsl>
#include <voxels/impl/voxel_malloc.glsl>
#include <voxels/gvox_model.glsl>
#include <voxels/pack_unpack.glsl>

#define UNIFORMITY_LOD_INDEX_IMPL(N)                                   \
    uint uniformity_lod_index_##N(uvec3 index_within_lod) {            \
        return index_within_lod.x + index_within_lod.y * uint(64 / N); \
    }
UNIFORMITY_LOD_INDEX_IMPL(2)
UNIFORMITY_LOD_INDEX_IMPL(4)
UNIFORMITY_LOD_INDEX_IMPL(8)
UNIFORMITY_LOD_INDEX_IMPL(16)
UNIFORMITY_LOD_INDEX_IMPL(32)
UNIFORMITY_LOD_INDEX_IMPL(64)

#define UNIFORMITY_LOD_MASK_IMPL(N)                        \
    uint uniformity_lod_mask_##N(uvec3 index_within_lod) { \
        return 1u << index_within_lod.z;                   \
    }
UNIFORMITY_LOD_MASK_IMPL(2)
UNIFORMITY_LOD_MASK_IMPL(4)
UNIFORMITY_LOD_MASK_IMPL(8)
UNIFORMITY_LOD_MASK_IMPL(16)
UNIFORMITY_LOD_MASK_IMPL(32)
UNIFORMITY_LOD_MASK_IMPL(64)

#define LINEAR_INDEX(N, within_lod_i) (within_lod_i.x + within_lod_i.y * N + within_lod_i.z * N * N)

uint new_uniformity_lod_index_2(uvec3 within_lod_i) { return (LINEAR_INDEX(4, (within_lod_i & 3)) + 0) >> 5; }
uint new_uniformity_lod_index_4(uvec3 within_lod_i) { return (LINEAR_INDEX(2, (within_lod_i & 1)) + 64) >> 5; }
uint new_uniformity_lod_index_8(uvec3 within_lod_i) { return (LINEAR_INDEX(1, (within_lod_i & 0)) + 72) >> 5; }
uint new_uniformity_lod_index_16(uvec3 within_lod_i) { return (LINEAR_INDEX(4, within_lod_i) + 0) >> 5; }
uint new_uniformity_lod_index_32(uvec3 within_lod_i) { return (LINEAR_INDEX(2, within_lod_i) + 64) >> 5; }
uint new_uniformity_lod_index_64(uvec3 within_lod_i) { return (LINEAR_INDEX(1, within_lod_i) + 72) >> 5; }

uint new_uniformity_lod_bit_pos_2(uvec3 within_lod_i) { return (LINEAR_INDEX(4, (within_lod_i & 3)) + 0) & 31; }
uint new_uniformity_lod_bit_pos_4(uvec3 within_lod_i) { return (LINEAR_INDEX(2, (within_lod_i & 1)) + 64) & 31; }
uint new_uniformity_lod_bit_pos_8(uvec3 within_lod_i) { return (LINEAR_INDEX(1, (within_lod_i & 0)) + 72) & 31; }
uint new_uniformity_lod_bit_pos_16(uvec3 within_lod_i) { return (LINEAR_INDEX(4, within_lod_i) + 0) & 31; }
uint new_uniformity_lod_bit_pos_32(uvec3 within_lod_i) { return (LINEAR_INDEX(2, within_lod_i) + 64) & 31; }
uint new_uniformity_lod_bit_pos_64(uvec3 within_lod_i) { return (LINEAR_INDEX(1, within_lod_i) + 72) & 31; }

uint new_uniformity_lod_mask_2(uvec3 within_lod_i) { return 1 << new_uniformity_lod_bit_pos_2(within_lod_i); }
uint new_uniformity_lod_mask_4(uvec3 within_lod_i) { return 1 << new_uniformity_lod_bit_pos_4(within_lod_i); }
uint new_uniformity_lod_mask_8(uvec3 within_lod_i) { return 1 << new_uniformity_lod_bit_pos_8(within_lod_i); }
uint new_uniformity_lod_mask_16(uvec3 within_lod_i) { return 1 << new_uniformity_lod_bit_pos_16(within_lod_i); }
uint new_uniformity_lod_mask_32(uvec3 within_lod_i) { return 1 << new_uniformity_lod_bit_pos_32(within_lod_i); }
uint new_uniformity_lod_mask_64(uvec3 within_lod_i) { return 1 << new_uniformity_lod_bit_pos_64(within_lod_i); }

#define uniformity_lod_index(N) uniformity_lod_index_##N
#define new_uniformity_lod_index(N) new_uniformity_lod_index_##N
#define uniformity_lod_mask(N) uniformity_lod_mask_##N
#define new_uniformity_lod_mask(N) new_uniformity_lod_mask_##N

bool VoxelUniformityChunk_lod_nonuniform_2(daxa_BufferPtr(VoxelMallocPageAllocator) allocator, PaletteHeader palette_header, uint index, uint mask) {
    if (palette_header.variant_n < 2) {
        return false;
    }

    daxa_RWBufferPtr(uint) blob_u32s;
    voxel_malloc_address_to_u32_ptr(allocator, palette_header.blob_ptr, blob_u32s);
    return (deref(advance(blob_u32s, index)) & mask) != 0;
}
bool VoxelUniformityChunk_lod_nonuniform_4(daxa_BufferPtr(VoxelMallocPageAllocator) allocator, PaletteHeader palette_header, uint index, uint mask) {
    if (palette_header.variant_n < 2) {
        return false;
    }
    daxa_RWBufferPtr(uint) blob_u32s;
    voxel_malloc_address_to_u32_ptr(allocator, palette_header.blob_ptr, blob_u32s);
    return (deref(advance(blob_u32s, index)) & mask) != 0;
}
bool VoxelUniformityChunk_lod_nonuniform_8(daxa_BufferPtr(VoxelMallocPageAllocator) allocator, PaletteHeader palette_header, uint index, uint mask) {
    return palette_header.variant_n > 1;
}

bool VoxelUniformityChunk_lod_nonuniform_16(daxa_BufferPtr(VoxelLeafChunk) voxel_chunk_ptr, uint index, uint mask) {
    return (deref(voxel_chunk_ptr).uniformity_bits[index] & mask) != 0;
}
bool VoxelUniformityChunk_lod_nonuniform_32(daxa_BufferPtr(VoxelLeafChunk) voxel_chunk_ptr, uint index, uint mask) {
    return (deref(voxel_chunk_ptr).uniformity_bits[index] & mask) != 0;
}
bool VoxelUniformityChunk_lod_nonuniform_64(daxa_BufferPtr(VoxelLeafChunk) voxel_chunk_ptr, uint index, uint mask) {
    return (deref(voxel_chunk_ptr).uniformity_bits[index] & mask) != 0;
}

#define voxel_uniformity_lod_nonuniform(N) VoxelUniformityChunk_lod_nonuniform_##N

// 3D Leaf Chunk index => uint index in buffer
uint calc_chunk_index(daxa_BufferPtr(VoxelWorldGlobals) voxel_globals, uvec3 chunk_i, uvec3 chunk_n) {
#if ENABLE_CHUNK_WRAPPING
    // Modulate the chunk index to be wrapped around relative to the chunk offset provided.
    chunk_i = uvec3((ivec3(chunk_i) + (deref(voxel_globals).offset >> ivec3(6 + LOG2_VOXEL_SIZE))) % ivec3(chunk_n));
#endif
    uint chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;
    return chunk_index;
}

uint calc_chunk_index_from_worldspace(ivec3 chunk_i, uvec3 chunk_n) {
    chunk_i = chunk_i % ivec3(chunk_n) + ivec3(chunk_i.x < 0, chunk_i.y < 0, chunk_i.z < 0) * ivec3(chunk_n);
    uint chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;
    return chunk_index;
}

uint calc_palette_region_index(uvec3 inchunk_voxel_i) {
    uvec3 palette_region_i = inchunk_voxel_i / PALETTE_REGION_SIZE;
    return palette_region_i.x + palette_region_i.y * PALETTES_PER_CHUNK_AXIS + palette_region_i.z * PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS;
}

uint calc_palette_voxel_index(uvec3 inchunk_voxel_i) {
    uvec3 palette_voxel_i = inchunk_voxel_i & (PALETTE_REGION_SIZE - 1);
    return palette_voxel_i.x + palette_voxel_i.y * PALETTE_REGION_SIZE + palette_voxel_i.z * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE;
}

#define READ_FROM_HEAP 1
// This function assumes the variant_n is greater than 1.
PackedVoxel sample_palette(daxa_BufferPtr(VoxelMallocPageAllocator) allocator, PaletteHeader palette_header, uint palette_voxel_index) {
#if READ_FROM_HEAP
    daxa_RWBufferPtr(uint) blob_u32s;
    voxel_malloc_address_to_u32_ptr(allocator, palette_header.blob_ptr, blob_u32s);
    blob_u32s = advance(blob_u32s, PALETTE_ACCELERATION_STRUCTURE_SIZE_U32S);
#endif
    if (palette_header.variant_n > PALETTE_MAX_COMPRESSED_VARIANT_N) {
#if READ_FROM_HEAP
        return PackedVoxel(deref(advance(blob_u32s, palette_voxel_index)));
#else
        return PackedVoxel(0x01ffff00);
#endif
    }
#if READ_FROM_HEAP
    uint bits_per_variant = ceil_log2(palette_header.variant_n);
    uint mask = (~0u) >> (32 - bits_per_variant);
    uint bit_index = palette_voxel_index * bits_per_variant;
    uint data_index = bit_index / 32;
    uint data_offset = bit_index - data_index * 32;
    uint my_palette_index = (deref(advance(blob_u32s, palette_header.variant_n + data_index + 0)) >> data_offset) & mask;
    if (data_offset + bits_per_variant > 32) {
        uint shift = bits_per_variant - ((data_offset + bits_per_variant) & 0x1f);
        my_palette_index |= (deref(advance(blob_u32s, palette_header.variant_n + data_index + 1)) << shift) & mask;
    }
    uint voxel_data = deref(advance(blob_u32s, my_palette_index));
    return PackedVoxel(voxel_data);
#else
    return PackedVoxel(0x01ffff00);
#endif
}

PackedVoxel sample_voxel_chunk(daxa_BufferPtr(VoxelMallocPageAllocator) allocator, daxa_BufferPtr(VoxelLeafChunk) voxel_chunk_ptr, uvec3 inchunk_voxel_i) {
    uint palette_region_index = calc_palette_region_index(inchunk_voxel_i);
    uint palette_voxel_index = calc_palette_voxel_index(inchunk_voxel_i);
    PaletteHeader palette_header = deref(voxel_chunk_ptr).palette_headers[palette_region_index];
    if (palette_header.variant_n < 2) {
        return PackedVoxel(palette_header.blob_ptr);
    }
    return sample_palette(allocator, palette_header, palette_voxel_index);
}

PackedVoxel sample_voxel_chunk(VoxelBufferPtrs ptrs, uvec3 chunk_n, vec3 voxel_p, vec3 bias) {
    vec3 offset = vec3((deref(ptrs.globals).offset) & ((1 << (6 + LOG2_VOXEL_SIZE)) - 1)) + vec3(chunk_n) * CHUNK_WORLDSPACE_SIZE * 0.5;
    uvec3 voxel_i = uvec3(floor((voxel_p + offset) * VOXEL_SCL + bias));
    uvec3 chunk_i = voxel_i / CHUNK_SIZE;
    uint chunk_index = calc_chunk_index(ptrs.globals, chunk_i, chunk_n);
    return sample_voxel_chunk(ptrs.allocator, advance(ptrs.voxel_chunks_ptr, chunk_index), voxel_i - chunk_i * CHUNK_SIZE);
}

PackedVoxel sample_temp_voxel_chunk(
    daxa_BufferPtr(VoxelWorldGlobals) voxel_globals,
    daxa_BufferPtr(VoxelMallocPageAllocator) allocator,
    daxa_BufferPtr(VoxelLeafChunk) voxel_chunks_ptr,
    daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunks,
    uvec3 chunk_n, uvec3 voxel_i) {

    uvec3 chunk_i = voxel_i / CHUNK_SIZE;
    uvec3 inchunk_voxel_i = voxel_i - chunk_i * CHUNK_SIZE;
    uint chunk_index = calc_chunk_index(voxel_globals, chunk_i, chunk_n);
    daxa_BufferPtr(VoxelLeafChunk) voxel_chunk_ptr = advance(voxel_chunks_ptr, chunk_index);
    uint update_index = deref(voxel_chunk_ptr).update_index;
    if (update_index == 0) {
        return sample_voxel_chunk(allocator, voxel_chunk_ptr, inchunk_voxel_i);
    } else {
        daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr = advance(temp_voxel_chunks, update_index - 1);
        return deref(temp_voxel_chunk_ptr).voxels[inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE];
    }
}

uint sample_lod(daxa_BufferPtr(VoxelMallocPageAllocator) allocator, daxa_BufferPtr(VoxelLeafChunk) voxel_chunk_ptr, uvec3 chunk_i, uvec3 inchunk_voxel_i, out PackedVoxel voxel_data) {
    uint lod_index_x2 = new_uniformity_lod_index(2)(inchunk_voxel_i / 2);
    uint lod_mask_x2 = new_uniformity_lod_mask(2)(inchunk_voxel_i / 2);
    uint lod_index_x4 = new_uniformity_lod_index(4)(inchunk_voxel_i / 4);
    uint lod_mask_x4 = new_uniformity_lod_mask(4)(inchunk_voxel_i / 4);
    uint lod_index_x8 = new_uniformity_lod_index(8)(inchunk_voxel_i / 8);
    uint lod_mask_x8 = new_uniformity_lod_mask(8)(inchunk_voxel_i / 8);
    uint lod_index_x16 = new_uniformity_lod_index(16)(inchunk_voxel_i / 16);
    uint lod_mask_x16 = new_uniformity_lod_mask(16)(inchunk_voxel_i / 16);
    uint lod_index_x32 = new_uniformity_lod_index(32)(inchunk_voxel_i / 32);
    uint lod_mask_x32 = new_uniformity_lod_mask(32)(inchunk_voxel_i / 32);
    uint lod_index_x64 = new_uniformity_lod_index(64)(inchunk_voxel_i / 64);
    uint lod_mask_x64 = new_uniformity_lod_mask(64)(inchunk_voxel_i / 64);
    uint chunk_flags = deref(voxel_chunk_ptr).flags;
    if ((chunk_flags & CHUNK_FLAGS_ACCEL_GENERATED) == 0)
        return 7;

    uint palette_region_index = calc_palette_region_index(inchunk_voxel_i);
    uint palette_voxel_index = calc_palette_voxel_index(inchunk_voxel_i);
    PaletteHeader palette_header = deref(voxel_chunk_ptr).palette_headers[palette_region_index];

#if (!defined(TraceDepthPrepassComputeShader)) || VOXEL_ACCEL_UNIFORMITY
    if (palette_header.variant_n < 2) {
        voxel_data = PackedVoxel(palette_header.blob_ptr);
    } else {
        voxel_data = sample_palette(allocator, palette_header, palette_voxel_index);
    }

    // if ((voxel_data & 0xff000000) != 0)
    //     return 0;
    Voxel voxel = unpack_voxel(voxel_data);
    if (voxel.material_type != 0) {
        return 0;
    }
#endif

    if (voxel_uniformity_lod_nonuniform(2)(allocator, palette_header, lod_index_x2, lod_mask_x2))
        return 1;
    if (voxel_uniformity_lod_nonuniform(4)(allocator, palette_header, lod_index_x4, lod_mask_x4))
        return 2;
    if (voxel_uniformity_lod_nonuniform(8)(allocator, palette_header, lod_index_x8, lod_mask_x8))
        return 3;
    if (voxel_uniformity_lod_nonuniform(16)(voxel_chunk_ptr, lod_index_x16, lod_mask_x16))
        return 4;
    if (voxel_uniformity_lod_nonuniform(32)(voxel_chunk_ptr, lod_index_x32, lod_mask_x32))
        return 5;
    if (voxel_uniformity_lod_nonuniform(64)(voxel_chunk_ptr, lod_index_x64, lod_mask_x64))
        return 6;

    return 7;
}

uint sample_lod(daxa_BufferPtr(VoxelWorldGlobals) voxel_globals, daxa_BufferPtr(VoxelMallocPageAllocator) allocator, daxa_BufferPtr(VoxelLeafChunk) voxel_chunks_ptr, uvec3 chunk_n, vec3 voxel_p, out PackedVoxel voxel_data) {
    uvec3 voxel_i = uvec3(voxel_p * VOXEL_SCL);
    uvec3 chunk_i = voxel_i / CHUNK_SIZE;
    uint chunk_index = calc_chunk_index(voxel_globals, chunk_i, chunk_n);
    return sample_lod(allocator, advance(voxel_chunks_ptr, chunk_index), chunk_i, voxel_i - chunk_i * CHUNK_SIZE, voxel_data);
}
