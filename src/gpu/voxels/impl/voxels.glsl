#pragma once

#include <shared/app.inl>

#include <utils/math.glsl>
#include <voxels/impl/voxel_malloc.glsl>
#include <voxels/gvox_model.glsl>

#define INVALID_CHUNK_I daxa_i32vec3(0x80000000)
#define CHUNK_WORLDSPACE_SIZE (CHUNK_SIZE / voxel_scl)

#define UNIFORMITY_LOD_INDEX_IMPL(N)                                  \
    daxa_u32 uniformity_lod_index_##N(daxa_u32vec3 index_within_lod) {          \
        return index_within_lod.x + index_within_lod.y * daxa_u32(64 / N); \
    }
UNIFORMITY_LOD_INDEX_IMPL(2)
UNIFORMITY_LOD_INDEX_IMPL(4)
UNIFORMITY_LOD_INDEX_IMPL(8)
UNIFORMITY_LOD_INDEX_IMPL(16)
UNIFORMITY_LOD_INDEX_IMPL(32)
UNIFORMITY_LOD_INDEX_IMPL(64)

#define UNIFORMITY_LOD_MASK_IMPL(N)                         \
    daxa_u32 uniformity_lod_mask_##N(daxa_u32vec3 index_within_lod) { \
        return 1u << index_within_lod.z;                    \
    }
UNIFORMITY_LOD_MASK_IMPL(2)
UNIFORMITY_LOD_MASK_IMPL(4)
UNIFORMITY_LOD_MASK_IMPL(8)
UNIFORMITY_LOD_MASK_IMPL(16)
UNIFORMITY_LOD_MASK_IMPL(32)
UNIFORMITY_LOD_MASK_IMPL(64)

#define LINEAR_INDEX(N, within_lod_i) (within_lod_i.x + within_lod_i.y * N + within_lod_i.z * N * N)

daxa_u32 new_uniformity_lod_index_2 (daxa_u32vec3 within_lod_i) { return (LINEAR_INDEX(4, (within_lod_i & 3)) + 0) >> 5; }
daxa_u32 new_uniformity_lod_index_4 (daxa_u32vec3 within_lod_i) { return (LINEAR_INDEX(2, (within_lod_i & 1)) + 64) >> 5; }
daxa_u32 new_uniformity_lod_index_8 (daxa_u32vec3 within_lod_i) { return (LINEAR_INDEX(1, (within_lod_i & 0)) + 72) >> 5; }
daxa_u32 new_uniformity_lod_index_16(daxa_u32vec3 within_lod_i) { return (LINEAR_INDEX(4, within_lod_i) + 0) >> 5; }
daxa_u32 new_uniformity_lod_index_32(daxa_u32vec3 within_lod_i) { return (LINEAR_INDEX(2, within_lod_i) + 64) >> 5; }
daxa_u32 new_uniformity_lod_index_64(daxa_u32vec3 within_lod_i) { return (LINEAR_INDEX(1, within_lod_i) + 72) >> 5; }

daxa_u32 new_uniformity_lod_bit_pos_2 (daxa_u32vec3 within_lod_i) { return (LINEAR_INDEX(4, (within_lod_i & 3)) + 0) & 31; }
daxa_u32 new_uniformity_lod_bit_pos_4 (daxa_u32vec3 within_lod_i) { return (LINEAR_INDEX(2, (within_lod_i & 1)) + 64) & 31; }
daxa_u32 new_uniformity_lod_bit_pos_8 (daxa_u32vec3 within_lod_i) { return (LINEAR_INDEX(1, (within_lod_i & 0)) + 72) & 31; }
daxa_u32 new_uniformity_lod_bit_pos_16(daxa_u32vec3 within_lod_i) { return (LINEAR_INDEX(4, within_lod_i) + 0) & 31; }
daxa_u32 new_uniformity_lod_bit_pos_32(daxa_u32vec3 within_lod_i) { return (LINEAR_INDEX(2, within_lod_i) + 64) & 31; }
daxa_u32 new_uniformity_lod_bit_pos_64(daxa_u32vec3 within_lod_i) { return (LINEAR_INDEX(1, within_lod_i) + 72) & 31; }

daxa_u32 new_uniformity_lod_mask_2 (daxa_u32vec3 within_lod_i) { return 1 << new_uniformity_lod_bit_pos_2(within_lod_i); }
daxa_u32 new_uniformity_lod_mask_4 (daxa_u32vec3 within_lod_i) { return 1 << new_uniformity_lod_bit_pos_4(within_lod_i); }
daxa_u32 new_uniformity_lod_mask_8 (daxa_u32vec3 within_lod_i) { return 1 << new_uniformity_lod_bit_pos_8(within_lod_i); }
daxa_u32 new_uniformity_lod_mask_16(daxa_u32vec3 within_lod_i) { return 1 << new_uniformity_lod_bit_pos_16(within_lod_i); }
daxa_u32 new_uniformity_lod_mask_32(daxa_u32vec3 within_lod_i) { return 1 << new_uniformity_lod_bit_pos_32(within_lod_i); }
daxa_u32 new_uniformity_lod_mask_64(daxa_u32vec3 within_lod_i) { return 1 << new_uniformity_lod_bit_pos_64(within_lod_i); }

#define uniformity_lod_index(N) uniformity_lod_index_##N
#define new_uniformity_lod_index(N) new_uniformity_lod_index_##N
#define uniformity_lod_mask(N) uniformity_lod_mask_##N
#define new_uniformity_lod_mask(N) new_uniformity_lod_mask_##N

daxa_b32 VoxelUniformityChunk_lod_nonuniform_2(daxa_BufferPtr(VoxelMallocPageAllocator) allocator, PaletteHeader palette_header, daxa_u32 index, daxa_u32 mask) {
    if (palette_header.variant_n < 2) {
        return false;
    }

    daxa_RWBufferPtr(daxa_u32) blob_u32s;
    voxel_malloc_address_to_u32_ptr(allocator, palette_header.blob_ptr, blob_u32s);
    return (deref(blob_u32s[index]) & mask) != 0;
}
daxa_b32 VoxelUniformityChunk_lod_nonuniform_4(daxa_BufferPtr(VoxelMallocPageAllocator) allocator, PaletteHeader palette_header, daxa_u32 index, daxa_u32 mask) {
    if (palette_header.variant_n < 2) {
        return false;
    }
    daxa_RWBufferPtr(daxa_u32) blob_u32s;
    voxel_malloc_address_to_u32_ptr(allocator, palette_header.blob_ptr, blob_u32s);
    return (deref(blob_u32s[index]) & mask) != 0;
}
daxa_b32 VoxelUniformityChunk_lod_nonuniform_8(daxa_BufferPtr(VoxelMallocPageAllocator) allocator, PaletteHeader palette_header, daxa_u32 index, daxa_u32 mask) {
    return palette_header.variant_n > 1;
}

daxa_b32 VoxelUniformityChunk_lod_nonuniform_16(daxa_BufferPtr(VoxelLeafChunk) voxel_chunk_ptr, daxa_u32 index, daxa_u32 mask) {
    return (deref(voxel_chunk_ptr).uniformity_bits[index] & mask) != 0;
}
daxa_b32 VoxelUniformityChunk_lod_nonuniform_32(daxa_BufferPtr(VoxelLeafChunk) voxel_chunk_ptr, daxa_u32 index, daxa_u32 mask) {
    return (deref(voxel_chunk_ptr).uniformity_bits[index] & mask) != 0;
}
daxa_b32 VoxelUniformityChunk_lod_nonuniform_64(daxa_BufferPtr(VoxelLeafChunk) voxel_chunk_ptr, daxa_u32 index, daxa_u32 mask) {
    return (deref(voxel_chunk_ptr).uniformity_bits[index] & mask) != 0;
}

#define voxel_uniformity_lod_nonuniform(N) VoxelUniformityChunk_lod_nonuniform_##N

// 3D Leaf Chunk index => daxa_u32 index in buffer
daxa_u32 calc_chunk_index(daxa_BufferPtr(VoxelWorldGlobals) voxel_globals, daxa_u32vec3 chunk_i, daxa_u32vec3 chunk_n, daxa_u32 lod_index) {
#if ENABLE_CHUNK_WRAPPING
    // Modulate the chunk index to be wrapped around relative to the chunk offset provided.
    chunk_i = daxa_u32vec3((daxa_i32vec3(chunk_i) + (deref(voxel_globals).offset >> daxa_i32vec3(3 + lod_index))) % daxa_i32vec3(chunk_n));
#endif
    daxa_u32 chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;
    return chunk_index;
}

daxa_u32 calc_chunk_index_from_worldspace(daxa_i32vec3 chunk_i, daxa_u32vec3 chunk_n) {
    chunk_i = chunk_i % daxa_i32vec3(chunk_n) + daxa_i32vec3(chunk_i.x < 0, chunk_i.y < 0, chunk_i.z < 0) * daxa_i32vec3(chunk_n);
    daxa_u32 chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;
    return chunk_index;
}

daxa_u32 calc_palette_region_index(daxa_u32vec3 inchunk_voxel_i) {
    daxa_u32vec3 palette_region_i = inchunk_voxel_i / PALETTE_REGION_SIZE;
    return palette_region_i.x + palette_region_i.y * PALETTES_PER_CHUNK_AXIS + palette_region_i.z * PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS;
}

daxa_u32 calc_palette_voxel_index(daxa_u32vec3 inchunk_voxel_i) {
    daxa_u32vec3 palette_voxel_i = inchunk_voxel_i & (PALETTE_REGION_SIZE - 1);
    return palette_voxel_i.x + palette_voxel_i.y * PALETTE_REGION_SIZE + palette_voxel_i.z * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE;
}

#define READ_FROM_HEAP 1
// This function assumes the variant_n is greater than 1.
daxa_u32 sample_palette(daxa_BufferPtr(VoxelMallocPageAllocator) allocator, PaletteHeader palette_header, daxa_u32 palette_voxel_index) {
#if READ_FROM_HEAP
    daxa_RWBufferPtr(daxa_u32) blob_u32s;
    voxel_malloc_address_to_u32_ptr(allocator, palette_header.blob_ptr, blob_u32s);
    blob_u32s = blob_u32s + PALETTE_ACCELERATION_STRUCTURE_SIZE_U32S;
#endif
    if (palette_header.variant_n > PALETTE_MAX_COMPRESSED_VARIANT_N) {
#if READ_FROM_HEAP
        return deref(blob_u32s[palette_voxel_index]);
#else
        return 0x01ffff00;
#endif
    }
#if READ_FROM_HEAP
    daxa_u32 bits_per_variant = ceil_log2(palette_header.variant_n);
    daxa_u32 mask = (~0u) >> (32 - bits_per_variant);
    daxa_u32 bit_index = palette_voxel_index * bits_per_variant;
    daxa_u32 data_index = bit_index / 32;
    daxa_u32 data_offset = bit_index - data_index * 32;
    daxa_u32 my_palette_index = (deref(blob_u32s[palette_header.variant_n + data_index + 0]) >> data_offset) & mask;
    if (data_offset + bits_per_variant > 32) {
        daxa_u32 shift = bits_per_variant - ((data_offset + bits_per_variant) & 0x1f);
        my_palette_index |= (deref(blob_u32s[palette_header.variant_n + data_index + 1]) << shift) & mask;
    }
    daxa_u32 voxel_data = deref(blob_u32s[my_palette_index]);
    return voxel_data;
#else
    return 0x01ff00ff;
#endif
}

daxa_u32 sample_voxel_chunk(daxa_BufferPtr(VoxelMallocPageAllocator) allocator, daxa_BufferPtr(VoxelLeafChunk) voxel_chunk_ptr, daxa_u32vec3 inchunk_voxel_i) {
    daxa_u32 palette_region_index = calc_palette_region_index(inchunk_voxel_i);
    daxa_u32 palette_voxel_index = calc_palette_voxel_index(inchunk_voxel_i);
    PaletteHeader palette_header = deref(voxel_chunk_ptr).palette_headers[palette_region_index];
    if (palette_header.variant_n < 2) {
        return palette_header.blob_ptr;
    }
    return sample_palette(allocator, palette_header, palette_voxel_index);
}

daxa_u32 sample_voxel_chunk(VoxelBufferPtrs ptrs, daxa_u32vec3 chunk_n, daxa_f32vec3 voxel_p, daxa_u32 lod_index, daxa_f32vec3 bias) {
    daxa_f32 voxel_scl = daxa_f32(VOXEL_SCL) / daxa_f32(1 << lod_index);
    daxa_f32vec3 offset = daxa_f32vec3((deref(ptrs.globals).offset) & ((1 << (lod_index + 3)) - 1)) + daxa_f32vec3(chunk_n) * CHUNK_WORLDSPACE_SIZE * 0.5;
    daxa_u32vec3 voxel_i = daxa_u32vec3(floor((voxel_p + offset) * voxel_scl + bias));
    daxa_u32vec3 chunk_i = voxel_i / CHUNK_SIZE;
    daxa_u32 chunk_index = calc_chunk_index(ptrs.globals, chunk_i, chunk_n, lod_index);
    return sample_voxel_chunk(ptrs.allocator, ptrs.voxel_chunks_ptr[chunk_index], voxel_i - chunk_i * CHUNK_SIZE);
}

daxa_u32 sample_lod(daxa_BufferPtr(VoxelMallocPageAllocator) allocator, daxa_BufferPtr(VoxelLeafChunk) voxel_chunk_ptr, daxa_u32vec3 chunk_i, daxa_u32vec3 inchunk_voxel_i, out daxa_u32 voxel_data) {
    daxa_u32 lod_index_x2 = new_uniformity_lod_index(2)(inchunk_voxel_i / 2);
    daxa_u32 lod_mask_x2 = new_uniformity_lod_mask(2)(inchunk_voxel_i / 2);
    daxa_u32 lod_index_x4 = new_uniformity_lod_index(4)(inchunk_voxel_i / 4);
    daxa_u32 lod_mask_x4 = new_uniformity_lod_mask(4)(inchunk_voxel_i / 4);
    daxa_u32 lod_index_x8 = new_uniformity_lod_index(8)(inchunk_voxel_i / 8);
    daxa_u32 lod_mask_x8 = new_uniformity_lod_mask(8)(inchunk_voxel_i / 8);
    daxa_u32 lod_index_x16 = new_uniformity_lod_index(16)(inchunk_voxel_i / 16);
    daxa_u32 lod_mask_x16 = new_uniformity_lod_mask(16)(inchunk_voxel_i / 16);
    daxa_u32 lod_index_x32 = new_uniformity_lod_index(32)(inchunk_voxel_i / 32);
    daxa_u32 lod_mask_x32 = new_uniformity_lod_mask(32)(inchunk_voxel_i / 32);
    daxa_u32 lod_index_x64 = new_uniformity_lod_index(64)(inchunk_voxel_i / 64);
    daxa_u32 lod_mask_x64 = new_uniformity_lod_mask(64)(inchunk_voxel_i / 64);
    daxa_u32 chunk_flags = deref(voxel_chunk_ptr).flags;
    if ((chunk_flags & CHUNK_FLAGS_ACCEL_GENERATED) == 0)
        return 7;

#if !defined(TRACE_DEPTH_PREPASS_COMPUTE) || VOXEL_ACCEL_UNIFORMITY
    daxa_u32 palette_region_index = calc_palette_region_index(inchunk_voxel_i);
    daxa_u32 palette_voxel_index = calc_palette_voxel_index(inchunk_voxel_i);
    PaletteHeader palette_header = deref(voxel_chunk_ptr).palette_headers[palette_region_index];

    if (palette_header.variant_n < 2) {
        voxel_data = palette_header.blob_ptr;
    } else {
        voxel_data = sample_palette(allocator, palette_header, palette_voxel_index);
    }

    if ((voxel_data & 0xff000000) != 0)
        return 0;
#endif
#if TRACE_SECONDARY_COMPUTE
    // I have found, at least on memory bound GPUs (all GPUs), that never sampling
    // the X2 uniformity in the accel structure actually results in about 20% better
    // perf for the secondary trace, due to the fact that the secondary rays are
    // very divergent. This improves cache coherency, despite increasing the number
    // of total steps required to reach the intersection.
    if (voxel_uniformity_lod_nonuniform(4)(allocator, palette_header, lod_index_x4, lod_mask_x4))
        return 1;
    if (voxel_uniformity_lod_nonuniform(8)(allocator, palette_header, lod_index_x8, lod_mask_x8))
        return 3;
#elif RTDGI_TRACE_COMPUTE
    if (voxel_uniformity_lod_nonuniform(8)(allocator, palette_header, lod_index_x8, lod_mask_x8))
        return 1;
#else
    if (voxel_uniformity_lod_nonuniform(2)(allocator, palette_header, lod_index_x2, lod_mask_x2))
        return 1;
    if (voxel_uniformity_lod_nonuniform(4)(allocator, palette_header, lod_index_x4, lod_mask_x4))
        return 2;
    if (voxel_uniformity_lod_nonuniform(8)(allocator, palette_header, lod_index_x8, lod_mask_x8))
        return 3;
#endif
    if (voxel_uniformity_lod_nonuniform(16)(voxel_chunk_ptr, lod_index_x16, lod_mask_x16))
        return 4;
    if (voxel_uniformity_lod_nonuniform(32)(voxel_chunk_ptr, lod_index_x32, lod_mask_x32))
        return 5;
    if (voxel_uniformity_lod_nonuniform(64)(voxel_chunk_ptr, lod_index_x64, lod_mask_x64))
        return 6;

    return 7;
}

daxa_u32 sample_lod(daxa_BufferPtr(VoxelWorldGlobals) voxel_globals, daxa_BufferPtr(VoxelMallocPageAllocator) allocator, daxa_BufferPtr(VoxelLeafChunk) voxel_chunks_ptr, daxa_u32vec3 chunk_n, daxa_f32vec3 voxel_p, daxa_u32 lod_index, out daxa_u32 voxel_data) {
    daxa_f32 voxel_scl = daxa_f32(VOXEL_SCL) / daxa_f32(1 << lod_index);
    daxa_u32vec3 voxel_i = daxa_u32vec3(voxel_p * voxel_scl);
    daxa_u32vec3 chunk_i = voxel_i / CHUNK_SIZE;
    daxa_u32 chunk_index = calc_chunk_index(voxel_globals, chunk_i, chunk_n, lod_index);
    return sample_lod(allocator, voxel_chunks_ptr[chunk_index], chunk_i, voxel_i - chunk_i * CHUNK_SIZE, voxel_data);
}
