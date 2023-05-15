#extension GL_EXT_shader_atomic_int64 : require

#include <shared/shared.inl>

#include <utils/math.glsl>
#include <utils/voxel_malloc.glsl>
#include <utils/voxels.glsl>

shared u32 compression_result[PALETTE_REGION_TOTAL_SIZE];
shared u64 voted_results[PALETTE_REGION_TOTAL_SIZE];
shared u32 palette_size;
#if USE_OLD_ALLOC
shared VoxelMalloc_Pointer blob_ptr;
#endif

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

#define SETTINGS deref(settings)
#define VOXEL_WORLD deref(globals).voxel_world
layout(local_size_x = PALETTE_REGION_SIZE, local_size_y = PALETTE_REGION_SIZE, local_size_z = PALETTE_REGION_SIZE) in;
void main() {
    u32vec3 chunk_n = u32vec3(1u << SETTINGS.log2_chunks_per_axis);
    u32 temp_chunk_index = gl_GlobalInvocationID.z / CHUNK_SIZE;
    u32vec3 chunk_i = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].i;
    u32 chunk_index = calc_chunk_index(chunk_i, chunk_n);
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

#if USE_OLD_ALLOC
    if (palette_region_voxel_index == 0) {
        if (prev_variant_n > 1) {
            VoxelMalloc_free(voxel_malloc_global_allocator, voxel_chunk_ptr, prev_blob_ptr);
        }
    }

    barrier();
    memoryBarrierShared();
#endif

    u32 bits_per_variant = ceil_log2(palette_size);

    u32 compressed_size = 0;
#if USE_OLD_ALLOC
#else
    VoxelMalloc_Pointer blob_ptr = my_voxel;
#endif

    if (palette_size > PALETTE_MAX_COMPRESSED_VARIANT_N) {
        compressed_size = PALETTE_REGION_TOTAL_SIZE;
#if USE_OLD_ALLOC
        if (palette_region_voxel_index == 0) {
            blob_ptr = VoxelMalloc_malloc(voxel_malloc_global_allocator, voxel_chunk_ptr, compressed_size);
#else
        if (prev_variant_n > 1) {
            blob_ptr = prev_blob_ptr;
            VoxelMalloc_realloc(voxel_malloc_global_allocator, voxel_chunk_ptr, blob_ptr, compressed_size);
        } else {
            blob_ptr = VoxelMalloc_malloc(voxel_malloc_global_allocator, voxel_chunk_ptr, compressed_size);
        }
        if (palette_region_voxel_index == 0) {
#endif
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].variant_n = palette_size;
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].blob_ptr = blob_ptr;
        }

        compression_result[palette_region_voxel_index] = my_voxel;
    } else if (palette_size > 1) {
        compressed_size = palette_size + (bits_per_variant * PALETTE_REGION_TOTAL_SIZE + 31) / 32;
#if USE_OLD_ALLOC
        if (palette_region_voxel_index == 0) {
            blob_ptr = VoxelMalloc_malloc(voxel_malloc_global_allocator, voxel_chunk_ptr, compressed_size);
#else
        if (prev_variant_n > 1) {
            blob_ptr = prev_blob_ptr;
            VoxelMalloc_realloc(voxel_malloc_global_allocator, voxel_chunk_ptr, blob_ptr, compressed_size);
        } else {
            blob_ptr = VoxelMalloc_malloc(voxel_malloc_global_allocator, voxel_chunk_ptr, compressed_size);
        }
        if (palette_region_voxel_index == 0) {
#endif
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
                VoxelMalloc_free(voxel_malloc_global_allocator, voxel_chunk_ptr, prev_blob_ptr);
            }
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].variant_n = palette_size;
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].blob_ptr = my_voxel;
        }
    }

    barrier();
    memoryBarrierShared();

    if (palette_region_voxel_index < compressed_size) {
        daxa_RWBufferPtr(daxa_u32) blob_u32s = voxel_malloc_address_to_u32_ptr(voxel_malloc_global_allocator, blob_ptr);
        deref(blob_u32s[palette_region_voxel_index]) = compression_result[palette_region_voxel_index];
    }
}
#undef VOXEL_WORLD
#undef SETTINGS
