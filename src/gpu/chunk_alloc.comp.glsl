#extension GL_EXT_shader_atomic_int64 : require

#include <shared/shared.inl>

#include <utils/math.glsl>
#include <utils/allocator.glsl>
#include <utils/voxels.glsl>

DAXA_USE_PUSH_CONSTANT(ChunkAllocComputePush)

#define WARP_SIZE 32
#define SUBGROUP_N (PALETTE_REGION_TOTAL_SIZE / WARP_SIZE)

shared u32 palette_result[PALETTE_REGION_TOTAL_SIZE];
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
                palette_result[palette_size] = my_voxel;
                palette_size++;
            } else if (my_voxel == u32(vote_result)) {
                my_palette_index = algo_i + 1;
            }
        }
        barrier();
        if (voted_results[algo_i] == 0) {
            break;
        }
    }
}

#define SETTINGS deref(daxa_push_constant.gpu_settings)
#define VOXEL_WORLD deref(daxa_push_constant.gpu_globals).voxel_world
layout(local_size_x = PALETTE_REGION_SIZE, local_size_y = PALETTE_REGION_SIZE, local_size_z = PALETTE_REGION_SIZE) in;
void main() {
    u32vec3 chunk_n;
    chunk_n.x = 1u << SETTINGS.log2_chunks_per_axis;
    chunk_n.y = chunk_n.x;
    chunk_n.z = chunk_n.x;
    u32 temp_chunk_index = gl_GlobalInvocationID.z / CHUNK_SIZE;
    u32vec3 chunk_i = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].i;
    u32 chunk_index = calc_chunk_index(chunk_i, chunk_n);
    u32vec3 inchunk_voxel_i = gl_GlobalInvocationID.xyz - u32vec3(0, 0, temp_chunk_index * CHUNK_SIZE);
    u32 inchunk_voxel_index = inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE;
    u32 palette_region_voxel_index =
        gl_LocalInvocationID.x +
        gl_LocalInvocationID.y * PALETTE_REGION_SIZE +
        gl_LocalInvocationID.z * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE;
    u32 palette_region_index =
        gl_WorkGroupID.x +
        gl_WorkGroupID.y * PALETTES_PER_CHUNK_AXIS +
        (gl_WorkGroupID.z - temp_chunk_index * PALETTES_PER_CHUNK_AXIS) * PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS;

    daxa_BufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr = daxa_push_constant.temp_voxel_chunks + temp_chunk_index;
    daxa_RWBufferPtr(VoxelChunk) voxel_chunk_ptr = daxa_push_constant.voxel_chunks + chunk_index;

    u32 my_voxel = deref(temp_voxel_chunk_ptr).voxels[inchunk_voxel_index].col_and_id;
    u32 my_palette_index = 0;

    process_palette_region(palette_region_voxel_index, my_voxel, my_palette_index);

    barrier();

    if (palette_size == 1) {
        if (palette_region_voxel_index == 0) {
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].variant_n = 1;
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].blob_offset = my_voxel;
        }
    } else {
        if (palette_size > PALETTE_MAX_COMPRESSED_VARIANT_N) {
            if (palette_region_voxel_index == 0) {
                deref(voxel_chunk_ptr).palette_headers[palette_region_index].variant_n = palette_size;
                deref(voxel_chunk_ptr).palette_headers[palette_region_index].blob_offset = gpu_malloc(daxa_push_constant.gpu_allocator, PALETTE_REGION_TOTAL_SIZE);
            }
            barrier();
            u32 final_blob_offset = deref(voxel_chunk_ptr).palette_headers[palette_region_index].blob_offset;
            deref(daxa_push_constant.gpu_allocator.heap[final_blob_offset + palette_region_voxel_index]) = my_voxel;
        } else {
            u32 bits_per_variant = ceil_log2(palette_size);
            if (palette_region_voxel_index == 0) {
                // round up to nearest byte
                u32 compressed_size = (bits_per_variant * PALETTE_REGION_TOTAL_SIZE + 7) / 8;
                // round up to the nearest uint32_t, and add an extra
                compressed_size = (compressed_size + 3) / 4 + 1;
                // add the size of the palette data
                compressed_size += palette_size;

                deref(voxel_chunk_ptr).palette_headers[palette_region_index].variant_n = palette_size;
                u32 blob_offset = gpu_malloc(daxa_push_constant.gpu_allocator, compressed_size);
                deref(voxel_chunk_ptr).palette_headers[palette_region_index].blob_offset = blob_offset;

                for (u32 i = 0; i < palette_size; ++i) {
                    deref(daxa_push_constant.gpu_allocator.heap[blob_offset + i]) = palette_result[i];
                }
            }

            barrier();

            u32 mask = (~0u) >> (32 - bits_per_variant);
            u32 bit_index = palette_region_voxel_index * bits_per_variant;
            u32 data_index = bit_index / 32;
            u32 data_offset = bit_index - data_index * 32;
            u32 data = (my_palette_index - 1) & mask;
            u32 blob_offset = deref(voxel_chunk_ptr).palette_headers[palette_region_index].blob_offset;
            u32 address = blob_offset + palette_size + data_index;
            // clang-format off
            atomicAnd(deref(daxa_push_constant.gpu_allocator.heap[address + 0]), ~(mask << data_offset));
            atomicOr (deref(daxa_push_constant.gpu_allocator.heap[address + 0]),   data << data_offset);
            if (data_offset + bits_per_variant > 32) {
                u32 shift = bits_per_variant - ((data_offset + bits_per_variant) & 0x1f);
                atomicAnd(deref(daxa_push_constant.gpu_allocator.heap[address + 1]), ~(mask >> shift));
                atomicOr (deref(daxa_push_constant.gpu_allocator.heap[address + 1]),   data >> shift);
            }
            // clang-format on
        }
    }
}
#undef VOXEL_WORLD
#undef SETTINGS
