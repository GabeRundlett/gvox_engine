#include <shared/shared.inl>

#include <utils/math.glsl>
#include <utils/allocator.glsl>
#include <utils/voxels.glsl>

DAXA_USE_PUSH_CONSTANT(ChunkAllocComputePush)

#define SETTINGS deref(daxa_push_constant.gpu_settings)
#define VOXEL_WORLD deref(daxa_push_constant.gpu_globals).voxel_world
layout(local_size_x = PALETTE_REGION_SIZE, local_size_y = PALETTE_REGION_SIZE, local_size_z = PALETTE_REGION_SIZE) in;
void main() {
    u32vec3 chunk_n;
    chunk_n.x = 1u << SETTINGS.log2_chunks_per_axis;
    chunk_n.y = chunk_n.x;
    chunk_n.z = chunk_n.x;
    u32 temp_chunk_index = gl_GlobalInvocationID.z / CHUNK_SIZE;
    u32vec3 chunk_i = VOXEL_WORLD.chunk_update_is[temp_chunk_index];
    u32 chunk_index = calc_chunk_index(chunk_i, chunk_n);
    u32vec3 inchunk_voxel_i = gl_GlobalInvocationID.xyz - u32vec3(0, 0, temp_chunk_index * CHUNK_SIZE);
    u32 inchunk_voxel_index = inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE;

    daxa_BufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr = daxa_push_constant.temp_voxel_chunks + temp_chunk_index;
    daxa_RWBufferPtr(VoxelChunk) voxel_chunk_ptr = daxa_push_constant.voxel_chunks + chunk_index;

    u32 palette_region_voxel_index =
        gl_LocalInvocationID.x +
        gl_LocalInvocationID.y * PALETTE_REGION_SIZE +
        gl_LocalInvocationID.z * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE;
    u32 palette_region_index =
        gl_WorkGroupID.x +
        gl_WorkGroupID.y * PALETTES_PER_CHUNK_AXIS +
        (gl_WorkGroupID.z - temp_chunk_index * PALETTES_PER_CHUNK_AXIS) * PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS;

    TempVoxel my_voxel = deref(temp_voxel_chunk_ptr).voxels[inchunk_voxel_index];
    u32 my_palette_index = 0;

    b32 whole_palette_uniform = false;

    {
        u32 lod_index_x8 = uniformity_lod_index(8)(inchunk_voxel_i / 8);
        u32 lod_mask_x8 = uniformity_lod_mask(inchunk_voxel_i / 8);
        whole_palette_uniform = !voxel_rw_uniformity_lod_nonuniform(8)(voxel_chunk_ptr, lod_index_x8, lod_mask_x8);
    }

    if (whole_palette_uniform) {
        if (palette_region_voxel_index == 0) {
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].variant_n = 1;
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].blob_offset = my_voxel.col_and_id;
        }
    } else {
        u32 palette_size = 512;
        if (palette_size > PALETTE_MAX_COMPRESSED_VARIANT_N) {
            if (palette_region_voxel_index == 0) {
                deref(voxel_chunk_ptr).palette_headers[palette_region_index].variant_n = palette_size;
                deref(voxel_chunk_ptr).palette_headers[palette_region_index].blob_offset = gpu_malloc(daxa_push_constant.gpu_allocator, 512);
            }
            barrier();
            memoryBarrier();
            memoryBarrierShared();
            groupMemoryBarrier();

            u32 final_blob_offset = deref(voxel_chunk_ptr).palette_headers[palette_region_index].blob_offset;
            deref(daxa_push_constant.gpu_allocator.heap[final_blob_offset + palette_region_voxel_index]) = my_voxel.col_and_id;
        } else {
            // TODO...
        }
    }
}
#undef VOXEL_WORLD
#undef SETTINGS
