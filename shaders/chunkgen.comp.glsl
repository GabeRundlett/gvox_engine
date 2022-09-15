#version 450

#include <shared.inl>
#include <utils/voxel.glsl>

DAXA_USE_PUSH_CONSTANT(ChunkgenCompPush)

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    u32vec3 voxel_i = gl_GlobalInvocationID.xyz;
    u32 voxel_index = voxel_i.x + voxel_i.y * CHUNK_SIZE + voxel_i.z * CHUNK_SIZE * CHUNK_SIZE;

    Voxel result;
    result.col = f32vec3(1, 0.5, 0.8);
    result.nrm = normalize(f32vec3(1, -1, 0));
    result.mat_id = voxel_index;

    // CHUNK.packed_voxels[voxel_index] = pack_voxel(result);
}
