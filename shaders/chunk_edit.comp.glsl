#include <shared/shared.inl>

DAXA_USE_PUSH_CONSTANT(ChunkEditCompPush)

#include <utils/voxel.glsl>

void chunk_edit(in u32 chunk_index) {
    u32vec3 voxel_i = gl_GlobalInvocationID.xyz;
    u32 voxel_index = voxel_i.x + voxel_i.y * CHUNK_SIZE + voxel_i.z * CHUNK_SIZE * CHUNK_SIZE;

    f32vec3 voxel_p = f32vec3(voxel_i) / VOXEL_SCL + VOXEL_CHUNKS[chunk_index].box.bound_min;
    Voxel result;

    result.col = f32vec3(1, 0.5, 0.8);
    result.nrm = f32vec3(0, 0, 1);
    result.block_id = BlockID_Air;

    if (length(voxel_p - GLOBALS.pick_pos) < 10.0 / VOXEL_SCL) {
        VOXEL_CHUNKS[chunk_index].packed_voxels[voxel_index] = pack_voxel(result);
    }
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    i32vec3 pick_chunk_i = get_chunk_i(get_voxel_i(GLOBALS.pick_pos));
    u32 chunk_index = get_chunk_index(pick_chunk_i);
    if (VOXEL_WORLD.chunks_genstate[chunk_index].edit_stage != 4)
        return;

    chunk_edit(chunk_index);
}
