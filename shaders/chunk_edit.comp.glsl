#include <utils/impl_brush_input.glsl>
#define BRUSH_INPUT 1

#include <shared/shared.inl>

DAXA_USE_PUSH_CONSTANT(ChunkEditCompPush)

#include <utils/impl_brush_kernel.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    i32vec3 pick_chunk_i = get_chunk_i(get_voxel_i_WORLD(GLOBALS.brush_origin));
    u32 workgroup_index = gl_WorkGroupID.z / 8;
    u32 chunk_index = VOXEL_WORLD.chunk_update_indices[workgroup_index];
    if (VOXEL_WORLD.chunks_genstate[chunk_index].edit_stage != 3)
        return;
    u32vec3 voxel_i = gl_GlobalInvocationID.xyz - u32vec3(0, 0, workgroup_index * 64);
    u32 voxel_index = voxel_i.x + voxel_i.y * CHUNK_SIZE + voxel_i.z * CHUNK_SIZE * CHUNK_SIZE;
    f32vec3 voxel_p = f32vec3(voxel_i) / VOXEL_SCL + VOXEL_WORLD.voxel_chunks[chunk_index].box.bound_min;
    Voxel result = brush_kernel(voxel_p);
    if (result.block_id != BlockID_Debug) {
        VOXEL_WORLD.voxel_chunks[chunk_index].packed_voxels[voxel_index] = pack_voxel(result);
    }
}
