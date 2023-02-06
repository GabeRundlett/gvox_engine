#include <utils/impl_brush_input.glsl>
#define BRUSH_INPUT 1

#include <shared/shared.inl>

DAXA_USE_PUSH_CONSTANT(ChunkEditCompPush)

#include <utils/impl_brush_kernel.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    u32 workgroup_index = gl_WorkGroupID.z / 8;
    u32 chunk_index = VOXEL_BRUSH.chunk_update_indices[workgroup_index];
    u32vec3 voxel_i = gl_GlobalInvocationID.xyz - u32vec3(0, 0, workgroup_index * 64);
    u32 voxel_index = get_voxel_index(voxel_i);
    f32vec3 voxel_p = f32vec3(voxel_i) / VOXEL_SCL + VOXEL_BRUSH.voxel_chunks[chunk_index].box.bound_min + VOXEL_BRUSH.box.bound_min;
    Voxel result = brush_kernel(voxel_p);
    // VOXEL_BRUSH.voxel_chunks[chunk_index].packed_voxels[voxel_index] = pack_voxel(result);
}
