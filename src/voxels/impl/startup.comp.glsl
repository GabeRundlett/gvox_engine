#include "voxel_world.inl"

DAXA_DECL_PUSH_CONSTANT(VoxelWorldStartupComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_RWBufferPtr)

#include <voxels/voxels.glsl>

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    VoxelRWBufferPtrs ptrs = VOXELS_RW_BUFFER_PTRS;
    deref(ptrs.globals).chunk_update_n = 0;
}
