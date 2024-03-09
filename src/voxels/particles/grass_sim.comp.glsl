#include "voxel_particles.inl"

DAXA_DECL_PUSH_CONSTANT(GrassStrandSimComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(VoxelParticlesState) particles_state = push.uses.particles_state;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_BufferPtr)
daxa_RWBufferPtr(uint) cube_rendered_particle_indices = push.uses.cube_rendered_particle_indices;
daxa_RWBufferPtr(uint) splat_rendered_particle_indices = push.uses.splat_rendered_particle_indices;
daxa_RWBufferPtr(uint) placed_voxel_particles = push.uses.placed_voxel_particles;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
}
