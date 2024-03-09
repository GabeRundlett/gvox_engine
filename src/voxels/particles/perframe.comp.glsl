#include "voxel_particles.inl"

DAXA_DECL_PUSH_CONSTANT(VoxelParticlePerframeComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuOutput) gpu_output = push.uses.gpu_output;
daxa_RWBufferPtr(VoxelParticlesState) particles_state = push.uses.particles_state;
daxa_RWBufferPtr(SimulatedVoxelParticle) simulated_voxel_particles = push.uses.simulated_voxel_particles;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_RWBufferPtr)

#include <renderer/kajiya/inc/camera.glsl>
#include <voxels/voxels.glsl>
#include "particle.glsl"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    deref(particles_state).simulation_dispatch = uvec3(MAX_SIMULATED_VOXEL_PARTICLES / 64, 1, 1);

    deref(particles_state).cube_draw_params.index_count = 8;
    deref(particles_state).cube_draw_params.instance_count = 0;
    deref(particles_state).cube_draw_params.first_index = 0;
    deref(particles_state).cube_draw_params.vertex_offset = 0;
    deref(particles_state).cube_draw_params.first_instance = 0;

    deref(particles_state).splat_draw_params.vertex_count = 0;
    deref(particles_state).splat_draw_params.instance_count = 1;
    deref(particles_state).splat_draw_params.first_vertex = 0;
    deref(particles_state).splat_draw_params.first_instance = 0;

    deref(particles_state).place_count = 0;
    deref(particles_state).place_bounds_min = uvec3(1000000);
    deref(particles_state).place_bounds_max = uvec3(0);
}
