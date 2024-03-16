#include "voxel_particles.inl"

DAXA_DECL_PUSH_CONSTANT(VoxelParticlePerframeComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuOutput) gpu_output = push.uses.gpu_output;
daxa_RWBufferPtr(VoxelParticlesState) particles_state = push.uses.particles_state;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_RWBufferPtr)
SIMPLE_STATIC_ALLOCATOR_BUFFERS_PUSH_USES(GrassStrandAllocator, grass_allocator)
SIMPLE_STATIC_ALLOCATOR_BUFFERS_PUSH_USES(DandelionAllocator, dandelion_allocator)

#include <renderer/kajiya/inc/camera.glsl>
#include <voxels/voxels.glsl>

#define UserAllocatorType GrassStrandAllocator
#define UserIndexType uint
#define UserMaxElementCount MAX_GRASS_BLADES
#include <utilities/allocator.glsl>

#define UserAllocatorType DandelionAllocator
#define UserIndexType uint
#define UserMaxElementCount MAX_DANDELIONS
#include <utilities/allocator.glsl>

void reset_draw_params(in out IndirectDrawIndexedParams params) {
    params.index_count = 8;
    params.instance_count = 0;
    params.first_index = 0;
    params.vertex_offset = 0;
    params.first_instance = 0;
}

void reset_draw_params(in out IndirectDrawParams params) {
    params.vertex_count = 0;
    params.instance_count = 1;
    params.first_vertex = 0;
    params.first_instance = 0;
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    deref(particles_state).simulation_dispatch = uvec3(MAX_SIMULATED_VOXEL_PARTICLES / 64, 1, 1);

    deref(particles_state).place_count = 0;
    deref(particles_state).place_bounds_min = uvec3(1000000);
    deref(particles_state).place_bounds_max = uvec3(0);

    // sim particle
    reset_draw_params(deref(particles_state).sim_particle.cube_draw_params);
    reset_draw_params(deref(particles_state).sim_particle.shadow_cube_draw_params);
    reset_draw_params(deref(particles_state).sim_particle.splat_draw_params);

    // grass
    reset_draw_params(deref(particles_state).grass.cube_draw_params);
    reset_draw_params(deref(particles_state).grass.shadow_cube_draw_params);
    reset_draw_params(deref(particles_state).grass.splat_draw_params);
    GrassStrandAllocator_perframe(grass_allocator);

    // dandelion
    reset_draw_params(deref(particles_state).dandelion.cube_draw_params);
    reset_draw_params(deref(particles_state).dandelion.shadow_cube_draw_params);
    reset_draw_params(deref(particles_state).dandelion.splat_draw_params);
    DandelionAllocator_perframe(dandelion_allocator);
}
