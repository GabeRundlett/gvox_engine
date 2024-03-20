#include "sim_particle.inl"
#include "sim_particle.glsl"

DAXA_DECL_PUSH_CONSTANT(SimParticleSimComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(VoxelParticlesState) particles_state = push.uses.particles_state;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_BufferPtr)
daxa_RWBufferPtr(SimulatedVoxelParticle) simulated_voxel_particles = push.uses.simulated_voxel_particles;
daxa_RWBufferPtr(PackedParticleVertex) cube_rendered_particle_verts = push.uses.cube_rendered_particle_verts;
daxa_RWBufferPtr(PackedParticleVertex) shadow_cube_rendered_particle_verts = push.uses.shadow_cube_rendered_particle_verts;
daxa_RWBufferPtr(PackedParticleVertex) splat_rendered_particle_verts = push.uses.splat_rendered_particle_verts;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint particle_index = gl_GlobalInvocationID.x;
    SimulatedVoxelParticle self = deref(advance(simulated_voxel_particles, particle_index));

    bool should_place = false;
    particle_update(self, particle_index, VOXELS_BUFFER_PTRS, gpu_input, should_place);

    deref(advance(simulated_voxel_particles, particle_index)) = self;

    // if (should_place) {
    //     particle_voxelize(placed_voxel_particles, particles_state, self, particle_index);
    // }

    if (self.flags != 0) {
        PackedParticleVertex packed_vertex = PackedParticleVertex(particle_index);
        ParticleVertex vert = get_sim_particle_vertex(gpu_input, daxa_BufferPtr(SimulatedVoxelParticle)(as_address(simulated_voxel_particles)), packed_vertex);
        particle_render(cube_rendered_particle_verts, shadow_cube_rendered_particle_verts, splat_rendered_particle_verts, particles_state, gpu_input, vert, packed_vertex, true);
    }
}
