#include <shared/shared.inl>
#include <utils/voxel_particle.glsl>

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    u32 particle_index = gl_GlobalInvocationID.x;
    SimulatedVoxelParticle self = deref(simulated_voxel_particles[particle_index]);
    if (self.flags == 0) {
        return;
    }
    particle_update(self, gpu_input);
    deref(simulated_voxel_particles[particle_index]) = self;

#if USE_POINTS
    u32 my_render_index = atomicAdd(deref(globals).voxel_particles_state.draw_params.vertex_count, 1);
#else
    u32 my_render_index = atomicAdd(deref(globals).voxel_particles_state.draw_params.vertex_count, 36) / 36;
#endif
    deref(rendered_voxel_particles[my_render_index]) = particle_index;
}
