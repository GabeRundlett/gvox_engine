#include "voxel_particles.inl"

DAXA_DECL_PUSH_CONSTANT(VoxelParticleSimComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(VoxelParticlesState) particles_state = push.uses.particles_state;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_BufferPtr)
daxa_RWBufferPtr(SimulatedVoxelParticle) simulated_voxel_particles = push.uses.simulated_voxel_particles;
daxa_RWBufferPtr(uint) rendered_voxel_particles = push.uses.rendered_voxel_particles;
daxa_RWBufferPtr(uint) placed_voxel_particles = push.uses.placed_voxel_particles;

#include "particle.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint particle_index = gl_GlobalInvocationID.x;
    SimulatedVoxelParticle self = deref(advance(simulated_voxel_particles, particle_index));

    bool should_place = false;
    particle_update(self, VOXELS_BUFFER_PTRS, gpu_input, should_place);
    deref(advance(simulated_voxel_particles, particle_index)) = self;

    // if (self.pos.z < 600.5) {
    //     return;
    // }

    if (should_place) {
        uvec3 my_voxel_i = uvec3(self.pos * VOXEL_SCL);
        const uvec3 max_pos = uvec3(2048);
        if (my_voxel_i.x < max_pos.x && my_voxel_i.y < max_pos.y && my_voxel_i.z < max_pos.z) {
            // Commented out, since placing particles in the voxel volume is not well optimized yet.

            // uint my_place_index = atomicAdd(deref(particles_state).place_count, 1);
            // if (my_place_index == 0) {
            //     ChunkWorkItem brush_work_item;
            //     brush_work_item.i = uvec3(0);
            //     brush_work_item.brush_id = BRUSH_FLAGS_PARTICLE_BRUSH;
            //     brush_work_item.brush_input = deref(particles_state).brush_input;
            //     zero_work_item_children(brush_work_item);
            //     queue_root_work_item(particles_state, brush_work_item);
            // }
            // deref(advance(placed_voxel_particles, my_place_index)) = particle_index;
            // atomicMin(deref(particles_state).place_bounds_min.x, my_voxel_i.x);
            // atomicMin(deref(particles_state).place_bounds_min.y, my_voxel_i.y);
            // atomicMin(deref(particles_state).place_bounds_min.z, my_voxel_i.z);
            // atomicMax(deref(particles_state).place_bounds_max.x, my_voxel_i.x);
            // atomicMax(deref(particles_state).place_bounds_max.y, my_voxel_i.y);
            // atomicMax(deref(particles_state).place_bounds_max.z, my_voxel_i.z);
        }
    }

    if (self.flags == 0) {
        return;
    }

    uint my_render_index = atomicAdd(deref(particles_state).draw_params.vertex_count, 36) / 36;

    deref(advance(rendered_voxel_particles, my_render_index)) = particle_index;
}
