#include <shared/app.inl>
#include <voxels/voxel_particle.glsl>

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    u32 particle_index = gl_GlobalInvocationID.x;
    SimulatedVoxelParticle self = deref(simulated_voxel_particles[particle_index]);
    // if (self.flags == 0) {
    //     return;
    // }

    bool should_place = false;
    particle_update(self, gpu_input, should_place);
    deref(simulated_voxel_particles[particle_index]) = self;

    // if (self.pos.z < 600.5) {
    //     return;
    // }

    if (should_place) {
        u32vec3 my_voxel_i = u32vec3(self.pos * VOXEL_SCL);
        const u32vec3 max_pos = u32vec3(2048);
        if (my_voxel_i.x < max_pos.x && my_voxel_i.y < max_pos.y && my_voxel_i.z < max_pos.z) {
            // Commented out, since placing particles in the voxel volume is not well optimized yet.

            // u32 my_place_index = atomicAdd(deref(globals).voxel_particles_state.place_count, 1);
            // if (my_place_index == 0) {
            //     ChunkWorkItem brush_work_item;
            //     brush_work_item.i = u32vec3(0);
            //     brush_work_item.brush_id = BRUSH_FLAGS_PARTICLE_BRUSH;
            //     brush_work_item.brush_input = deref(globals).brush_input;
            //     zero_work_item_children(brush_work_item);
            //     queue_root_work_item(globals, brush_work_item);
            // }
            // deref(placed_voxel_particles[my_place_index]) = particle_index;
            // atomicMin(deref(globals).voxel_particles_state.place_bounds_min.x, my_voxel_i.x);
            // atomicMin(deref(globals).voxel_particles_state.place_bounds_min.y, my_voxel_i.y);
            // atomicMin(deref(globals).voxel_particles_state.place_bounds_min.z, my_voxel_i.z);
            // atomicMax(deref(globals).voxel_particles_state.place_bounds_max.x, my_voxel_i.x);
            // atomicMax(deref(globals).voxel_particles_state.place_bounds_max.y, my_voxel_i.y);
            // atomicMax(deref(globals).voxel_particles_state.place_bounds_max.z, my_voxel_i.z);
        }
    }

    u32 my_render_index = atomicAdd(deref(globals).voxel_particles_state.draw_params.vertex_count, 36) / 36;

    deref(rendered_voxel_particles[my_render_index]) = particle_index;
}
