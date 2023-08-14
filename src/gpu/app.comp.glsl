#include <shared/app.inl>

#if STARTUP_COMPUTE

#include <utils/player.glsl>
#include <voxels/core.glsl>

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    player_startup(gpu_input, globals);
    voxel_world_startup(globals, VOXELS_RW_BUFFER_PTRS);
}

#endif

#if PERFRAME_COMPUTE

#include <utils/player.glsl>

#include <voxels/core.glsl>
#include <voxels/voxel_particle.glsl>

#define INPUT deref(gpu_input)
#define BRUSH_STATE deref(globals).brush_state
#define PLAYER deref(globals).player
#define CHUNKS(i) deref(voxel_chunks[i])
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    player_perframe(gpu_input, globals);
    voxel_world_perframe(gpu_input, gpu_output, globals, VOXELS_RW_BUFFER_PTRS);

    {
        f32vec2 frame_dim = INPUT.frame_dim;
        f32vec2 inv_frame_dim = f32vec2(1.0) / frame_dim;
        f32vec2 uv = get_uv(deref(gpu_input).mouse.pos, f32vec4(frame_dim, inv_frame_dim));
        ViewRayContext vrc = vrc_from_uv(globals, uv);
        f32vec3 ray_dir = ray_dir_ws(vrc);
        f32vec3 cam_pos = ray_origin_ws(vrc);
        f32vec3 ray_pos = cam_pos;
        voxel_trace(VoxelTraceInfo(VOXELS_BUFFER_PTRS, ray_dir, MAX_STEPS, MAX_DIST, 0.0, true), ray_pos);

        if (BRUSH_STATE.is_editing == 0) {
            BRUSH_STATE.initial_ray = ray_pos - cam_pos;
        }

        deref(globals).brush_input.prev_pos = deref(globals).brush_input.pos;
        deref(globals).brush_input.pos = length(BRUSH_STATE.initial_ray) * ray_dir + cam_pos + f32vec3(deref(globals).player.chunk_offset);

        if (INPUT.actions[GAME_ACTION_BRUSH_A] != 0) {
            {
                // ChunkWorkItem brush_work_item;
                // TODO: Issue a work item with a correct root coordinate. I think that we should turn this
                // coordinate space from being in root node space, to actually be in root CHILD node space.
                // This would make it so that we can issue work items with more granularity.
                // brush_work_item.i = i32vec3(0, 0, 0);
                // brush_work_item.brush_id = BRUSH_FLAGS_USER_BRUSH_A;
                // brush_work_item.brush_input = deref(globals).brush_input;
                // zero_work_item_children(brush_work_item);
                // queue_root_work_item(globals, brush_work_item);
            }
            BRUSH_STATE.is_editing = 1;
        } else if (INPUT.actions[GAME_ACTION_BRUSH_B] != 0) {
            if (BRUSH_STATE.is_editing == 0) {
                BRUSH_STATE.initial_frame = INPUT.frame_index;
            }
            {
                // ChunkWorkItem brush_work_item;
                // brush_work_item.i = i32vec3(0, 0, 0);
                // brush_work_item.brush_id = BRUSH_FLAGS_USER_BRUSH_B;
                // brush_work_item.brush_input = deref(globals).brush_input;
                // zero_work_item_children(brush_work_item);
                // queue_root_work_item(globals, brush_work_item);
            }
            BRUSH_STATE.is_editing = 1;
        } else {
            BRUSH_STATE.is_editing = 0;
        }
    }

    deref(gpu_output[INPUT.fif_index]).player_pos = PLAYER.pos + f32vec3(PLAYER.chunk_offset);
    deref(gpu_output[INPUT.fif_index]).player_rot = f32vec3(PLAYER.yaw, PLAYER.pitch, PLAYER.roll);
    deref(gpu_output[INPUT.fif_index]).chunk_offset = f32vec3(PLAYER.chunk_offset);

    deref(globals).voxel_particles_state.simulation_dispatch = u32vec3(MAX_SIMULATED_VOXEL_PARTICLES / 64, 1, 1);
    deref(globals).voxel_particles_state.draw_params.vertex_count = 0;
    deref(globals).voxel_particles_state.draw_params.instance_count = 1;
    deref(globals).voxel_particles_state.draw_params.first_vertex = 0;
    deref(globals).voxel_particles_state.draw_params.first_instance = 0;
    deref(globals).voxel_particles_state.place_count = 0;
    deref(globals).voxel_particles_state.place_bounds_min = u32vec3(1000000);
    deref(globals).voxel_particles_state.place_bounds_max = u32vec3(0);

    if (INPUT.frame_index == 0) {
        for (u32 i = 0; i < MAX_SIMULATED_VOXEL_PARTICLES; ++i) {
            SimulatedVoxelParticle self = deref(simulated_voxel_particles[i]);
            particle_spawn(self, i);
            deref(simulated_voxel_particles[i]) = self;
        }
    }
}
#undef CHUNKS
#undef PLAYER
#undef INPUT

#endif
