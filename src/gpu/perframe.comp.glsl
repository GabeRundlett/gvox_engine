#include <utils/player.glsl>
#include <utils/voxel_world.glsl>
#include <utils/voxel_malloc.glsl>
#include <utils/voxel_particle.glsl>
#include <utils/trace.glsl>

#define UserAllocatorType VoxelLeafChunkAllocator
#define UserIndexType u32
#include <utils/allocator.glsl>

#define UserAllocatorType VoxelParentChunkAllocator
#define UserIndexType u32
#include <utils/allocator.glsl>

// Queue a L0 terrain generation item
void queue_terrain_generation_work_item(i32vec3 chunk_offset) {
    ChunkWorkItem terrain_work_item;
    terrain_work_item.i = i32vec3(0);
    terrain_work_item.chunk_offset = chunk_offset;
    terrain_work_item.brush_id = CHUNK_FLAGS_WORLD_BRUSH;
    zero_work_item_children(terrain_work_item);
    queue_root_work_item(globals, terrain_work_item);
}

#define SETTINGS deref(settings)
#define INPUT deref(gpu_input)
#define BRUSH_STATE deref(globals).brush_state
#define THREAD_POOL deref(globals).chunk_thread_pool_state
#define PLAYER deref(globals).player
#define CHUNKS(i) deref(voxel_chunks[i])
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    // Update previous chunk offset
    PLAYER.prev_chunk_offset = PLAYER.chunk_offset;

    player_perframe(settings, gpu_input, globals);
    voxel_world_perframe(settings, gpu_input, globals);

    if (ENABLE_HIERARCHICAL_QUEUE != 0) {
        // Check if the player moved at least 1 chunk in any direction
        if (PLAYER.chunk_offset != PLAYER.prev_chunk_offset) {
            // Issue a new terrain generation
            queue_terrain_generation_work_item(PLAYER.chunk_offset);

            i32vec3 chunk_n = i32vec3(1 << SETTINGS.log2_chunks_per_axis);

            // Clamped difference between the last and current chunk offset
            i32vec3 diff = clamp(i32vec3(PLAYER.chunk_offset - PLAYER.prev_chunk_offset), -chunk_n, chunk_n);

            // If there was a movement along the x-axis, reset the correct chunks
            if (abs(diff.x) > 0) {
                u32 x_start = diff.x < 0 ? 0 : chunk_n.x - diff.x;
                u32 x_end = diff.x < 0 ? -diff.x : chunk_n.x;
                for (u32 x = x_start; x < x_end; x++)
                    for (u32 y = 0; y < chunk_n.y; y++)
                        for (u32 z = 0; z < chunk_n.z; z++)
                            CHUNKS(calc_chunk_index(u32vec3(x, y, z), chunk_n)).flags &= ~CHUNK_FLAGS_ACCEL_GENERATED;
            }
            // Same for y-axis
            if (abs(diff.y) > 0) {
                u32 y_start = diff.y < 0 ? 0 : chunk_n.y - diff.y;
                u32 y_end = diff.y < 0 ? -diff.y : chunk_n.y;
                for (u32 x = 0; x < chunk_n.x; x++)
                    for (u32 y = y_start; y < y_end; y++)
                        for (u32 z = 0; z < chunk_n.z; z++)
                            CHUNKS(calc_chunk_index(u32vec3(x, y, z), chunk_n)).flags &= ~CHUNK_FLAGS_ACCEL_GENERATED;
            }
            // Same for z-axis
            if (abs(diff.z) > 0) {
                u32 z_start = diff.z < 0 ? 0 : chunk_n.z - diff.z;
                u32 z_end = diff.z < 0 ? -diff.z : chunk_n.z;
                for (u32 x = 0; x < chunk_n.x; x++)
                    for (u32 y = 0; y < chunk_n.y; y++)
                        for (u32 z = z_start; z < z_end; z++)
                            CHUNKS(calc_chunk_index(u32vec3(x, y, z), chunk_n)).flags &= ~CHUNK_FLAGS_ACCEL_GENERATED;
            }

            // Update previous chunk offset
            PLAYER.prev_chunk_offset = PLAYER.chunk_offset;
        }
    }

    {
        f32vec2 frame_dim = INPUT.frame_dim;
        f32vec2 inv_frame_dim = f32vec2(1.0) / frame_dim;
        f32vec2 uv = get_uv(deref(gpu_input).mouse.pos, f32vec4(frame_dim, inv_frame_dim));
        ViewRayContext vrc = vrc_from_uv(globals, uv);
        f32vec3 ray_dir = ray_dir_ws(vrc);
        f32vec3 cam_pos = ray_origin_ws(vrc);
        f32vec3 ray_pos = cam_pos;
        u32vec3 chunk_n = u32vec3(1u << SETTINGS.log2_chunks_per_axis);
        trace_hierarchy_traversal(VoxelTraceInfo(voxel_malloc_page_allocator, voxel_chunks, chunk_n, ray_dir, MAX_STEPS, MAX_DIST, 0.0, true), ray_pos);

        if (BRUSH_STATE.is_editing == 0) {
            BRUSH_STATE.initial_ray = ray_pos - cam_pos;
        }

        deref(globals).brush_input.prev_pos = deref(globals).brush_input.pos;
        deref(globals).brush_input.pos = length(BRUSH_STATE.initial_ray) * ray_dir + cam_pos + f32vec3(deref(globals).player.chunk_offset) * CHUNK_WORLDSPACE_SIZE;

        if (INPUT.actions[GAME_ACTION_BRUSH_A] != 0) {
            {
                ChunkWorkItem brush_work_item;
                // TODO: Issue a work item with a correct root coordinate. I think that we should turn this
                // coordinate space from being in root node space, to actually be in root CHILD node space.
                // This would make it so that we can issue work items with more granularity.
                brush_work_item.i = i32vec3(0, 0, 0);
                brush_work_item.brush_id = CHUNK_FLAGS_USER_BRUSH_A;
                brush_work_item.brush_input = deref(globals).brush_input;
                zero_work_item_children(brush_work_item);
                queue_root_work_item(globals, brush_work_item);
            }
            BRUSH_STATE.is_editing = 1;
        } else if (INPUT.actions[GAME_ACTION_BRUSH_B] != 0) {
            if (BRUSH_STATE.is_editing == 0) {
                BRUSH_STATE.initial_frame = INPUT.frame_index;
            }
            {
                ChunkWorkItem brush_work_item;
                brush_work_item.i = i32vec3(0, 0, 0);
                brush_work_item.brush_id = CHUNK_FLAGS_USER_BRUSH_B;
                brush_work_item.brush_input = deref(globals).brush_input;
                zero_work_item_children(brush_work_item);
                queue_root_work_item(globals, brush_work_item);
            }
            BRUSH_STATE.is_editing = 1;
        } else {
            BRUSH_STATE.is_editing = 0;
        }
    }

    deref(gpu_output[INPUT.fif_index]).player_pos = PLAYER.pos + f32vec3(PLAYER.chunk_offset) * CHUNK_WORLDSPACE_SIZE;
    deref(gpu_output[INPUT.fif_index]).player_rot = f32vec3(PLAYER.yaw, PLAYER.pitch, PLAYER.roll);
    deref(gpu_output[INPUT.fif_index]).chunk_offset = f32vec3(PLAYER.chunk_offset);

    VoxelMallocPageAllocator_perframe(voxel_malloc_page_allocator);
    VoxelLeafChunkAllocator_perframe(voxel_leaf_chunk_allocator);
    VoxelParentChunkAllocator_perframe(voxel_parent_chunk_allocator);

    deref(gpu_output[INPUT.fif_index]).voxel_malloc_output.current_element_count =
        VoxelMallocPageAllocator_get_consumed_element_count(voxel_malloc_page_allocator);
    deref(gpu_output[INPUT.fif_index]).voxel_leaf_chunk_output.current_element_count =
        VoxelLeafChunkAllocator_get_consumed_element_count(voxel_leaf_chunk_allocator);
    deref(gpu_output[INPUT.fif_index]).voxel_parent_chunk_output.current_element_count =
        VoxelParentChunkAllocator_get_consumed_element_count(voxel_parent_chunk_allocator);

    THREAD_POOL.queue_index = 1 - THREAD_POOL.queue_index;

    THREAD_POOL.total_jobs_ran = THREAD_POOL.work_items_l0_uncompleted + THREAD_POOL.work_items_l1_uncompleted;

    deref(gpu_output[INPUT.fif_index]).job_counters_packed = 0;
    deref(gpu_output[INPUT.fif_index]).total_jobs_ran = THREAD_POOL.total_jobs_ran;

    // clang-format off
    THREAD_POOL.work_items_l0_queued      = THREAD_POOL.work_items_l0_uncompleted;
    THREAD_POOL.work_items_l0_completed   = 0;
    THREAD_POOL.work_items_l0_uncompleted = 0;

    THREAD_POOL.work_items_l1_queued      = THREAD_POOL.work_items_l1_uncompleted;
    THREAD_POOL.work_items_l1_completed   = 0;
    THREAD_POOL.work_items_l1_uncompleted = 0;
    // clang-format on

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
#undef SETTINGS
