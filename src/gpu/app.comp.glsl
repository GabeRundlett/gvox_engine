#include <shared/app.inl>

#if STARTUP_COMPUTE

#include <utils/player.glsl>
#include <voxels/voxel_world.glsl>

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    player_startup(gpu_input, globals);
    voxel_world_startup(globals);
}

#endif

#if PERFRAME_COMPUTE

#include <utils/player.glsl>

#include <voxels/trace.glsl>
#include <voxels/voxel_world.glsl>
#include <voxels/voxel_malloc.glsl>
#include <voxels/voxel_particle.glsl>

// #define UserAllocatorType VoxelLeafChunkAllocator
// #define UserIndexType u32
// #include <utils/allocator.glsl>

// #define UserAllocatorType VoxelParentChunkAllocator
// #define UserIndexType u32
// #include <utils/allocator.glsl>

// Queue a L0 terrain generation item
// void queue_terrain_generation_work_item(i32vec3 chunk_offset) {
//     ChunkWorkItem terrain_work_item;
//     terrain_work_item.i = i32vec3(0);
//     terrain_work_item.chunk_offset = chunk_offset;
//     terrain_work_item.brush_id = BRUSH_FLAGS_WORLD_BRUSH;
//     zero_work_item_children(terrain_work_item);
//     queue_root_work_item(globals, terrain_work_item);
// }

#define INPUT deref(gpu_input)
#define BRUSH_STATE deref(globals).brush_state
#define PLAYER deref(globals).player
#define CHUNKS(i) deref(voxel_chunks[i])
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    // Update previous chunk offset
    PLAYER.prev_chunk_offset = PLAYER.chunk_offset;

    player_perframe(gpu_input, globals);
    voxel_world_perframe(gpu_input, globals);

    {
        f32vec2 frame_dim = INPUT.frame_dim;
        f32vec2 inv_frame_dim = f32vec2(1.0) / frame_dim;
        f32vec2 uv = get_uv(deref(gpu_input).mouse.pos, f32vec4(frame_dim, inv_frame_dim));
        ViewRayContext vrc = vrc_from_uv(globals, uv);
        f32vec3 ray_dir = ray_dir_ws(vrc);
        f32vec3 cam_pos = ray_origin_ws(vrc);
        f32vec3 ray_pos = cam_pos;
        u32vec3 chunk_n = u32vec3(1u << deref(gpu_input).log2_chunks_per_axis);
        trace_hierarchy_traversal(VoxelTraceInfo(VOXEL_TRACE_INFO_PTRS, chunk_n, ray_dir, MAX_STEPS, MAX_DIST, 0.0, true), ray_pos);

        if (BRUSH_STATE.is_editing == 0) {
            BRUSH_STATE.initial_ray = ray_pos - cam_pos;
        }

        deref(globals).brush_input.prev_pos = deref(globals).brush_input.pos;
        deref(globals).brush_input.pos = length(BRUSH_STATE.initial_ray) * ray_dir + cam_pos + f32vec3(deref(globals).player.chunk_offset) * CHUNK_WORLDSPACE_SIZE;

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

    deref(gpu_output[INPUT.fif_index]).player_pos = PLAYER.pos + f32vec3(PLAYER.chunk_offset) * CHUNK_WORLDSPACE_SIZE;
    deref(gpu_output[INPUT.fif_index]).player_rot = f32vec3(PLAYER.yaw, PLAYER.pitch, PLAYER.roll);
    deref(gpu_output[INPUT.fif_index]).chunk_offset = f32vec3(PLAYER.chunk_offset);

    VoxelMallocPageAllocator_perframe(voxel_malloc_page_allocator);
    // VoxelLeafChunkAllocator_perframe(voxel_leaf_chunk_allocator);
    // VoxelParentChunkAllocator_perframe(voxel_parent_chunk_allocator);

    deref(gpu_output[INPUT.fif_index]).voxel_malloc_output.current_element_count =
        VoxelMallocPageAllocator_get_consumed_element_count(daxa_BufferPtr(VoxelMallocPageAllocator)(voxel_malloc_page_allocator));
    // deref(gpu_output[INPUT.fif_index]).voxel_leaf_chunk_output.current_element_count =
    //     VoxelLeafChunkAllocator_get_consumed_element_count(voxel_leaf_chunk_allocator);
    // deref(gpu_output[INPUT.fif_index]).voxel_parent_chunk_output.current_element_count =
    //     VoxelParentChunkAllocator_get_consumed_element_count(voxel_parent_chunk_allocator);

    deref(gpu_output[INPUT.fif_index]).job_counters_packed = 0;
    deref(gpu_output[INPUT.fif_index]).total_jobs_ran = 0;

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
