
#include <utils/player.glsl>
#include <utils/voxel_world.glsl>
#include <utils/voxel_malloc.glsl>
#include <utils/trace.glsl>

#define SETTINGS deref(settings)
#define INPUT deref(gpu_input)
#define BRUSH_STATE deref(globals).brush_state
#define THREAD_POOL deref(globals).chunk_thread_pool_state
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    player_perframe(settings, gpu_input, globals);
    voxel_world_perframe(settings, gpu_input, globals);

    deref(gpu_output[INPUT.fif_index]).job_counters_packed = THREAD_POOL.job_counters_packed;
    THREAD_POOL.total_jobs_ran2 += THREAD_POOL.total_jobs_ran;
    deref(gpu_output[INPUT.fif_index]).total_jobs_ran = THREAD_POOL.total_jobs_ran2;
    u64 packed_counters = atomicAdd(THREAD_POOL.job_counters_packed, 0);
    u32 available_threads_queue_top = u32(packed_counters >> 0x00);
    u32 available_threads_queue_bottom = u32(packed_counters >> 0x20);
    available_threads_queue_top &= MAX_NODE_THREADS - 1;
    available_threads_queue_bottom &= MAX_NODE_THREADS - 1;

    u32 active_threads = 0;
    for (u32 i = 0; i < MAX_NODE_THREADS; ++i) {
        if (THREAD_POOL.chunk_node_work_items[i].flags != 0) {
            ++active_threads;
        }
    }

    {
        f32vec2 frame_dim = INPUT.frame_dim;
        f32vec2 inv_frame_dim = f32vec2(1.0) / frame_dim;
        f32 aspect = frame_dim.x * inv_frame_dim.y;
        f32vec2 uv = (deref(gpu_input).mouse.pos * inv_frame_dim - 0.5) * f32vec2(2 * aspect, 2);
        f32vec3 ray_pos = create_view_pos(deref(globals).player);
        f32vec3 cam_pos = ray_pos;
        f32vec3 ray_dir = create_view_dir(deref(globals).player, uv);
        u32vec3 chunk_n = u32vec3(1u << SETTINGS.log2_chunks_per_axis);
        trace(voxel_malloc_global_allocator, voxel_chunks, chunk_n, ray_pos, ray_dir);

        if (BRUSH_STATE.is_editing == 0) {
            BRUSH_STATE.initial_ray = ray_pos - cam_pos;
        }

        BRUSH_STATE.prev_pos = BRUSH_STATE.pos;
        BRUSH_STATE.pos = length(BRUSH_STATE.initial_ray) * ray_dir + cam_pos;

        if (INPUT.actions[GAME_ACTION_BRUSH_A] != 0 || INPUT.actions[GAME_ACTION_BRUSH_B] != 0) {
            if (BRUSH_STATE.is_editing == 0) {
                if (active_threads < MAX_NODE_THREADS) {
                    ChunkNodeWorkItem new_work_item;
                    new_work_item.i = 0;
                    new_work_item.flags = CHUNK_WORK_FLAG_IS_READY_BIT;

                    u32 work_index = u32(atomicAdd(THREAD_POOL.job_counters_packed, u64(1) << 0x20)) & (MAX_NODE_THREADS - 1);
                    THREAD_POOL.chunk_node_work_items[work_index] = new_work_item;
                    ++active_threads;
                }
            }
            BRUSH_STATE.is_editing = 1;
        } else {
            BRUSH_STATE.is_editing = 0;
        }
    }

    deref(gpu_output[INPUT.fif_index]).player_pos = deref(globals).player.pos;

#if USE_OLD_ALLOC
    deref(gpu_output[INPUT.fif_index]).heap_size = deref(voxel_malloc_global_allocator).offset;
#else
    deref(gpu_output[INPUT.fif_index]).heap_size =
        (deref(voxel_malloc_global_allocator).page_count -
         deref(voxel_malloc_global_allocator).available_pages_stack_size) *
        VOXEL_MALLOC_PAGE_SIZE_U32S;

    // Debug - reset the allocator
    // deref(voxel_malloc_global_allocator).page_count = 0;
    // deref(voxel_malloc_global_allocator).available_pages_stack_size = 0;
    // deref(voxel_malloc_global_allocator).released_pages_stack_size = 0;
#endif

    THREAD_POOL.job_counters_packed = (u64(available_threads_queue_top) << 0x00) | (u64(available_threads_queue_bottom) << 0x20);
    THREAD_POOL.total_jobs_ran = 0;

    if (active_threads == 0) {
        deref(globals).indirect_dispatch.chunk_hierarchy_dispatch = u32vec3(0, 0, 0);
    } else {
        deref(globals).indirect_dispatch.chunk_hierarchy_dispatch = u32vec3(MAX_NODE_THREADS, 1, 1);
    }

    voxel_malloc_perframe(
        gpu_input,
        voxel_malloc_global_allocator);
}
#undef INPUT
#undef SETTINGS
