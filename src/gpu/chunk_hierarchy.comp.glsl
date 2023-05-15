#extension GL_EXT_shader_atomic_int64 : require

#include <shared/shared.inl>

#define THREAD_POOL deref(globals).chunk_thread_pool_state

u32 work_index;
ChunkNodeWorkItem work_item;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    work_index = gl_WorkGroupID.x;

    while (true) {
        barrier();
        memoryBarrier();
        u32 prev_threads_ran = atomicAdd(THREAD_POOL.total_jobs_ran, 0);
        if (prev_threads_ran >= 1) {
            break;
        }
        work_item = THREAD_POOL.chunk_node_work_items[work_index];
        // barrier, then we'll add our thread to the queue of available threads
        barrier();
        if (work_item.flags != 0) {
            // say you can write a job to our worker
            if (gl_LocalInvocationIndex == 0) {
                atomicAdd(THREAD_POOL.total_jobs_ran, 1);
                if ((work_item.flags & CHUNK_WORK_FLAG_IS_READY_BIT) == 1) {
                    // make unready
                    atomicOr(THREAD_POOL.chunk_node_work_items[work_index].flags, CHUNK_WORK_FLAG_IS_ACTIVE_BIT);
                    // push myself to the available queue
                    u32 index = u32(atomicAdd(THREAD_POOL.job_counters_packed, u64(1) << 0x00)) & (MAX_NODE_THREADS - 1);
                    THREAD_POOL.available_threads_queue[index] = work_index;
                }
            }
            // now we can do our work
            if (work_item.i < 2) {
                ChunkNodeWorkItem new_work_item;
                new_work_item.i = work_item.i + 1;
                // push new job to the queue
                while (true) {
                    prev_threads_ran = atomicAdd(THREAD_POOL.total_jobs_ran, 0);
                    if (prev_threads_ran >= 1) {
                        break;
                    }
                    u64 packed_counters = atomicAdd(THREAD_POOL.job_counters_packed, 0);
                    u32 available_threads_queue_top = u32(packed_counters >> 0x00);
                    u32 available_threads_queue_bottom = u32(packed_counters >> 0x20);
                    if (available_threads_queue_top - available_threads_queue_bottom > 0) {
                        u64 new_packed_counters = packed_counters + (u64(1) << 0x20);
                        u64 old_value = atomicCompSwap(THREAD_POOL.job_counters_packed, packed_counters, new_packed_counters);
                        if (packed_counters == old_value) {
                            // I managed to swap in!
                            u32 out_work_index = available_threads_queue_bottom & (MAX_NODE_THREADS - 1);
                            atomicOr(THREAD_POOL.chunk_node_work_items[out_work_index].flags, CHUNK_WORK_FLAG_IS_READY_BIT);
                            THREAD_POOL.chunk_node_work_items[out_work_index].i = new_work_item.i;
                            break;
                        }
                    }
                }
                if (gl_LocalInvocationIndex == 0) {
                    atomicAnd(THREAD_POOL.chunk_node_work_items[work_index].flags, ~CHUNK_WORK_FLAG_IS_ACTIVE_BIT);
                }
            }
            // OOPS... this can't work at all. if one thread succeeds but the rest are waiting, the first thread
            // will be waiting in the barrier. This means that thread is just dead-locked by its workgroup peers.
        }
    }
}
