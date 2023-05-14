#extension GL_EXT_shader_atomic_int64 : require

#include <shared/shared.inl>

#define THREAD_POOL deref(globals).chunk_thread_pool_state

shared u32 acquired_job_index;

void acquire_job() {
    if (gl_LocalInvocationIndex == 0) {
        u64 packed_counters = atomicAdd(THREAD_POOL.job_counters_packed, 0);
        u64 packed_counters2 = atomicAdd(THREAD_POOL.job_counters_packed2, u64(1u) << 0x20);
        u32 jobs_total_count = u32(packed_counters >> 0x00) & 0xffffffff;
        u32 jobs_ready_count = u32(packed_counters2 >> 0x00) & 0xffffffff;
        u32 jobs_finished_count = u32(packed_counters >> 0x20) & 0xffffffff;
        acquired_job_index = u32(packed_counters2 >> 0x20) & 0xffffffff;
        if (acquired_job_index < MAX_NODE_UPDATES_PER_FRAME && jobs_total_count != jobs_finished_count) {
            // Wait until our work state has been written out
            while (jobs_ready_count <= acquired_job_index) {
                packed_counters = atomicAdd(THREAD_POOL.job_counters_packed, 0);
                packed_counters2 = atomicAdd(THREAD_POOL.job_counters_packed2, 0);
                jobs_total_count = u32(packed_counters >> 0x00) & 0xffffffff;
                jobs_ready_count = u32(packed_counters2 >> 0x00) & 0xffffffff;
                jobs_finished_count = u32(packed_counters >> 0x20) & 0xffffffff;
                if (jobs_total_count == jobs_finished_count) {
                    acquired_job_index = 0xffffffff;
                    break;
                }
            }
        } else {
            acquired_job_index = 0xffffffff;
        }
    }
    barrier();
    memoryBarrier();
}

void finish_job() {
    // I would have thought this was necessary
    // barrier();

    if (gl_LocalInvocationIndex == 0) {
        atomicAdd(THREAD_POOL.job_counters_packed, u64(1u) << 0x20);
    }
}

void spawn_job(ChunkNodeWorkItem new_work_item) {
    u64 packed_counters = atomicAdd(THREAD_POOL.job_counters_packed, 1);
    u32 output_job_index = u32(packed_counters >> 0x00) & 0xffffffff;
    // write job desc
    if (output_job_index < MAX_NODE_UPDATES_PER_FRAME) {
        THREAD_POOL.chunk_node_work_items[output_job_index] = new_work_item;
        // notify worker that the job desc is ready
        atomicAdd(THREAD_POOL.job_counters_packed2, 1);
    }
}

void do_job() {
    ChunkNodeWorkItem work_item = THREAD_POOL.chunk_node_work_items[acquired_job_index];

    // potentially 0-512 new jobs can be spawned!
    if (work_item.i < 2) {
        ChunkNodeWorkItem new_work_item;
        new_work_item.i = work_item.i + 1;
        spawn_job(new_work_item);
    }
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    while (true) {
        acquire_job();

        // No more work to do!
        if (acquired_job_index == 0xffffffff) {
            break;
        }

        do_job();

        finish_job();
    }
}
