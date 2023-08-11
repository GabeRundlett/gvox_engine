#include <shared/app.inl>

#include <utils/voxels.glsl>

#define THREAD_POOL deref(globals).chunk_thread_pool_state

// clang-format off
#if CHUNK_LEVEL == 0
#define WORK_ITEMS             THREAD_POOL.chunk_work_items_l0[THREAD_POOL.queue_index]
#define UNCOMPLETED_WORK_ITEMS THREAD_POOL.chunk_work_items_l0[1 - THREAD_POOL.queue_index]
#define WORK_ITEMS_COMPLETED   THREAD_POOL.work_items_l0_completed
#define WORK_ITEMS_UNCOMPLETED THREAD_POOL.work_items_l0_uncompleted
#define SUB_NODE_SIZE          L1_CHUNK_SIZE
#define SUB_WORK_ITEMS         THREAD_POOL.chunk_work_items_l1[THREAD_POOL.queue_index]
#define SUB_WORK_ITEMS_QUEUED  THREAD_POOL.work_items_l1_queued
#define MAX_SUB_WORK_ITEMS     MAX_CHUNK_WORK_ITEMS_L1
#elif CHUNK_LEVEL == 1
#define WORK_ITEMS             THREAD_POOL.chunk_work_items_l1[THREAD_POOL.queue_index]
#define UNCOMPLETED_WORK_ITEMS THREAD_POOL.chunk_work_items_l1[1 - THREAD_POOL.queue_index]
#define WORK_ITEMS_COMPLETED   THREAD_POOL.work_items_l1_completed
#define WORK_ITEMS_UNCOMPLETED THREAD_POOL.work_items_l1_uncompleted
#define SUB_NODE_SIZE          L2_CHUNK_SIZE
#else
#error This should never happen
#endif
// clang-format on

shared u32 children_finished;
shared ChunkWorkItem work_item;
i32vec3 node_i;
u32 work_item_index;
u32 child_u32_index;
u32 child_bit_offset;

// First thread: Write the work item corresponding to the current work group to the GPU cache
// All other threads: Wait for the work item variable to be available
void get_work_item() {
    work_item_index = gl_WorkGroupID.x; // gl_WorkGroupID size: (number of parent work items, 1, 1)

    // Only one thread should set the work item
    if (gl_LocalInvocationIndex == 0) {
        children_finished = 0; // Reset the finished children count
        work_item = WORK_ITEMS[work_item_index]; // Set the work item
    }
    // Other threads should wait for this work_item to be available
    barrier();
    memoryBarrierShared();
}

#if CHUNK_LEVEL < 1
// Queue a L1 work item
bool queue_sub_work_item(in ChunkWorkItem new_work_item) {
    // Increment the number of L1 work items queued
    u32 new_index = atomicAdd(SUB_WORK_ITEMS_QUEUED, 1);

    // Clamp the number of work items queued so that it doesn't exceed the maximum number of L1 work items allowed
    atomicMin(SUB_WORK_ITEMS_QUEUED, MAX_SUB_WORK_ITEMS);

    // Check if the new index is less than the maximum
    if (new_index < MAX_SUB_WORK_ITEMS) {
        SUB_WORK_ITEMS[new_index] = new_work_item;
        return true;
    } else {
        return false;
    }
}
#endif

void complete_work_item() {
    barrier();
    memoryBarrierShared();
    // Run for the first 16 threads
    if (gl_LocalInvocationIndex < 16) {
        WORK_ITEMS[work_item_index].children_finished[gl_LocalInvocationIndex] = work_item.children_finished[gl_LocalInvocationIndex];
    }
    // Run for the first thread in the work group
    if (gl_LocalInvocationIndex == 0) {
        if (children_finished == 512) {
            atomicAdd(WORK_ITEMS_COMPLETED, 1);
        } else {
            u32 my_index = atomicAdd(WORK_ITEMS_UNCOMPLETED, 1);
            // TODO: check if I can write this with multiple threads?
            UNCOMPLETED_WORK_ITEMS[my_index] = work_item;
        }
    }
}

#define VOXEL_WORLD deref(globals).voxel_world
#define INDIRECT deref(globals).indirect_dispatch
#define CHUNKS(i) deref(voxel_chunks[i])
bool elect_chunk_for_update(i32vec3 chunk_i, i32vec3 chunk_offset, u32 chunk_index, u32 brush_flags, in out BrushInput brush_input) {
    // Increment the number of chunks to update
    u32 prev_update_n = atomicAdd(VOXEL_WORLD.chunk_update_n, 1);

    // Check if the work item can be added
    if (prev_update_n < MAX_CHUNK_UPDATES_PER_FRAME) {
        // Set the chunk edit dispatch z axis (64/8, 64/8, 64 x 8 x 8 / 8 = 64 x 8) = (8, 8, 512)
        atomicAdd(INDIRECT.chunk_edit_dispatch.z, CHUNK_SIZE / 8);
        atomicAdd(INDIRECT.subchunk_x2x4_dispatch.z, 1);
        atomicAdd(INDIRECT.subchunk_x8up_dispatch.z, 1);
        u32 prev_flags = atomicOr(CHUNKS(chunk_index).flags, brush_flags);
        // Set the chunk update infos
        VOXEL_WORLD.chunk_update_infos[prev_update_n].i = chunk_i;
        VOXEL_WORLD.chunk_update_infos[prev_update_n].brush_input = brush_input;
        VOXEL_WORLD.chunk_update_infos[prev_update_n].chunk_offset = chunk_offset;
        if ((prev_flags & CHUNK_FLAGS_BRUSH_MASK) != 0) {
            VOXEL_WORLD.chunk_update_infos[prev_update_n].i = INVALID_CHUNK_I;
        }
        return true;
    }
    return false;
}
#undef CHUNKS
#undef INDIRECT
#undef VOXEL_WORLD

#define MODEL deref(gvox_model)
#define CHUNKS(i) deref(voxel_chunks[i])
void perform_work_item() {
    // (const) number of chunks in each axis (1 << x = 2^x)
    u32vec3 chunk_n = u32vec3(1u << deref(gpu_input).log2_chunks_per_axis);
    // Child work item index
    i32vec3 sub_node_i = node_i * 8 + i32vec3(gl_LocalInvocationID.xyz);
    // Child index in leaf chunk space (0,0,0)->(31,31,31)
    i32vec3 sub_node_chunk_i = sub_node_i * SUB_NODE_SIZE / CHUNK_SIZE;

    // Is the current child work item completed ?
    bool thread_completed = false;

    // Check if sub-node update is necessary.
    bool needs_subdiv =
        (sub_node_chunk_i.x < i32(chunk_n.x)) &&
        (sub_node_chunk_i.y < i32(chunk_n.y)) &&
        (sub_node_chunk_i.z < i32(chunk_n.z));
    // Note: MUST CAST `chunk_n` TO A SIGNED INTEGER BEFORE COMPARISON!
    // without casting, it implicitly casts the left side to uint instead... This is absurd.
    //
    // Also, this comparison is completely broken. This comparison was originally to make sure
    // chunk in question was only subdivided if the chunk's position was within the "loaded"
    // region. Previously, the only chunk indices this needed to discard were chunks with
    // indices greater than the world size. Now that the world wraps around, the chunk_i can
    // be negative, and in general, this condition should check if the chunk_i is actually
    // inside the loaded region. When it's outside the loaded region, extra logic should
    // be implemented to store these queued work items, potentially on disk.

    // Switch depending on the brush ID to decide when to subdivide
    if ((work_item.brush_id & CHUNK_FLAGS_WORLD_BRUSH) != 0) {
#if GEN_MODEL
        i32vec3 chunk_bi = sub_node_chunk_i + 0;
        needs_subdiv = needs_subdiv && (chunk_bi.x <= (MODEL.extent_x >> 6) && chunk_bi.y <= (MODEL.extent_y >> 6) && chunk_bi.z <= (MODEL.extent_z >> 6));
#endif
        // Only subdivide in an L2 work item if the child chunk has not already been generated
        if (CHUNK_LEVEL == 1) {
            u32 chunk_index = calc_chunk_index_from_worldspace(sub_node_i, chunk_n);
            if ((CHUNKS(chunk_index).flags & CHUNK_FLAGS_ACCEL_GENERATED) != 0)
                needs_subdiv = false;
        }
        // Only subdivide if the child chunk is in a specified radius around the player
        // if (length(deref(globals).player.pos.xy/8.0 - f32vec2(sub_node_chunk_i.xy)) > 12.0) 
        //     needs_subdiv = false;
    } else if ((work_item.brush_id & CHUNK_FLAGS_USER_BRUSH_A) != 0) {
        f32vec3 chunk_pos = (f32vec3(sub_node_chunk_i) + 0.5) * CHUNK_WORLDSPACE_SIZE;
        f32vec3 delta = chunk_pos - deref(globals).brush_input.pos;
        f32vec3 dist3 = abs(delta);
        if (CHUNK_LEVEL > 0) {
            needs_subdiv = needs_subdiv && (max(dist3.x, max(dist3.y, dist3.z)) < (31.0 + CHUNK_SIZE / 2) / VOXEL_SCL);
        }
    } else if ((work_item.brush_id & CHUNK_FLAGS_USER_BRUSH_B) != 0) {
        f32vec3 chunk_pos = (f32vec3(sub_node_chunk_i) + 0.5) * CHUNK_WORLDSPACE_SIZE;
        f32vec3 delta = chunk_pos - deref(globals).brush_input.pos;
        f32vec3 dist3 = abs(delta);
        if (CHUNK_LEVEL > 0) {
            needs_subdiv = needs_subdiv && (max(dist3.x, max(dist3.y, dist3.z)) < (31.0 + CHUNK_SIZE / 2) / VOXEL_SCL);
        }
    } else if ((work_item.brush_id & CHUNK_FLAGS_PARTICLE_BRUSH) != 0) {
        f32vec3 chunk_min = f32vec3(sub_node_chunk_i) * CHUNK_SIZE;
        f32vec3 chunk_max = f32vec3(sub_node_chunk_i) * CHUNK_SIZE + SUB_NODE_SIZE;
        f32vec3 place_bounds_min = f32vec3(deref(globals).voxel_particles_state.place_bounds_min);
        f32vec3 place_bounds_max = f32vec3(deref(globals).voxel_particles_state.place_bounds_max);
        if (CHUNK_LEVEL > 0) {
            needs_subdiv = needs_subdiv && rectangles_overlap(chunk_min, chunk_max, place_bounds_min, place_bounds_max);
        }
    }

    // If this thread needs subdivision, create a new child work item
    if (needs_subdiv) {
    // Create and try to queue a L1 work item
#if CHUNK_LEVEL == 0
        ChunkWorkItem new_work_item;
        new_work_item.i = sub_node_i;
        new_work_item.brush_id = work_item.brush_id;
        new_work_item.brush_input = work_item.brush_input;
        new_work_item.chunk_offset = work_item.chunk_offset;
        zero_work_item_children(new_work_item);
        thread_completed = queue_sub_work_item(new_work_item);
    // Create a L2 work item
#elif CHUNK_LEVEL == 1
        i32vec3 chunk_i = sub_node_i;
        // We must check that the chunk_i is within the loaded range. We can't elect chunks
        // that are outside the loaded range. If the chunk is outside the range, then we should
        // put these chunk updates into some fat stale queue or something.
        // if (chunk_i.x)
        u32 chunk_index = calc_chunk_index_from_worldspace(chunk_i, chunk_n);
        thread_completed = elect_chunk_for_update(chunk_i, work_item.chunk_offset, chunk_index, work_item.brush_id, work_item.brush_input);
#endif
    } else {
        // If there is no need for subdivision, no child work item is created
        thread_completed = true;
    }

    if (thread_completed) {
        // Instead of creating a child work item, we only increase the parent work item's bitmask
        atomicOr(work_item.children_finished[child_u32_index], 1u << child_bit_offset);
        atomicAdd(children_finished, 1); // And increment the number of completed children for this work group for this frame
    }
}
#undef CHUNKS
#undef MODEL

// For each parent work item L0 or L1, this shader is executed
// in 512 threads (1 work group) to generate up to 512 child work items L1 or L2

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    // First thread: Write the work item to shared memory 
    // All other threads: wait for it to be written
    get_work_item();
    
    // Get the child index in the work item's children_finished[16] u32 array
    child_u32_index = gl_LocalInvocationIndex / 32;
    // Get the child offset in the u32
    child_bit_offset = gl_LocalInvocationIndex - child_u32_index * 32;
    // Parent work item index (L0: (0,0,0), L1: (0,0,0)-(7,7,7))
    node_i = work_item.i;

    // Check if the current child work item is already completed
    u32 already_finished = (work_item.children_finished[child_u32_index] >> child_bit_offset) & 1;
    
    if (already_finished == 1) {
        // If the child work item is already completed, increment the 
        // total number of completed children for this work group for this frame
        atomicAdd(children_finished, 1);
    } else {
        // Otherwise perform the work item (conditional subdivision)
        perform_work_item();
    }

    complete_work_item();
}
