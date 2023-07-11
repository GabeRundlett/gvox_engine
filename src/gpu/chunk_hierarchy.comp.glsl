#include <shared/shared.inl>

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
u32vec3 node_i;
u32 work_item_index;
u32 child_u32_index;
u32 child_bit_offset;

void get_work_item() {
    work_item_index = gl_WorkGroupID.x;
    if (gl_LocalInvocationIndex == 0) {
        children_finished = 0;
        // TODO: check if I can load this with multiple threads
        work_item = WORK_ITEMS[work_item_index];
    }
    barrier();
    memoryBarrierShared();
}

#if CHUNK_LEVEL < 1
bool queue_sub_work_item(in ChunkWorkItem new_work_item) {
    u32 queue_offset = atomicAdd(SUB_WORK_ITEMS_QUEUED, 1);
    atomicMin(SUB_WORK_ITEMS_QUEUED, MAX_SUB_WORK_ITEMS);
    if (queue_offset < MAX_SUB_WORK_ITEMS) {
        u32 index = queue_offset;
        SUB_WORK_ITEMS[index] = new_work_item;
        return true;
    } else {
        return false;
    }
}
#endif

void complete_work_item() {
    barrier();
    memoryBarrierShared();
    if (gl_LocalInvocationIndex < 16) {
        WORK_ITEMS[work_item_index].children_finished[gl_LocalInvocationIndex] = work_item.children_finished[gl_LocalInvocationIndex];
    }
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
bool elect_chunk_for_update(u32vec3 chunk_i, u32 chunk_index, u32 brush_flags, in out BrushInput brush_input) {
    u32 prev_update_n = atomicAdd(VOXEL_WORLD.chunk_update_n, 1);
    if (prev_update_n < MAX_CHUNK_UPDATES_PER_FRAME) {
        atomicAdd(INDIRECT.chunk_edit_dispatch.z, CHUNK_SIZE / 8);
        atomicAdd(INDIRECT.subchunk_x2x4_dispatch.z, 1);
        atomicAdd(INDIRECT.subchunk_x8up_dispatch.z, 1);
        u32 prev_flags = atomicOr(CHUNKS(chunk_index).flags, brush_flags);
        VOXEL_WORLD.chunk_update_infos[prev_update_n].i = chunk_i;
        VOXEL_WORLD.chunk_update_infos[prev_update_n].brush_input = brush_input;
        if ((prev_flags & CHUNK_FLAGS_BRUSH_MASK) == 0) {
            VOXEL_WORLD.chunk_update_infos[prev_update_n].flags = 1;
        } else {
            VOXEL_WORLD.chunk_update_infos[prev_update_n].flags = 0;
        }
        return true;
    }
    return false;
}
#undef CHUNKS
#undef INDIRECT
#undef VOXEL_WORLD

#define SETTINGS deref(settings)
#define MODEL deref(gvox_model)
void perform_work_item() {
    u32vec3 chunk_n = u32vec3(1u << SETTINGS.log2_chunks_per_axis);

    u32vec3 sub_node_i = node_i * 8 + gl_LocalInvocationID.xyz;
    u32vec3 sub_node_chunk_i = sub_node_i * SUB_NODE_SIZE / CHUNK_SIZE;
    bool thread_completed = false;

    // check if sub-node update is necessary
    bool needs_subdiv =
        (sub_node_chunk_i.x < chunk_n.x) &&
        (sub_node_chunk_i.y < chunk_n.y) &&
        (sub_node_chunk_i.z < chunk_n.z);

    if ((work_item.brush_id & CHUNK_FLAGS_WORLD_BRUSH) != 0) {
#if GEN_MODEL
        u32vec3 chunk_bi = sub_node_chunk_i + 0;
        needs_subdiv = needs_subdiv && (chunk_bi.x <= (MODEL.extent_x >> 6) && chunk_bi.y <= (MODEL.extent_y >> 6) && chunk_bi.z <= (MODEL.extent_z >> 6));
#endif
    } else if ((work_item.brush_id & CHUNK_FLAGS_USER_BRUSH_A) != 0) {
        f32vec3 chunk_pos = (f32vec3(sub_node_chunk_i) + 0.5) * CHUNK_SIZE / VOXEL_SCL;
        f32vec3 delta = chunk_pos - deref(globals).brush_input.pos;
        f32vec3 dist3 = abs(delta);
        if (CHUNK_LEVEL > 0) {
            needs_subdiv = needs_subdiv && (max(dist3.x, max(dist3.y, dist3.z)) < (31.0 + CHUNK_SIZE / 2) / VOXEL_SCL);
        }
    } else if ((work_item.brush_id & CHUNK_FLAGS_USER_BRUSH_B) != 0) {
        f32vec3 chunk_pos = (f32vec3(sub_node_chunk_i) + 0.5) * CHUNK_SIZE / VOXEL_SCL;
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

    if (needs_subdiv) {
#if CHUNK_LEVEL == 0
        ChunkWorkItem new_work_item;
        new_work_item.i = sub_node_i;
        new_work_item.brush_id = work_item.brush_id;
        new_work_item.brush_input = work_item.brush_input;
        zero_work_item_children(new_work_item);
        thread_completed = queue_sub_work_item(new_work_item);
#elif CHUNK_LEVEL == 1
        u32vec3 chunk_i = sub_node_i;
        u32 chunk_index = calc_chunk_index(chunk_i, chunk_n);
        thread_completed = elect_chunk_for_update(chunk_i, chunk_index, work_item.brush_id, work_item.brush_input);
#endif
    } else {
        thread_completed = true;
    }

    if (thread_completed) {
        atomicOr(work_item.children_finished[child_u32_index], 1u << child_bit_offset);
        atomicAdd(children_finished, 1);
    }
}
#undef MODEL
#undef SETTINGS

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    get_work_item();

    child_u32_index = gl_LocalInvocationIndex / 32;
    child_bit_offset = gl_LocalInvocationIndex - child_u32_index * 32;

    node_i = work_item.i;

    u32 already_finished = (work_item.children_finished[child_u32_index] >> child_bit_offset) & 1;

    if (already_finished == 1) {
        atomicAdd(children_finished, 1);
    } else {
        perform_work_item();
    }

    complete_work_item();
}
