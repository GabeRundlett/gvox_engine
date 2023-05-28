#extension GL_EXT_shader_atomic_int64 : require

#include <shared/shared.inl>

#include <utils/voxels.glsl>

#define THREAD_POOL deref(globals).chunk_thread_pool_state

// clang-format off
#if CHUNK_LEVEL == 0
#define WORK_ITEMS            THREAD_POOL.chunk_work_items_l0
#define WORK_ITEMS_BEGIN      THREAD_POOL.work_items_l0_begin
#define WORK_ITEMS_COMPLETED  THREAD_POOL.work_items_l0_completed
#define SUB_WORK_ITEMS        THREAD_POOL.chunk_work_items_l1
#define SUB_WORK_ITEMS_BEGIN  THREAD_POOL.work_items_l1_begin
#define SUB_WORK_ITEMS_QUEUED THREAD_POOL.work_items_l1_queued
#define MAX_SUB_WORK_ITEMS    MAX_CHUNK_WORK_ITEMS_L1
#elif CHUNK_LEVEL == 1
#define WORK_ITEMS            THREAD_POOL.chunk_work_items_l1
#define WORK_ITEMS_BEGIN      THREAD_POOL.work_items_l1_begin
#define WORK_ITEMS_COMPLETED  THREAD_POOL.work_items_l1_completed
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
    work_item_index = WORK_ITEMS_BEGIN + gl_WorkGroupID.x;
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
        u32 index = (SUB_WORK_ITEMS_BEGIN + queue_offset) % MAX_SUB_WORK_ITEMS;
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
        }
    }
}

#define VOXEL_WORLD deref(globals).voxel_world
#define INDIRECT deref(globals).indirect_dispatch
#define CHUNKS(i) deref(voxel_chunks[i])
bool elect_chunk_for_update(u32vec3 chunk_i, u32 chunk_index, u32 edit_stage) {
    u32 prev_update_n = atomicAdd(VOXEL_WORLD.chunk_update_n, 1);
    if (prev_update_n < MAX_CHUNK_UPDATES_PER_FRAME) {
        atomicAdd(INDIRECT.chunk_edit_dispatch.z, CHUNK_SIZE / 8);
        atomicAdd(INDIRECT.subchunk_x2x4_dispatch.z, 1);
        atomicAdd(INDIRECT.subchunk_x8up_dispatch.z, 1);
        // TODO: RACE CONDITION
        CHUNKS(chunk_index).edit_stage = edit_stage;
        VOXEL_WORLD.chunk_update_infos[prev_update_n].i = chunk_i;
        return true;
    }
    return false;
}
#undef CHUNKS
#undef INDIRECT
#undef VOXEL_WORLD

#define SETTINGS deref(settings)
void perform_work_item() {
    u32vec3 sub_node_i = node_i * 8 + gl_GlobalInvocationID.xyz;
    bool thread_completed = false;

    bool needs_subdiv = true;

#if CHUNK_LEVEL == 0
    // check if sub-node update is necessary
    if (needs_subdiv) {
        ChunkWorkItem new_work_item;
        new_work_item.packed_coordinate = ((sub_node_i.x << 0x00) & 0x3ff) + ((sub_node_i.y << 0x0a) & 0x3ff) + ((sub_node_i.z << 0x14) & 0x3ff);
        new_work_item.brush_id = work_item.brush_id;
        zero_work_item_children(new_work_item);
        thread_completed = queue_sub_work_item(new_work_item);
    } else {
        thread_completed = true;
    }
#elif CHUNK_LEVEL == 1
    if (needs_subdiv) {
        u32vec3 chunk_i = sub_node_i;
        u32vec3 chunk_n = u32vec3(1u << SETTINGS.log2_chunks_per_axis);
        u32 chunk_index = calc_chunk_index(chunk_i, chunk_n);
        thread_completed = elect_chunk_for_update(chunk_i, chunk_index, CHUNK_STAGE_WORLD_BRUSH + work_item.brush_id);
    } else {
        thread_completed = true;
    }
#endif

    if (thread_completed) {
        atomicOr(work_item.children_finished[child_u32_index], 1 << child_bit_offset);
        atomicAdd(children_finished, 1);
    }
}
#undef SETTINGS

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    get_work_item();

    child_u32_index = gl_LocalInvocationIndex / 32;
    child_bit_offset = child_bit_offset = gl_LocalInvocationIndex - child_u32_index;

    node_i = u32vec3(
        (work_item.packed_coordinate >> 0x00) & 0x3ff,
        (work_item.packed_coordinate >> 0x0a) & 0x3ff,
        (work_item.packed_coordinate >> 0x14) & 0x3ff);

    u32 already_finished = (work_item.children_finished[child_u32_index] >> child_bit_offset) & 1;

    if (already_finished == 1) {
        atomicAdd(children_finished, 1);
    } else {
        perform_work_item();
    }

    complete_work_item();
}
