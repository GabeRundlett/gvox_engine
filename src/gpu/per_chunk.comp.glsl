#include <utils/voxel_world.glsl>

DAXA_USE_PUSH_CONSTANT(PerChunkComputePush)

#define VOXEL_WORLD deref(daxa_push_constant.gpu_globals).voxel_world
#define INDIRECT deref(daxa_push_constant.gpu_globals).indirect_dispatch
#define CHUNKS(i) deref(daxa_push_constant.voxel_chunks[i])
void elect_chunk_for_update(u32vec3 chunk_i, u32 chunk_index, u32 edit_stage) {
    u32 prev_update_n = atomicAdd(VOXEL_WORLD.chunk_update_n, 1);
    if (prev_update_n < MAX_CHUNK_UPDATES_PER_FRAME) {
        atomicAdd(INDIRECT.chunk_edit_dispatch.z, CHUNK_SIZE / 8);
        atomicAdd(INDIRECT.subchunk_x2x4_dispatch.z, 1);
        atomicAdd(INDIRECT.subchunk_x8up_dispatch.z, 1);
        CHUNKS(chunk_index).edit_stage = edit_stage;
        VOXEL_WORLD.chunk_update_infos[prev_update_n].i = chunk_i;
        VOXEL_WORLD.chunk_update_infos[prev_update_n].score = length(f32vec3(chunk_i));
    }
}
#undef CHUNKS
#undef INDIRECT
#undef VOXEL_WORLD

#define SETTINGS deref(daxa_push_constant.gpu_settings)
#define INPUT deref(daxa_push_constant.gpu_input)
#define GLOBALS deref(daxa_push_constant.gpu_globals)
#define VOXEL_WORLD deref(daxa_push_constant.gpu_globals).voxel_world
#define INDIRECT deref(daxa_push_constant.gpu_globals).indirect_dispatch
#define CHUNKS(i) deref(daxa_push_constant.voxel_chunks[i])
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    u32vec3 chunk_n;
    chunk_n.x = 1u << SETTINGS.log2_chunks_per_axis;
    chunk_n.y = chunk_n.x;
    chunk_n.z = chunk_n.x;

    u32vec3 chunk_i = gl_GlobalInvocationID.xyz;

    // Discard any invocations outside the real list of chunks
    if (chunk_i.x >= chunk_n.x ||
        chunk_i.y >= chunk_n.y ||
        chunk_i.z >= chunk_n.z)
        return;

    u32 chunk_index = calc_chunk_index(chunk_i, chunk_n);

    // Debug - reset all chunks to default
    // if (INPUT.actions[GAME_ACTION_INTERACT0] != 0) {
    //     CHUNKS(chunk_index).edit_stage = 0;
    //     for (u32 i = 0; i < PALETTES_PER_CHUNK; ++i) {
    //         CHUNKS(chunk_index).palette_headers[i].variant_n = 0;
    //         CHUNKS(chunk_index).palette_headers[i].blob_ptr = 0;
    //     }
    // }

    // Bump all previously handled chunks to the next update stage
    if (CHUNKS(chunk_index).edit_stage == 1 ||
        CHUNKS(chunk_index).edit_stage == 3) {
        CHUNKS(chunk_index).edit_stage = 2;
    }

    // Select "random" chunks to be updated
    if (CHUNKS(chunk_index).edit_stage == 0) {
        elect_chunk_for_update(chunk_i, chunk_index, 1);
    }

    // if (INPUT.actions[GAME_ACTION_BREAK] != 0) {
    //     if (CHUNKS(chunk_index).edit_stage > 1) {
    //         f32vec3 chunk_pos = (f32vec3(chunk_i) + 0.5) * CHUNK_SIZE / VOXEL_SCL;
    //         f32vec3 delta = chunk_pos - GLOBALS.pick_pos;
    //         f32vec3 dist3 = abs(delta);
    //         if (max(dist3.x, max(dist3.y, dist3.z)) < (31.0 + CHUNK_SIZE / 2) / VOXEL_SCL) {
    //             elect_chunk_for_update(chunk_i, chunk_index, 3);
    //         }
    //     }
    // }
}
#undef CHUNKS
#undef INDIRECT
#undef VOXEL_WORLD
#undef GLOBALS
#undef INPUT
#undef SETTINGS
