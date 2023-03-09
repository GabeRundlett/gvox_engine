#include <utils/voxel_world.glsl>

DAXA_USE_PUSH_CONSTANT(PerChunkComputePush)

#define SETTINGS deref(daxa_push_constant.gpu_settings)
#define INPUT deref(daxa_push_constant.gpu_input)
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
    if (INPUT.actions[GAME_ACTION_INTERACT0] != 0) {
        CHUNKS(chunk_index).edit_stage = 0;
    }

    // Bump all previously handled chunks to the next update stage
    if (CHUNKS(chunk_index).edit_stage == 1) {
        CHUNKS(chunk_index).edit_stage = 2;
    }

    // Select "random" chunks to be updated
    if (CHUNKS(chunk_index).edit_stage == 0) {
        u32 prev_update_n = atomicAdd(VOXEL_WORLD.chunk_update_n, 1);
        if (prev_update_n < MAX_CHUNK_UPDATES) {
            atomicAdd(INDIRECT.chunk_edit_dispatch.z, CHUNK_SIZE / 8);
            atomicAdd(INDIRECT.subchunk_x2x4_dispatch.z, 1);
            atomicAdd(INDIRECT.subchunk_x8up_dispatch.z, 1);
            CHUNKS(chunk_index).edit_stage = 1;
            VOXEL_WORLD.chunk_update_is[prev_update_n] = chunk_i;
        }
    }
}
#undef CHUNKS
#undef INDIRECT
#undef VOXEL_WORLD
#undef INPUT
#undef SETTINGS
