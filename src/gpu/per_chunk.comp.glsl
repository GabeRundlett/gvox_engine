#extension GL_EXT_debug_printf : enable

#include <shared/shared.inl>

#include <utils/math.glsl>
#include <utils/voxels.glsl>

i32vec3 chunk_n;
i32vec3 chunk_i;
u32 chunk_index;

#define SETTINGS deref(settings)
#define VOXEL_WORLD deref(globals).voxel_world
#define PLAYER deref(globals).player
#define CHUNKS(i) deref(voxel_chunks[i])
#define INDIRECT deref(globals).indirect_dispatch

bool try_elect(in out ChunkWorkItem work_item) {
    u32 prev_update_n = atomicAdd(VOXEL_WORLD.chunk_update_n, 1);

    // Check if the work item can be added
    if (prev_update_n < MAX_CHUNK_UPDATES_PER_FRAME) {
        // Set the chunk edit dispatch z axis (64/8, 64/8, 64 x 8 x 8 / 8 = 64 x 8) = (8, 8, 512)
        atomicAdd(INDIRECT.chunk_edit_dispatch.z, CHUNK_SIZE / 8);
        atomicAdd(INDIRECT.subchunk_x2x4_dispatch.z, 1);
        atomicAdd(INDIRECT.subchunk_x8up_dispatch.z, 1);
        u32 prev_flags = atomicOr(CHUNKS(chunk_index).flags, work_item.brush_id);
        // Set the chunk update infos
        VOXEL_WORLD.chunk_update_infos[prev_update_n].i = chunk_i;
        VOXEL_WORLD.chunk_update_infos[prev_update_n].brush_input = work_item.brush_input;
        VOXEL_WORLD.chunk_update_infos[prev_update_n].chunk_offset = work_item.chunk_offset;
    }
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    // (const) number of chunks in each axis
    chunk_n = i32vec3(1 << SETTINGS.log2_chunks_per_axis);
    chunk_i = i32vec3(gl_GlobalInvocationID.xyz);
    chunk_index = calc_chunk_index_from_worldspace(chunk_i, chunk_n);

    // Temporary.
    ChunkWorkItem terrain_work_item;
    terrain_work_item.i = i32vec3(0);
    terrain_work_item.chunk_offset = PLAYER.chunk_offset;
    terrain_work_item.brush_id = CHUNK_FLAGS_WORLD_BRUSH;

    if ((CHUNKS(chunk_index).flags & CHUNK_FLAGS_ACCEL_GENERATED) == 0) {
        try_elect(terrain_work_item);
    } else if (PLAYER.chunk_offset != PLAYER.prev_chunk_offset) {
        // invalidate chunks outside the chunk_offset
        i32vec3 diff = clamp(i32vec3(PLAYER.chunk_offset - PLAYER.prev_chunk_offset), -chunk_n, chunk_n);

        i32vec3 start;
        i32vec3 end;

        start.x = diff.x < 0 ? 0 : chunk_n.x - diff.x;
        end.x = diff.x < 0 ? -diff.x : chunk_n.x;

        start.y = diff.y < 0 ? 0 : chunk_n.y - diff.y;
        end.y = diff.y < 0 ? -diff.y : chunk_n.y;

        start.z = diff.z < 0 ? 0 : chunk_n.z - diff.z;
        end.z = diff.z < 0 ? -diff.z : chunk_n.z;

        u32vec3 temp_chunk_i = u32vec3((i32vec3(chunk_i) - deref(globals).player.chunk_offset) % i32vec3(chunk_n));

        if ((temp_chunk_i.x >= start.x && temp_chunk_i.x < end.x) ||
            (temp_chunk_i.y >= start.y && temp_chunk_i.y < end.y) ||
            (temp_chunk_i.z >= start.z && temp_chunk_i.z < end.z)) {
            CHUNKS(chunk_index).flags &= ~CHUNK_FLAGS_ACCEL_GENERATED;
            try_elect(terrain_work_item);
        }
    }
}

#undef INDIRECT
#undef CHUNKS
#undef PLAYER
#undef VOXEL_WORLD
#undef SETTINGS
