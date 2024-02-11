#include <utilities/gpu/math.glsl>
#include "ircache_constants.glsl"

#define IRCACHE_USE_NORMAL_BASED_CELL_OFFSET 1

struct IrcacheCoord {
    uvec3 coord;
    uint cascade;
};

IrcacheCoord IrcacheCoord_from_coord_cascade(uvec3 coord, uint cascade) {
    IrcacheCoord res;
    res.coord = min(coord, (IRCACHE_CASCADE_SIZE - 1).xxx);
    res.cascade = min(cascade, IRCACHE_CASCADE_COUNT - 1);
    return res;
}

uint udot(uvec4 a, uvec4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

uint cell_idx(IrcacheCoord self) {
    return udot(
        uvec4(self.coord, self.cascade),
        uvec4(
            1,
            IRCACHE_CASCADE_SIZE,
            IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_SIZE,
            IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_SIZE));
}

uint ws_local_pos_to_cascade_idx(vec3 local_pos, uint reserved_cells) {
    const vec3 fcoord = local_pos / IRCACHE_GRID_CELL_DIAMETER;
    const float max_coord = max(abs(fcoord.x), max(abs(fcoord.y), abs(fcoord.z)));
    const float cascade_float = log2(max_coord / (IRCACHE_CASCADE_SIZE / 2 - reserved_cells));
    return uint(clamp(ceil(max(0.0, cascade_float)), 0, IRCACHE_CASCADE_COUNT - 1));
}

IrcacheCoord ws_pos_to_ircache_coord(daxa_RWBufferPtr(GpuGlobals) globals, daxa_BufferPtr(GpuInput) gpu_input, vec3 pos, vec3 normal, vec3 jitter) {
    vec3 center = deref(gpu_input).ircache_grid_center.xyz;
    pos += deref(globals).player.player_unit_offset;

    uint reserved_cells =
        select(IRCACHE_USE_NORMAL_BASED_CELL_OFFSET != 0,
               // Make sure we can actually offset towards the edge of a cascade.
               1,
               // Business as usual
               0);

    vec3 cell_offset = vec3(0);

    // Stochastic interpolation (no-op if jitter is zero)
    {
        const uint cascade = ws_local_pos_to_cascade_idx(pos - center, reserved_cells);
        const float cell_diameter = (IRCACHE_GRID_CELL_DIAMETER * (1u << cascade));
        pos += cell_diameter * jitter;
    }

    const uint cascade = ws_local_pos_to_cascade_idx(pos - center, reserved_cells);
    const float cell_diameter = (IRCACHE_GRID_CELL_DIAMETER * (1u << cascade));

    const ivec3 cascade_origin = deref(gpu_input).ircache_cascades[cascade].origin.xyz;

    // NOTE(grundlett): offset with 0.413 as a random number as opposed to 0.5.
    // With the cells lining up with voxels, 0.5 often results in samples being
    // right on the edge between two cells, so we get bad artifacts.
    cell_offset +=
        select(bvec3(IRCACHE_USE_NORMAL_BASED_CELL_OFFSET), normal * cell_diameter * 0.413, 0.0.xxx);

    ivec3 coord = ivec3(floor((pos + cell_offset) / cell_diameter)) - cascade_origin;

    IrcacheCoord res;
    res.cascade = cascade;
    res.coord = uvec3(clamp(coord, (0).xxx, (IRCACHE_CASCADE_SIZE - 1).xxx));
    return res;
}

float ircache_grid_cell_diameter_in_cascade(uint cascade) {
    return IRCACHE_GRID_CELL_DIAMETER * (1u << uint(cascade));
}
