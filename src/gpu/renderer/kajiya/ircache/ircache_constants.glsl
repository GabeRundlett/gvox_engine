#pragma once

#define IRCACHE_USE_TRILINEAR 0
#define IRCACHE_USE_POSITION_VOTING 1
#define IRCACHE_USE_UNIFORM_VOTING 1
#define IRCACHE_FREEZE false

#define IRCACHE_USE_SPHERICAL_HARMONICS 1

const uint IRCACHE_ENTRY_META_OCCUPIED = 1u;
const uint IRCACHE_ENTRY_META_JUST_ALLOCATED = 2u;

#define IRCACHE_ENTRY_LIFE_RECYCLE 0x8000000u
#define IRCACHE_ENTRY_LIFE_RECYCLED (IRCACHE_ENTRY_LIFE_RECYCLE + 1u)

const uint IRCACHE_ENTRY_LIFE_PER_RANK = 4;
const uint IRCACHE_ENTRY_RANK_COUNT = 3;

bool is_ircache_entry_life_valid(uint life) {
    return life < IRCACHE_ENTRY_LIFE_PER_RANK * IRCACHE_ENTRY_RANK_COUNT;
}

uint ircache_entry_life_to_rank(uint life) {
    return life / IRCACHE_ENTRY_LIFE_PER_RANK;
}

uint ircache_entry_life_for_rank(uint rank) {
    return rank * IRCACHE_ENTRY_LIFE_PER_RANK;
}

const uint IRCACHE_OCTA_DIMS = 4;
const uint IRCACHE_OCTA_DIMS2 = IRCACHE_OCTA_DIMS * IRCACHE_OCTA_DIMS;
const uint IRCACHE_IRRADIANCE_STRIDE = 3;
const uint IRCACHE_AUX_STRIDE = 4 * IRCACHE_OCTA_DIMS2;

const uint IRCACHE_SAMPLES_PER_FRAME = 4;
const uint IRCACHE_VALIDATION_SAMPLES_PER_FRAME = 4;
const uint IRCACHE_RESTIR_M_CLAMP = 30;

#include <utils/normal.glsl>

Vertex unpack_vertex(VertexPacked p) {
    Vertex res;
    res.position = p.data0.xyz;
    res.normal = unpack_normal_11_10_11(p.data0.w);
    return res;
}

VertexPacked pack_vertex(Vertex v) {
    VertexPacked p;
    p.data0.xyz = v.position;
    p.data0.w = pack_normal_11_10_11(v.normal);
    return p;
}
