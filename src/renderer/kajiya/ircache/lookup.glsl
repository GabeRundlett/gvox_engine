#pragma once

#include "../inc/quasi_random.glsl"
#include "../inc/sh.glsl"
#include <renderer/kajiya/inc/camera.glsl>

#include "ircache_grid.glsl"
#include "ircache_sampler_common.inc.glsl"

#define IRCACHE_LOOKUP_MAX 1

struct IrcacheLookup {
    uint entry_idx[IRCACHE_LOOKUP_MAX];
    float weight[IRCACHE_LOOKUP_MAX];
    uint count;
};

IrcacheLookup ircache_lookup(vec3 pt_ws, vec3 normal_ws, vec3 jitter) {
    IrcacheLookup result;
    result.count = 0;

    const IrcacheCoord rcoord = ws_pos_to_ircache_coord(globals, gpu_input, pt_ws, normal_ws, jitter);
    const uint cell_idx = cell_idx(rcoord);

    const IrcacheCell cell = deref(advance(ircache_grid_meta_buf, cell_idx));

    if ((cell.flags & IRCACHE_ENTRY_META_OCCUPIED) != 0) {
        const uint entry_idx = cell.entry_index;
        result.entry_idx[result.count] = entry_idx;
        result.weight[result.count] = 1;
        result.count = 1;
    }

    return result;
}

struct IrcacheLookupMaybeAllocate {
    IrcacheLookup lookup;
    Vertex proposal;
    bool just_allocated;
};

struct IrcacheLookupParams {
    vec3 query_from_ws;
    vec3 pt_ws;
    vec3 normal_ws;
    uint query_rank;
    bool stochastic_interpolation;
};

IrcacheLookupParams IrcacheLookupParams_create(vec3 query_from_ws, vec3 pt_ws, vec3 normal_ws) {
    IrcacheLookupParams res;
    res.query_from_ws = query_from_ws;
    res.pt_ws = pt_ws;
    res.normal_ws = normal_ws;
    res.query_rank = 0;
    res.stochastic_interpolation = false;
    return res;
}

IrcacheLookupParams with_query_rank(inout IrcacheLookupParams self, uint v) {
    IrcacheLookupParams res = self;
    res.query_rank = v;
    return res;
}

IrcacheLookupParams with_stochastic_interpolation(inout IrcacheLookupParams self, bool v) {
    IrcacheLookupParams res = self;
    res.stochastic_interpolation = v;
    return res;
}

IrcacheLookupMaybeAllocate lookup_maybe_allocate(
    inout IrcacheLookupParams self,
    inout uint rng) {
    bool allocated_by_us = false;
    bool just_allocated = false;

    const vec3 jitter = select(bvec3(self.stochastic_interpolation), (vec3(uint_to_u01_float(hash1_mut(rng)), uint_to_u01_float(hash1_mut(rng)), uint_to_u01_float(hash1_mut(rng))) - 0.5), 0.0.xxx);

#ifndef IRCACHE_LOOKUP_DONT_KEEP_ALIVE
    if (!IRCACHE_FREEZE) {
        const vec3 eye_pos = get_eye_position(gpu_input) + deref(gpu_input).player.player_unit_offset;

        const IrcacheCoord rcoord = ws_pos_to_ircache_coord(globals, gpu_input, self.pt_ws, self.normal_ws, jitter);

        const ivec3 scroll_offset = deref(gpu_input).ircache_cascades[rcoord.cascade].voxels_scrolled_this_frame.xyz;
        const bvec3 was_just_scrolled_in =
            select(greaterThan(scroll_offset, ivec3(0)), greaterThanEqual(ivec3(rcoord.coord) + scroll_offset, ivec3(IRCACHE_CASCADE_SIZE)), lessThan(ivec3(rcoord.coord), -scroll_offset));

        // When a voxel is just scrolled in to a cascade, allocating it via indirect rays
        // has a good chance of creating leaks. Delay the allocation for one frame
        // unless we have a suitable one from a primary ray.
        const bool skip_allocation =
            self.query_rank >= IRCACHE_ENTRY_RANK_COUNT || (any(was_just_scrolled_in) && self.query_rank > 0);

        const uint cell_idx = cell_idx(rcoord);
        const IrcacheCell cell = deref(advance(ircache_grid_meta_buf, cell_idx));
        const uint entry_flags = cell.flags;

        just_allocated = (entry_flags & IRCACHE_ENTRY_META_JUST_ALLOCATED) != 0;

        if (!skip_allocation) {
            if ((entry_flags & IRCACHE_ENTRY_META_OCCUPIED) == 0) {
                // Allocate

                uint prev = atomicOr(deref(advance(ircache_grid_meta_buf, cell_idx)).flags, IRCACHE_ENTRY_META_OCCUPIED | IRCACHE_ENTRY_META_JUST_ALLOCATED);
                // ircache_grid_meta_buf.InterlockedOr(
                //     sizeof(uvec2) * cell_idx + sizeof(uint),
                //     IRCACHE_ENTRY_META_OCCUPIED | IRCACHE_ENTRY_META_JUST_ALLOCATED,
                //     prev);

                if ((prev & IRCACHE_ENTRY_META_OCCUPIED) == 0) {
                    // We've allocated it!
                    just_allocated = true;
                    allocated_by_us = true;

                    uint alloc_idx = atomicAdd(deref(ircache_meta_buf).alloc_count, 1);

                    // Ref: 2af64eb1-745a-4778-8c80-04af6e2225e0
                    if (alloc_idx >= 1024 * 64) {
                        atomicAdd(deref(ircache_meta_buf).alloc_count, -1);
                        atomicAnd(deref(advance(ircache_grid_meta_buf, cell_idx)).flags, ~(IRCACHE_ENTRY_META_OCCUPIED | IRCACHE_ENTRY_META_JUST_ALLOCATED));
                    } else {
                        uint entry_idx = deref(advance(ircache_pool_buf, alloc_idx));
                        atomicMax(deref(ircache_meta_buf).entry_count, entry_idx + 1);

                        // Clear dead state, mark used.

                        deref(advance(ircache_life_buf, entry_idx)) = ircache_entry_life_for_rank(self.query_rank);
                        deref(advance(ircache_entry_cell_buf, entry_idx)) = cell_idx;
                        deref(advance(ircache_grid_meta_buf, cell_idx)).entry_index = entry_idx;
                    }
                }
            }
        }
    }
#endif

    IrcacheLookup lookup = ircache_lookup(self.pt_ws, self.normal_ws, jitter);

    const uint cascade = ws_pos_to_ircache_coord(globals, gpu_input, self.pt_ws, self.normal_ws, jitter).cascade;
    const float cell_diameter = ircache_grid_cell_diameter_in_cascade(cascade);

    vec3 to_eye = normalize(get_eye_position(gpu_input) + deref(gpu_input).player.player_unit_offset - self.pt_ws.xyz);
    vec3 offset_towards_query = self.query_from_ws - self.pt_ws.xyz;
    const float MAX_OFFSET = cell_diameter; // world units
    const float MAX_OFFSET_AS_FRAC = 0.5;   // fraction of the distance from query point
    offset_towards_query *= MAX_OFFSET / max(MAX_OFFSET / MAX_OFFSET_AS_FRAC, length(offset_towards_query));

    Vertex new_entry;
#if IRCACHE_USE_SPHERICAL_HARMONICS
    // probes
    new_entry.position = self.pt_ws.xyz + offset_towards_query;
#else
    // surface points
    new_entry.position = self.pt_ws.xyz;
#endif
    new_entry.normal = self.normal_ws;
    // new_entry.normal = to_eye;

    if (allocated_by_us) {
        for (uint i = 0; i < IRCACHE_LOOKUP_MAX; ++i)
            if (i < lookup.count) {
                const uint entry_idx = lookup.entry_idx[i];
                deref(advance(ircache_reposition_proposal_buf, entry_idx)) = pack_vertex(new_entry);
            }
    }

    IrcacheLookupMaybeAllocate res;
    res.lookup = lookup;
    res.just_allocated = just_allocated;
    res.proposal = new_entry;
    return res;
}

float eval_sh_simplified(vec4 sh, vec3 normal) {
    vec4 lobe_sh = vec4(0.8862, 1.0233 * normal);
    return dot(sh, lobe_sh);
}

float eval_sh_geometrics(vec4 sh, vec3 normal) {
    // http://www.geomerics.com/wp-content/uploads/2015/08/CEDEC_Geomerics_ReconstructingDiffuseLighting1.pdf

    float R0 = sh.x;

    vec3 R1 = 0.5f * vec3(sh.y, sh.z, sh.w);
    float lenR1 = length(R1);

    float q = 0.5f * (1.0f + dot(R1 / lenR1, normal));

    float p = 1.0f + 2.0f * lenR1 / R0;
    float a = (1.0f - lenR1 / R0) / (1.0f + lenR1 / R0);

    return R0 * (a + (1.0f - a) * (p + 1.0f) * pow(q, p));
}

float eval_sh_nope(vec4 sh, vec3 normal) {
    return sh.x / (0.282095 * 4);
}

#if IRCACHE_USE_SPHERICAL_HARMONICS
#if 0
#define eval_sh eval_sh_simplified
#else
#define eval_sh eval_sh_geometrics
#endif
#else
#define eval_sh eval_sh_nope
#endif

vec3 lookup(IrcacheLookupParams self, inout uint rng) {
    IrcacheLookupMaybeAllocate lookup = lookup_maybe_allocate(self, rng);

    if (lookup.just_allocated) {
        return 0.0.xxx;
    }

    vec3 irradiance_sum = 0.0.xxx;

#ifdef IRCACHE_LOOKUP_KEEP_ALIVE_PROB
    const bool should_propose_position = uint_to_u01_float(hash1_mut(rng)) < IRCACHE_LOOKUP_KEEP_ALIVE_PROB;
#else
    const bool should_propose_position = true;
#endif

    for (uint i = 0; i < IRCACHE_LOOKUP_MAX; ++i)
        if (i < lookup.lookup.count) {
            const uint entry_idx = lookup.lookup.entry_idx[i];

            vec3 irradiance = vec3(0);

#ifdef IRCACHE_LOOKUP_PRECISE
            {
                float weight_sum = 0;

                // TODO: counter distortion
                for (uint octa_idx = 0; octa_idx < IRCACHE_OCTA_DIMS2; ++octa_idx) {
                    const vec2 octa_coord = (vec2(octa_idx % IRCACHE_OCTA_DIMS, octa_idx / IRCACHE_OCTA_DIMS) + 0.5) / IRCACHE_OCTA_DIMS;

                    const Reservoir1spp r = Reservoir1spp_from_raw(floatBitsToUint(deref(advance(ircache_aux_buf, entry_idx * IRCACHE_AUX_STRIDE + octa_idx)).xy));
                    const vec3 dir = direction(SampleParams_from_raw(r.payload));

                    const float wt = dot(dir, self.normal_ws);
                    if (wt > 0.0) {
                        const vec4 contrib = deref(advance(ircache_aux_buf, entry_idx * IRCACHE_AUX_STRIDE + IRCACHE_OCTA_DIMS2 + octa_idx));
                        irradiance += contrib.rgb * wt * contrib.w;
                        weight_sum += wt;
                    }
                }

                irradiance /= max(1.0, weight_sum);
            }
#else
            for (uint basis_i = 0; basis_i < 3; ++basis_i) {
                irradiance[basis_i] += eval_sh(deref(advance(ircache_irradiance_buf, entry_idx * IRCACHE_IRRADIANCE_STRIDE + basis_i)), self.normal_ws);
            }
#endif

            irradiance = max(0.0.xxx, irradiance);
            irradiance_sum += irradiance * lookup.lookup.weight[i];

            if (!IRCACHE_FREEZE && should_propose_position) {
#ifndef IRCACHE_LOOKUP_DONT_KEEP_ALIVE
                const uint prev_life = deref(advance(ircache_life_buf, entry_idx));

                if (prev_life < IRCACHE_ENTRY_LIFE_RECYCLE) {
                    const uint new_life = ircache_entry_life_for_rank(self.query_rank);
                    if (new_life < prev_life) {
                        atomicMin(deref(advance(ircache_life_buf, entry_idx)), new_life);
                        // ircache_life_buf.Store(entry_idx * 4, new_life);
                    }

                    const uint prev_rank = ircache_entry_life_to_rank(prev_life);
                    if (self.query_rank <= prev_rank) {
                        uint prev_vote_count = atomicAdd(deref(advance(ircache_reposition_proposal_count_buf, entry_idx)), 1);

                        const float dart = uint_to_u01_float(hash1_mut(rng));
                        const float prob = 1.0 / (prev_vote_count + 1.0);

                        if (IRCACHE_USE_UNIFORM_VOTING == 0 || dart <= prob) {
                            deref(advance(ircache_reposition_proposal_buf, entry_idx)) = pack_vertex(lookup.proposal);
                        }
                    }
                }
#endif
            }

            // irradiance_sum = vec3(entry_idx % 64) / 63.0;
        }

    return irradiance_sum;
}
