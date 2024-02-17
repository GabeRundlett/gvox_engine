#include <renderer/kajiya/ircache.inl>
#include "ircache_constants.glsl"
#include <utilities/gpu/random.glsl>

DAXA_DECL_PUSH_CONSTANT(AgeIrcacheEntriesComputePush, push)
daxa_RWBufferPtr(IrcacheMetadata) ircache_meta_buf = push.uses.ircache_meta_buf;
daxa_RWBufferPtr(IrcacheCell) ircache_grid_meta_buf = push.uses.ircache_grid_meta_buf;
daxa_RWBufferPtr(uint) ircache_entry_cell_buf = push.uses.ircache_entry_cell_buf;
daxa_RWBufferPtr(uint) ircache_life_buf = push.uses.ircache_life_buf;
daxa_RWBufferPtr(uint) ircache_pool_buf = push.uses.ircache_pool_buf;
daxa_RWBufferPtr(VertexPacked) ircache_spatial_buf = push.uses.ircache_spatial_buf;
daxa_RWBufferPtr(VertexPacked) ircache_reposition_proposal_buf = push.uses.ircache_reposition_proposal_buf;
daxa_RWBufferPtr(uint) ircache_reposition_proposal_count_buf = push.uses.ircache_reposition_proposal_count_buf;
daxa_RWBufferPtr(vec4) ircache_irradiance_buf = push.uses.ircache_irradiance_buf;
daxa_RWBufferPtr(uint) entry_occupancy_buf = push.uses.entry_occupancy_buf;

void age_ircache_entry(uint entry_idx) {
    const uint prev_age = deref(ircache_life_buf[entry_idx]);
    const uint new_age = prev_age + 1;

    if (is_ircache_entry_life_valid(new_age)) {
        deref(ircache_life_buf[entry_idx]) = new_age;

        // TODO: just `Store` it (AMD doesn't like it unless it's a byte address buffer)
        const uint cell_idx = deref(ircache_entry_cell_buf[entry_idx]);
        atomicAnd(deref(ircache_grid_meta_buf[cell_idx]).flags, ~IRCACHE_ENTRY_META_JUST_ALLOCATED);
    } else {
        deref(ircache_life_buf[entry_idx]) = IRCACHE_ENTRY_LIFE_RECYCLED;
        // onoz, we killed it!
        // deallocate.

        for (uint i = 0; i < IRCACHE_IRRADIANCE_STRIDE; ++i) {
            deref(ircache_irradiance_buf[entry_idx * IRCACHE_IRRADIANCE_STRIDE + i]) = 0.0.xxxx;
        }

        uint entry_alloc_count = atomicAdd(deref(ircache_meta_buf).alloc_count, -1);
        deref(ircache_pool_buf[entry_alloc_count - 1]) = entry_idx;

        // TODO: just `Store` it (AMD doesn't like it unless it's a byte address buffer)
        const uint cell_idx = deref(ircache_entry_cell_buf[entry_idx]);
        atomicAnd(deref(ircache_grid_meta_buf[cell_idx]).flags,
                  ~(IRCACHE_ENTRY_META_OCCUPIED | IRCACHE_ENTRY_META_JUST_ALLOCATED));
    }
}

bool ircache_entry_life_needs_aging(uint life) {
    return life != IRCACHE_ENTRY_LIFE_RECYCLED;
}

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint entry_idx = gl_GlobalInvocationID.x;
    const uint total_entry_count = deref(ircache_meta_buf).entry_count;

    if (!IRCACHE_FREEZE) {
        if (entry_idx < total_entry_count) {
            const uint life = deref(ircache_life_buf[entry_idx]);

            if (ircache_entry_life_needs_aging(life)) {
                age_ircache_entry(entry_idx);
            }

#if IRCACHE_USE_POSITION_VOTING
#if 0
                    uint rng = hash2(uvec2(entry_idx, frame_constants.frame_index));
                    const float dart = uint_to_u01_float(hash1_mut(rng));
                    const float prob = 0.02;

                    if (dart <= prob)
#endif
            {

                // Flush the reposition proposal
                VertexPacked proposal = deref(ircache_reposition_proposal_buf[entry_idx]);
                deref(ircache_spatial_buf[entry_idx]) = proposal;
            }
#endif

            deref(ircache_reposition_proposal_count_buf[entry_idx]) = 0;
        } else {
            VertexPacked invalid;
            invalid.data0 = vec4(uintBitsToFloat(0));
            deref(ircache_spatial_buf[entry_idx]) = invalid;
        }
    }

    const uint life = deref(ircache_life_buf[entry_idx]);
    uint valid = uint(entry_idx < total_entry_count && is_ircache_entry_life_valid(life));
    deref(entry_occupancy_buf[entry_idx]) = valid;
}
