#include <shared/app.inl>

#include <utils/random.glsl>
#include "ircache_constants.glsl"
#include "ircache_grid.glsl"

void deallocate_cell(uint cell_idx) {
    const uvec2 meta = deref(ircache_grid_meta_buf[cell_idx]);

    if ((meta.y & IRCACHE_ENTRY_META_OCCUPIED) != 0) {
        // Clear the just-nuked entry
        const uint entry_idx = meta.x;

        deref(ircache_life_buf[entry_idx]) = IRCACHE_ENTRY_LIFE_RECYCLED;

        for (uint i = 0; i < IRCACHE_IRRADIANCE_STRIDE; ++i) {
            deref(ircache_irradiance_buf[entry_idx * IRCACHE_IRRADIANCE_STRIDE + i]) = 0.0.xxxx;
        }

        uint entry_alloc_count = atomicAdd(deref(ircache_meta_buf[IRCACHE_META_ALLOC_COUNT_INDEX]), -1);
        deref(ircache_pool_buf[entry_alloc_count - 1]) = entry_idx;
    }
}

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
void main() {
    uvec3 dispatch_thread_id = gl_GlobalInvocationID.xyz;
    const uvec3 dst_vx = uvec3(dispatch_thread_id.xy, dispatch_thread_id.z % IRCACHE_CASCADE_SIZE);
    const uint cascade = dispatch_thread_id.z / IRCACHE_CASCADE_SIZE;

    uint dst_cell_idx = cell_idx(IrcacheCoord_from_coord_cascade(dst_vx, cascade));

    ivec3 scroll_by =
        select(IRCACHE_FREEZE, (0).xxx, deref(gpu_input).ircache_cascades[cascade].voxels_scrolled_this_frame.xyz);

    if (!all(lessThan(uvec3(dst_vx - scroll_by), uvec3(IRCACHE_CASCADE_SIZE)))) {
        // If this entry is about to get overwritten, deallocate it.
        deallocate_cell(dst_cell_idx);
    }

    const uvec3 src_vx = dst_vx + scroll_by;

    if (all(lessThan(src_vx, uvec3(IRCACHE_CASCADE_SIZE)))) {
        const uint src_cell_idx = cell_idx(IrcacheCoord_from_coord_cascade(src_vx, cascade));

        const uvec2 cell_meta = deref(ircache_grid_meta_buf[src_cell_idx]);
        deref(ircache_grid_meta_buf2[dst_cell_idx]) = cell_meta;

        // Update the cell idx in the `ircache_entry_cell_buf`
        if ((cell_meta.y & IRCACHE_ENTRY_META_OCCUPIED) != 0) {
            const uint entry_idx = cell_meta.x;
            deref(ircache_entry_cell_buf[entry_idx]) = dst_cell_idx;
        }
    } else {
        deref(ircache_grid_meta_buf2[dst_cell_idx]) = (0).xx;
    }
}
