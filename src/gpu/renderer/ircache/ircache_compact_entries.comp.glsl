#include <shared/app.inl>
#include "ircache_constants.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint entry_idx = gl_GlobalInvocationID.x;
    const uint total_entry_count = deref(ircache_meta_buf).entry_count;

    const uint life = deref(ircache_life_buf[entry_idx]);
    if (entry_idx < total_entry_count && is_ircache_entry_life_valid(life)) {
        deref(ircache_entry_indirection_buf[deref(entry_occupancy_buf[entry_idx])]) = entry_idx;
    }
}
