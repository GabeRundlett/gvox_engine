#include <renderer/kajiya/ircache.inl>
#include "ircache_constants.glsl"

DAXA_DECL_PUSH_CONSTANT(IrcacheCompactEntriesComputePush, push)
daxa_RWBufferPtr(IrcacheMetadata) ircache_meta_buf = push.uses.ircache_meta_buf;
daxa_RWBufferPtr(uint) ircache_life_buf = push.uses.ircache_life_buf;
daxa_BufferPtr(uint) entry_occupancy_buf = push.uses.entry_occupancy_buf;
daxa_RWBufferPtr(uint) ircache_entry_indirection_buf = push.uses.ircache_entry_indirection_buf;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint entry_idx = gl_GlobalInvocationID.x;
    const uint total_entry_count = deref(ircache_meta_buf).entry_count;

    const uint life = deref(advance(ircache_life_buf, entry_idx));
    if (entry_idx < total_entry_count && is_ircache_entry_life_valid(life)) {
        deref(advance(ircache_entry_indirection_buf, deref(advance(entry_occupancy_buf, entry_idx)))) = entry_idx;
    }
}
