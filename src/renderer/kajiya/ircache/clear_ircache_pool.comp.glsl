#include <renderer/kajiya/ircache.inl>
#include "ircache_constants.glsl"

DAXA_DECL_PUSH_CONSTANT(ClearIrcachePoolComputePush, push)
daxa_RWBufferPtr(uint) ircache_pool_buf = push.uses.ircache_pool_buf;
daxa_RWBufferPtr(uint) ircache_life_buf = push.uses.ircache_life_buf;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    deref(advance(ircache_pool_buf, idx)) = idx;
    deref(advance(ircache_life_buf, idx)) = IRCACHE_ENTRY_LIFE_RECYCLED;
}
