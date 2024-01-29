#include <shared/app.inl>
#include "ircache_constants.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    deref(ircache_pool_buf[idx]) = idx;
    deref(ircache_life_buf[idx]) = IRCACHE_ENTRY_LIFE_RECYCLED;
}
