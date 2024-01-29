#include <shared/app.inl>
#include "ircache_constants.glsl"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint entry_count = deref(ircache_meta_buf[IRCACHE_META_ENTRY_COUNT_INDEX]);

    const uint threads_per_group = 64;
    const uint entries_per_thread = 1;
    const uint divisor = threads_per_group * entries_per_thread;

    deref(dispatch_args[0]) = uvec4((entry_count + divisor - 1) / divisor, 1, 1, 0);
}