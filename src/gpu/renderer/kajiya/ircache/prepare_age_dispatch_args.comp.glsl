#include <shared/renderer/ircache.inl>
#include "ircache_constants.glsl"

DAXA_DECL_PUSH_CONSTANT(IrcachePrepareAgeDispatchComputePush, push)
daxa_BufferPtr(IrcacheMetadata) ircache_meta_buf = push.uses.ircache_meta_buf;
daxa_RWBufferPtr(daxa_u32vec4) dispatch_args = push.uses.dispatch_args;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint entry_count = deref(ircache_meta_buf).entry_count;

    const uint threads_per_group = 64;
    const uint entries_per_thread = 1;
    const uint divisor = threads_per_group * entries_per_thread;

    deref(dispatch_args[0]) = uvec4((entry_count + divisor - 1) / divisor, 1, 1, 0);
}
