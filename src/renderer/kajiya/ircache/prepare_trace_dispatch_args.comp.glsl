#include <renderer/kajiya/ircache.inl>
#include "ircache_constants.glsl"

DAXA_DECL_PUSH_CONSTANT(IrcachePrepareTraceDispatchComputePush, push)
daxa_RWBufferPtr(IrcacheMetadata) ircache_meta_buf = push.uses.ircache_meta_buf;
daxa_RWBufferPtr(uvec4) dispatch_args = push.uses.dispatch_args;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint entry_count = deref(ircache_meta_buf).entry_count;
    const uint alloc_count = deref(ircache_meta_buf).alloc_count;

    deref(ircache_meta_buf).tracing_alloc_count = alloc_count;

    // Reset, sum up irradiance
    deref(dispatch_args[2]) = uvec4((alloc_count + 63) / 64, 1, 1, 0);

    uint main_rt_samples = alloc_count * IRCACHE_SAMPLES_PER_FRAME;
    uint accessibility_rt_samples = alloc_count * IRCACHE_OCTA_DIMS2;
    uint validity_rt_samples = alloc_count * IRCACHE_VALIDATION_SAMPLES_PER_FRAME;

// AMD ray-tracing bug workaround; indirect RT seems to be tracing with the same
// ray count for multiple dispatches (???)
// Search for c804a814-fdc8-4843-b2c8-9d0674c10a6f for other occurences.
#if 1
    const uint max_rt_samples =
        max(main_rt_samples, max(accessibility_rt_samples, validity_rt_samples));

    main_rt_samples = max_rt_samples;
    accessibility_rt_samples = max_rt_samples;
    validity_rt_samples = max_rt_samples;
#endif

    // Main ray tracing
    deref(dispatch_args[0]) = uvec4((main_rt_samples + 31) / 32, 1, 1, 0);

    // Accessibility tracing
    deref(dispatch_args[1]) = uvec4((accessibility_rt_samples + 31) / 32, 1, 1, 0);

    // Validity check
    deref(dispatch_args[3]) = uvec4((validity_rt_samples + 31) / 32, 1, 1, 0);
}
