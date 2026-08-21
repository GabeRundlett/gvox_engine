// Trace rays between the previous ReSTIR ircache trace origins and the newly proposed ones,
// reducing the memory of reservoirs that are inaccessible now.
//
// This speeds up transitions between indoors/outdoors for cache entries which span both sides.
#define DAXA_RAY_TRACING 1
#extension GL_EXT_ray_tracing : enable

#include <renderer/kajiya/ircache.inl>
DAXA_DECL_PUSH_CONSTANT(IrcacheTraceAccessRtPush, push)
#include <renderer/rt.glsl>

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_RAYGEN

layout(location = PAYLOAD_LOC) rayPayloadEXT RayPayload prd;

daxa_BufferPtr(VertexPacked) ircache_spatial_buf = push.uses.ircache_spatial_buf;
daxa_BufferPtr(uint) ircache_life_buf = push.uses.ircache_life_buf;
daxa_RWBufferPtr(VertexPacked) ircache_reposition_proposal_buf = push.uses.ircache_reposition_proposal_buf;
daxa_BufferPtr(IrcacheMetadata) ircache_meta_buf = push.uses.ircache_meta_buf;
daxa_RWBufferPtr(IrcacheAux) ircache_aux_buf = push.uses.ircache_aux_buf;
daxa_BufferPtr(uint) ircache_entry_indirection_buf = push.uses.ircache_entry_indirection_buf;

#include <renderer/kajiya/inc/rt.glsl>
#include <renderer/kajiya/inc/reservoir.glsl>
#include <utilities/gpu/normal.glsl>
#include "ircache_constants.glsl"

#include <voxels/voxel.glsl>

void main() {
    if (IRCACHE_FREEZE) {
        return;
    }

    const uint dispatch_idx = gl_LaunchIDEXT.x;

// AMD ray-tracing bug workaround; indirect RT seems to be tracing with the same
// ray count for multiple dispatches (???)
// Search for c804a814-fdc8-4843-b2c8-9d0674c10a6f for other occurences.
#if 1
    const uint alloc_count = deref(ircache_meta_buf).tracing_alloc_count;
    if (dispatch_idx >= alloc_count * IRCACHE_OCTA_DIMS2) {
        return;
    }
#endif

    const uint entry_idx = deref(advance(ircache_entry_indirection_buf, dispatch_idx / IRCACHE_OCTA_DIMS2));
    const uint octa_idx = dispatch_idx % IRCACHE_OCTA_DIMS2;
    const uint life = deref(advance(ircache_life_buf, entry_idx));

    if (!is_ircache_entry_life_valid(life)) {
        return;
    }

    const Vertex entry = unpack_vertex(deref(advance(ircache_spatial_buf, entry_idx)));

    Reservoir1spp r = Reservoir1spp_from_raw(deref(advance(ircache_aux_buf, entry_idx)).reservoirs[octa_idx].xy);
    Vertex prev_entry = unpack_vertex(deref(advance(ircache_aux_buf, entry_idx)).vertexes[octa_idx]);

    // Reduce weight of samples whose trace origins are not accessible now
    if (rt_is_shadowed(new_ray(
            entry.position,
            prev_entry.position - entry.position,
            0.001,
            0.999))) {
        r.M *= 0.8;
        deref(advance(ircache_aux_buf, entry_idx)).reservoirs[octa_idx].xy = as_raw(r);
    }
}
#endif
