#define VOXEL_TRACE_WORLDSPACE true

#include <shared/app.inl>

#include <utils/math.glsl>
// #include <utils/pack_unpack.glsl>
// #include <utils/frame_constants.glsl>
#include <utils/gbuffer.glsl>
#include <utils/brdf.glsl>
#include <utils/brdf_lut.glsl>
#include <utils/layered_brdf.glsl>
#include <utils/rt.glsl>
#include <utils/random.glsl>
#include <utils/quasi_random.glsl>
#include <utils/reservoir.glsl>
// #include <utils/bindless_textures.glsl>
// #include <utils/atmosphere.glsl>
#include <utils/normal.glsl>
// #include <utils/lights/triangle.glsl>
#include <utils/color.glsl>

// #include "../inc/sun.hlsl"
// #include "../wrc/lookup.hlsl"

// Sample straight from the `ircache_aux_buf` instead of the SH.
#define IRCACHE_LOOKUP_PRECISE
#include "lookup.glsl"

#include "ircache_sampler_common.inc.glsl"
#include "ircache_trace_common.inc.glsl"

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
void main() {
    if (IRCACHE_FREEZE) {
        return;
    }

    const uint dispatch_idx = gl_GlobalInvocationID.x;

// AMD ray-tracing bug workaround; indirect RT seems to be tracing with the same
// ray count for multiple dispatches (???)
// Search for c804a814-fdc8-4843-b2c8-9d0674c10a6f for other occurences.
#if 1
    const uint alloc_count = deref(ircache_meta_buf[IRCACHE_META_TRACING_ALLOC_COUNT_INDEX]);
    if (dispatch_idx >= alloc_count * IRCACHE_SAMPLES_PER_FRAME) {
        return;
    }
#endif

    const uint entry_idx = deref(ircache_entry_indirection_buf[dispatch_idx / IRCACHE_SAMPLES_PER_FRAME]);
    const uint sample_idx = dispatch_idx % IRCACHE_SAMPLES_PER_FRAME;
    const uint life = deref(ircache_life_buf[entry_idx * 4]);
    const uint rank = ircache_entry_life_to_rank(life);

    VertexPacked packed_entry = deref(ircache_spatial_buf[entry_idx]);
    const Vertex entry = unpack_vertex(packed_entry);

    DiffuseBrdf brdf;
    // const float3x3 tangent_to_world = build_orthonormal_basis(entry.normal);

    brdf.albedo = 1.0.xxx;

// Allocate fewer samples for further bounces
#if 0
        const uint sample_count_divisor = 
            select(rank <= 1
            , 1
            , 4);
#else
    const uint sample_count_divisor = 1;
#endif

    uint rng = hash1(hash1(entry_idx) + deref(gpu_input).frame_index);

    const SampleParams sample_params = SampleParams_from_spf_entry_sample_frame(
        IRCACHE_SAMPLES_PER_FRAME,
        entry_idx,
        sample_idx,
        deref(gpu_input).frame_index);

    IrcacheTraceResult traced = ircache_trace(entry, brdf, sample_params, life);

    const float self_lighting_limiter =
        select(USE_SELF_LIGHTING_LIMITER, mix(0.5, 1, smoothstep(-0.1, 0, dot(traced.direction, entry.normal))), 1.0);

    const vec3 new_value = traced.incident_radiance * self_lighting_limiter;
    const float new_lum = sRGB_to_luminance(new_value);

    Reservoir1sppStreamState stream_state = Reservoir1sppStreamState_create();
    Reservoir1spp reservoir = Reservoir1spp_create();
    init_with_stream(reservoir, new_lum, 1.0, stream_state, raw(sample_params));

    const uint octa_idx = octa_idx(sample_params);
    const uint output_idx = entry_idx * IRCACHE_AUX_STRIDE + octa_idx;

    vec4 prev_value_and_count =
        deref(ircache_aux_buf[output_idx + IRCACHE_OCTA_DIMS2]) * vec4((deref(gpu_input).pre_exposure_delta).xxx, 1);

    vec3 val_sel = new_value;
    bool selected_new = true;

    {
        const uint M_CLAMP = 30;

        Reservoir1spp r = Reservoir1spp_from_raw(floatBitsToUint(deref(ircache_aux_buf[output_idx]).xy));
        if (r.M > 0) {
            r.M = min(r.M, M_CLAMP);

            Vertex prev_entry = unpack_vertex(VertexPacked(deref(ircache_aux_buf[output_idx + IRCACHE_OCTA_DIMS2 * 2])));
            // prev_entry.position = entry.position;

            if (update_with_stream(reservoir,
                                   r, sRGB_to_luminance(prev_value_and_count.rgb), 1.0,
                                   stream_state, r.payload, rng)) {
                val_sel = prev_value_and_count.rgb;
                selected_new = false;
            }
        }
    }

    finish_stream(reservoir, stream_state);

    deref(ircache_aux_buf[output_idx]).xy = uintBitsToFloat(as_raw(reservoir));
    deref(ircache_aux_buf[output_idx + IRCACHE_OCTA_DIMS2]) = vec4(val_sel, reservoir.W);

    if (selected_new) {
        deref(ircache_aux_buf[output_idx + IRCACHE_OCTA_DIMS2 * 2]) = packed_entry.data0;
    }
}
