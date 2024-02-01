#include <shared/app.inl>
#include <utils/math.glsl>
// #include <utils/uv.glsl>
// #include <utils/pack_unpack.glsl>
// #include <utils/frame_constants.glsl>
#include <utils/gbuffer.glsl>
#include <utils/brdf.glsl>
#include <utils/brdf_lut.glsl>
#include <utils/layered_brdf.glsl>
// #include <utils/blue_noise.glsl>
#include <utils/rt.glsl>
// #include <utils/atmosphere.glsl>
// #include <utils/sun.glsl>
// #include <utils/lights/triangle.glsl>
#include <utils/reservoir.glsl>
// #include "../ircache/bindings.hlsl"
// #include "../wrc/bindings.hlsl"
// #include "../rtr/rtr_settings.hlsl"
#include "rtdgi_restir_settings.glsl"
#include "near_field_settings.glsl"

// #define IRCACHE_LOOKUP_DONT_KEEP_ALIVE
// #define IRCACHE_LOOKUP_KEEP_ALIVE_PROB 0.125

#include "../ircache/lookup.glsl"
// #include "../wrc/lookup.glsl"
#include "candidate_ray_dir.glsl"

#include "diffuse_trace_common.inc.glsl"
#include <utils/downscale.glsl>
#include <utils/safety.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    const uvec2 px = gl_GlobalInvocationID.xy;
    const ivec2 hi_px_offset = ivec2(HALFRES_SUBSAMPLE_OFFSET);
    const uvec2 hi_px = px * 2 + hi_px_offset;

    float depth = safeTexelFetch(depth_tex, ivec2(hi_px), 0).r;

    if (0.0 == depth) {
        safeImageStore(candidate_irradiance_out_tex, ivec2(px), vec4(0));
        safeImageStore(candidate_normal_out_tex, ivec2(px), vec4(0, 0, 1, 0));
        safeImageStore(rt_history_invalidity_out_tex, ivec2(px), vec4(0));
        return;
    }

    const vec2 uv = get_uv(hi_px, push.gbuffer_tex_size);
    const ViewRayContext view_ray_context = vrc_from_uv_and_biased_depth(globals, uv_to_ss(gpu_input, uv, push.gbuffer_tex_size), depth);

    const float NEAR_FIELD_FADE_OUT_END = -ray_hit_vs(view_ray_context).z * (SSGI_NEAR_FIELD_RADIUS * push.gbuffer_tex_size.w * 0.5);

#if RTDGI_INTERLEAVED_VALIDATION_ALWAYS_TRACE_NEAR_FIELD
    if (true)
#else
    if (is_rtdgi_tracing_frame())
#endif
    {
        const vec3 normal_vs = safeTexelFetch(half_view_normal_tex, ivec2(px), 0).xyz;
        const vec3 normal_ws = direction_view_to_world(globals, normal_vs);
        const mat3 tangent_to_world = build_orthonormal_basis(normal_ws);
        const vec3 outgoing_dir = rtdgi_candidate_ray_dir(px, tangent_to_world);

        RayDesc outgoing_ray;
        outgoing_ray.Direction = outgoing_dir;
        outgoing_ray.Origin = biased_secondary_ray_origin_ws_with_normal(view_ray_context, normal_ws);
        outgoing_ray.TMin = 0;

        if (is_rtdgi_tracing_frame()) {
            outgoing_ray.TMax = SKY_DIST;
        } else {
            outgoing_ray.TMax = NEAR_FIELD_FADE_OUT_END;
        }

        uint rng = hash3(uvec3(px, deref(gpu_input).frame_index & 31));
        TraceResult result = do_the_thing(px, normal_ws, rng, outgoing_ray);

#if RTDGI_INTERLEAVED_VALIDATION_ALWAYS_TRACE_NEAR_FIELD
        if (!is_rtdgi_tracing_frame() && !result.is_hit) {
            // If we were only tracing short rays, make sure we don't try to output
            // sky color upon misses.
            result.out_value = vec3(0);
            result.hit_t = SKY_DIST;
        }
#endif

        const vec3 hit_offset_ws = outgoing_ray.Direction * result.hit_t;

        const float cos_theta = dot(normalize(outgoing_dir - ray_dir_ws(view_ray_context)), normal_ws);
        safeImageStore(candidate_irradiance_out_tex, ivec2(px), vec4(result.out_value, rtr_encode_cos_theta_for_fp16(cos_theta)));
        safeImageStore(candidate_hit_out_tex, ivec2(px), vec4(hit_offset_ws, result.pdf * select(is_rtdgi_tracing_frame(), 1, -1)));
        safeImageStore(candidate_normal_out_tex, ivec2(px), vec4(direction_world_to_view(globals, result.hit_normal_ws), 0));
    }
    // } else {
    //     const vec4 reproj = reprojection_tex[hi_px];
    //     const ivec2 reproj_px = floor(px + gbuffer_tex_size.xy * reproj.xy / 2 + 0.5);
    //     candidate_irradiance_out_tex[px] = 0.0;
    //     candidate_hit_out_tex[px] = 0.0;
    //     candidate_normal_out_tex[px] = 0.0;
    // }

    const vec4 reproj = safeTexelFetch(reprojection_tex, ivec2(hi_px), 0);
    const ivec2 reproj_px = ivec2(floor(vec2(px) + push.gbuffer_tex_size.xy * reproj.xy / 2.0 + 0.5));
    safeImageStore(rt_history_invalidity_out_tex, ivec2(px), safeTexelFetch(rt_history_invalidity_in_tex, ivec2(reproj_px), 0));
}
