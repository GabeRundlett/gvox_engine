#include <renderer/kajiya/rtdgi.inl>

#include <utilities/gpu/math.glsl>
// #include <utilities/gpu/uv.glsl>
// #include <utilities/gpu/pack_unpack.glsl>
// #include <utilities/gpu/frame_constants.glsl>
#include <utilities/gpu/gbuffer.glsl>
#include <utilities/gpu/brdf.glsl>
#include <utilities/gpu/brdf_lut.glsl>
#include <utilities/gpu/layered_brdf.glsl>
// #include <utilities/gpu/blue_noise.glsl>
#include <utilities/gpu/rt.glsl>
// #include <utilities/gpu/atmosphere.glsl>
// #include <utilities/gpu/sun.glsl>
// #include <utilities/gpu/lights/triangle.glsl>
#include <utilities/gpu/reservoir.glsl>
// #include "../ircache/bindings.hlsl"
// #include "../wrc/bindings.hlsl"
#include "../rtr/rtr_settings.glsl"
#include "rtdgi_restir_settings.glsl"
#include "near_field_settings.glsl"

// #define IRCACHE_LOOKUP_DONT_KEEP_ALIVE
// #define IRCACHE_LOOKUP_KEEP_ALIVE_PROB 0.125

DAXA_DECL_PUSH_CONSTANT(RtdgiTraceComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewIndex half_view_normal_tex = push.uses.half_view_normal_tex;
daxa_ImageViewIndex depth_tex = push.uses.depth_tex;
daxa_ImageViewIndex reprojected_gi_tex = push.uses.reprojected_gi_tex;
daxa_ImageViewIndex reprojection_tex = push.uses.reprojection_tex;
daxa_ImageViewIndex blue_noise_vec2 = push.uses.blue_noise_vec2;
daxa_ImageViewIndex sky_cube_tex = push.uses.sky_cube_tex;
daxa_ImageViewIndex transmittance_lut = push.uses.transmittance_lut;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_BufferPtr)
IRCACHE_USE_BUFFERS_PUSH_USES()
daxa_ImageViewIndex ray_orig_history_tex = push.uses.ray_orig_history_tex;
daxa_ImageViewIndex candidate_irradiance_out_tex = push.uses.candidate_irradiance_out_tex;
daxa_ImageViewIndex candidate_normal_out_tex = push.uses.candidate_normal_out_tex;
daxa_ImageViewIndex candidate_hit_out_tex = push.uses.candidate_hit_out_tex;
daxa_ImageViewIndex rt_history_invalidity_in_tex = push.uses.rt_history_invalidity_in_tex;
daxa_ImageViewIndex rt_history_invalidity_out_tex = push.uses.rt_history_invalidity_out_tex;

#include "../ircache/lookup.glsl"
// #include "../wrc/lookup.glsl"
#include "candidate_ray_dir.glsl"

#include "diffuse_trace_common.inc.glsl"
#include <utilities/gpu/downscale.glsl>
#include <utilities/gpu/safety.glsl>

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
    const ViewRayContext view_ray_context = vrc_from_uv_and_biased_depth(globals, uv, depth);

    const float NEAR_FIELD_FADE_OUT_END = -ray_hit_vs(view_ray_context).z * (SSGI_NEAR_FIELD_RADIUS * push.gbuffer_tex_size.w * 0.5);

#if RTDGI_INTERLEAVED_VALIDATION_ALWAYS_TRACE_NEAR_FIELD
    if (true)
#else
    if (is_rtdgi_tracing_frame(deref(gpu_input).frame_index))
#endif
    {
        const vec3 normal_vs = safeTexelFetch(half_view_normal_tex, ivec2(px), 0).xyz;
        const vec3 normal_ws = direction_view_to_world(globals, normal_vs);
        const mat3 tangent_to_world = build_orthonormal_basis(normal_ws);
        const vec3 outgoing_dir = rtdgi_candidate_ray_dir(blue_noise_vec2, deref(gpu_input).frame_index, px, tangent_to_world);

        RayDesc outgoing_ray;
        outgoing_ray.Direction = outgoing_dir;
        outgoing_ray.Origin = biased_secondary_ray_origin_ws_with_normal(view_ray_context, normal_ws);
        outgoing_ray.TMin = 0;

        if (is_rtdgi_tracing_frame(deref(gpu_input).frame_index)) {
            outgoing_ray.TMax = SKY_DIST;
        } else {
            outgoing_ray.TMax = NEAR_FIELD_FADE_OUT_END;
        }

        uint rng = hash3(uvec3(px, deref(gpu_input).frame_index & 31));
        TraceResult result = do_the_thing(px, normal_ws, rng, outgoing_ray);

#if RTDGI_INTERLEAVED_VALIDATION_ALWAYS_TRACE_NEAR_FIELD
        if (!is_rtdgi_tracing_frame(deref(gpu_input).frame_index) && !result.is_hit) {
            // If we were only tracing short rays, make sure we don't try to output
            // sky color upon misses.
            result.out_value = vec3(0);
            result.hit_t = SKY_DIST;
        }
#endif

        const vec3 hit_offset_ws = outgoing_ray.Direction * result.hit_t;

        const float cos_theta = dot(normalize(outgoing_dir - ray_dir_ws(view_ray_context)), normal_ws);
        safeImageStore(candidate_irradiance_out_tex, ivec2(px), vec4(result.out_value, rtr_encode_cos_theta_for_fp16(cos_theta)));
        safeImageStore(candidate_hit_out_tex, ivec2(px), vec4(hit_offset_ws, result.pdf * select(is_rtdgi_tracing_frame(deref(gpu_input).frame_index), 1, -1)));
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
