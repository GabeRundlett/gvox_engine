#include <renderer/kajiya/rtr.inl>

// #include <utilities/gpu/uv.glsl>
// #include <utilities/gpu/pack_unpack.glsl>
// #include <utilities/gpu/frame_constants.glsl>
#include <utilities/gpu/gbuffer.glsl>
#include <utilities/gpu/brdf.glsl>
#include <utilities/gpu/brdf_lut.glsl>
#include <utilities/gpu/layered_brdf.glsl>
#include "blue_noise.glsl"
#include <utilities/gpu/rt.glsl>
// #include <utilities/gpu/atmosphere.glsl>
// #include <utilities/gpu/sun.glsl>
// #include <utilities/gpu/lights/triangle.glsl>
#include <utilities/gpu/reservoir.glsl>
// #include "../ircache/bindings.hlsl"
// #include "../wrc/bindings.hlsl"
#include "rtr_settings.glsl"

DAXA_DECL_PUSH_CONSTANT(RtrValidateComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_BufferPtr)
IRCACHE_USE_BUFFERS_PUSH_USES()
daxa_ImageViewIndex gbuffer_tex = push.uses.gbuffer_tex;
daxa_ImageViewIndex depth_tex = push.uses.depth_tex;
daxa_ImageViewIndex rtdgi_tex = push.uses.rtdgi_tex;
daxa_ImageViewIndex sky_cube_tex = push.uses.sky_cube_tex;
daxa_ImageViewIndex transmittance_lut = push.uses.transmittance_lut;
daxa_ImageViewIndex refl_restir_invalidity_tex = push.uses.refl_restir_invalidity_tex;
daxa_ImageViewIndex ray_orig_history_tex = push.uses.ray_orig_history_tex;
daxa_ImageViewIndex ray_history_tex = push.uses.ray_history_tex;
daxa_ImageViewIndex rng_history_tex = push.uses.rng_history_tex;
daxa_ImageViewIndex irradiance_history_tex = push.uses.irradiance_history_tex;
daxa_ImageViewIndex reservoir_history_tex = push.uses.reservoir_history_tex;

// #define IRCACHE_LOOKUP_KEEP_ALIVE_PROB 0.125
#include "../ircache/lookup.glsl"
// #include "../wrc/lookup.hlsl"

#include "reflection_trace_common.inc.glsl"
#include <utilities/gpu/downscale.glsl>
#include <utilities/gpu/safety.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    if (RTR_RESTIR_USE_PATH_VALIDATION == 0) {
        return;
    }

    // Validation at half-res
    const uvec2 px = gl_GlobalInvocationID.xy * 2 + HALFRES_SUBSAMPLE_OFFSET;
    // const uvec2 px = DispatchRaysIndex().xy;

    // Standard jitter from the other reflection passes
    const uvec2 hi_px = px * 2 + HALFRES_SUBSAMPLE_OFFSET;
    float depth = safeTexelFetch(depth_tex, ivec2(hi_px), 0).r;

    // refl_restir_invalidity_tex[px] = 0;

    if (0.0 == depth) {
        safeImageStoreU(refl_restir_invalidity_tex, ivec2(px), uvec4(1));
        return;
    }

    const vec2 uv = get_uv(hi_px, push.gbuffer_tex_size);

    GbufferData gbuffer = unpack(GbufferDataPacked(safeTexelFetchU(gbuffer_tex, ivec2(hi_px), 0)));
    gbuffer.roughness = max(gbuffer.roughness, RTR_ROUGHNESS_CLAMP);

    const mat3 tangent_to_world = build_orthonormal_basis(gbuffer.normal);

#if RTR_USE_TIGHTER_RAY_BIAS
    const ViewRayContext view_ray_context = vrc_from_uv_and_biased_depth(globals, uv, depth);
    const vec3 refl_ray_origin_ws = biased_secondary_ray_origin_ws_with_normal(view_ray_context, gbuffer.normal);
#else
    const ViewRayContext view_ray_context = vrc_from_uv_and_depth(globals, uv, depth);
    const vec3 refl_ray_origin_ws = biased_secondary_ray_origin_ws(view_ray_context);
#endif

    // TODO: frame consistency
    const uint noise_offset = deref(gpu_input).frame_index * select(USE_TEMPORAL_JITTER, 1, 0);

    const vec3 ray_orig_ws = safeTexelFetch(ray_orig_history_tex, ivec2(px), 0).xyz + get_prev_eye_position(globals);
    const vec3 ray_hit_ws = safeTexelFetch(ray_history_tex, ivec2(px), 0).xyz + ray_orig_ws;

    RayDesc outgoing_ray;
    outgoing_ray.Direction = normalize(ray_hit_ws - ray_orig_ws);
    outgoing_ray.Origin = ray_orig_ws;
    outgoing_ray.TMin = 0;
    outgoing_ray.TMax = SKY_DIST;

    // uint rng = hash2(px);
    uint rng = safeTexelFetchU(rng_history_tex, ivec2(px), 0).x;
    RtrTraceResult result = do_the_thing(px, gbuffer.normal, gbuffer.roughness, rng, outgoing_ray);

    Reservoir1spp r = Reservoir1spp_from_raw(safeTexelFetchU(reservoir_history_tex, ivec2(px), 0).xy);

    const vec4 prev_irradiance_packed = safeTexelFetch(irradiance_history_tex, ivec2(px), 0);
    const vec3 prev_irradiance = max(0.0.xxx, prev_irradiance_packed.rgb * deref(gpu_input).pre_exposure_delta);
    const vec3 check_radiance = max(0.0.xxx, result.total_radiance);

    const float rad_diff = length(abs(prev_irradiance - check_radiance) / max(vec3(1e-3), prev_irradiance + check_radiance));
    const float invalidity = smoothstep(0.1, 0.5, rad_diff / length(1.0.xxx));

    // r.M = max(0, min(r.M, exp2(log2(float(RTR_RESTIR_TEMPORAL_M_CLAMP)) * (1.0 - invalidity))));
    r.M *= 1 - invalidity;

    // TODO: also update hit point and normal
    // TODO: does this also need a hit_t check as in rtdgi restir validation?
    // TOD:: rename to radiance
    safeImageStore(irradiance_history_tex, ivec2(px), vec4(check_radiance, prev_irradiance_packed.a));

    safeImageStore(refl_restir_invalidity_tex, ivec2(px), vec4(invalidity));
    safeImageStoreU(reservoir_history_tex, ivec2(px), uvec4(as_raw(r), 0, 0));

// Also reduce M of the neighbors in case we have fewer validation rays than irradiance rays.
#if 1
    for (uint i = 1; i <= 3; ++i) {
        // const uvec2 main_px = px;
        // const uvec2 px = (main_px & ~1u) + HALFRES_SUBSAMPLE_OFFSET;
        const uvec2 px = gl_GlobalInvocationID.xy * 2 + hi_px_subpixels[(deref(gpu_input).frame_index + i) & 3];

        const vec4 neighbor_prev_irradiance_packed = safeTexelFetch(irradiance_history_tex, ivec2(px), 0);
        {
            const vec3 a = max(0.0.xxx, neighbor_prev_irradiance_packed.rgb * deref(gpu_input).pre_exposure_delta);
            const vec3 b = prev_irradiance;
            const float neigh_rad_diff = length(abs(a - b) / max(vec3(1e-8), a + b));

            // If the neighbor and us tracked similar radiance, assume it would also have
            // a similar change in value upon validation.
            if (neigh_rad_diff < 0.2) {
                // With this assumption, we'll replace the neighbor's old radiance with our own new one.
                safeImageStore(irradiance_history_tex, ivec2(px), vec4(check_radiance, neighbor_prev_irradiance_packed.a));
            }
        }

        safeImageStore(refl_restir_invalidity_tex, ivec2(px), vec4(invalidity));

        if (invalidity > 0) {
            Reservoir1spp r = Reservoir1spp_from_raw(safeTexelFetchU(reservoir_history_tex, ivec2(px), 0).xy);
            // r.M = max(0, min(r.M, exp2(log2(float(RTR_RESTIR_TEMPORAL_M_CLAMP)) * (1.0 - invalidity))));
            r.M *= 1 - invalidity;
            safeImageStoreU(reservoir_history_tex, ivec2(px), uvec4(as_raw(r), 0, 0));
        }
    }
#endif
}
