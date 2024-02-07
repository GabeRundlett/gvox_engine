#include <renderer/kajiya/rtr.inl>

// #include <utils/uv.glsl>
// #include <utils/pack_unpack.glsl>
// #include <utils/frame_constants.glsl>
#include <utils/gbuffer.glsl>
#include <utils/brdf.glsl>
#include <utils/brdf_lut.glsl>
#include <utils/layered_brdf.glsl>
#include "blue_noise.glsl"
#include <utils/rt.glsl>
// #include <utils/atmosphere.glsl>
#include <utils/sky.glsl>
// #include <utils/sun.glsl>
// #include <utils/lights/triangle.glsl>
// #include "../ircache/bindings.glsl"
// #include "../wrc/bindings.hlsl"
#include "rtr_settings.glsl"

DAXA_DECL_PUSH_CONSTANT(RtrTraceComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_BufferPtr(daxa_i32) ranking_tile_buf = push.uses.ranking_tile_buf;
daxa_BufferPtr(daxa_i32) scambling_tile_buf = push.uses.scambling_tile_buf;
daxa_BufferPtr(daxa_i32) sobol_buf = push.uses.sobol_buf;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_BufferPtr)
IRCACHE_USE_BUFFERS_PUSH_USES()
daxa_ImageViewIndex gbuffer_tex = push.uses.gbuffer_tex;
daxa_ImageViewIndex depth_tex = push.uses.depth_tex;
daxa_ImageViewIndex rtdgi_tex = push.uses.rtdgi_tex;
daxa_ImageViewIndex sky_cube_tex = push.uses.sky_cube_tex;
daxa_ImageViewIndex transmittance_lut = push.uses.transmittance_lut;
daxa_ImageViewIndex out0_tex = push.uses.out0_tex;
daxa_ImageViewIndex out1_tex = push.uses.out1_tex;
daxa_ImageViewIndex out2_tex = push.uses.out2_tex;
daxa_ImageViewIndex rng_out_tex = push.uses.rng_out_tex;

// #define IRCACHE_LOOKUP_KEEP_ALIVE_PROB 0.125
#include "../ircache/lookup.glsl"
// #include "../wrc/lookup.hlsl"

#include "reflection_trace_common.inc.glsl"
#include <utils/downscale.glsl>
#include <utils/safety.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    const uvec2 px = gl_GlobalInvocationID.xy;
    const uvec2 hi_px = px * 2 + HALFRES_SUBSAMPLE_OFFSET;
    float depth = safeTexelFetch(depth_tex, ivec2(hi_px), 0).r;

    if (0.0 == depth) {
        safeImageStore(out0_tex, ivec2(px), vec4(0.0.xxx, -SKY_DIST));
        return;
    }

    const vec2 uv = get_uv(hi_px, push.gbuffer_tex_size);

    GbufferData gbuffer = unpack(GbufferDataPacked(safeTexelFetchU(gbuffer_tex, ivec2(hi_px), 0)));
    gbuffer.roughness = max(gbuffer.roughness, RTR_ROUGHNESS_CLAMP);

    // Initially, the candidate buffers contain candidates generated via diffuse tracing.
    // For rough surfaces we can skip generating new candidates just for reflections.
    // TODO: make this metric depend on spec contrast
    if (/* push.reuse_rtdgi_rays != 0 */ true && gbuffer.roughness > 0.6) {
        return;
    }

    const mat3 tangent_to_world = build_orthonormal_basis(gbuffer.normal);

#if RTR_USE_TIGHTER_RAY_BIAS
    const ViewRayContext view_ray_context = vrc_from_uv_and_biased_depth(globals, uv, depth);
    const vec3 refl_ray_origin_ws = biased_secondary_ray_origin_ws_with_normal(view_ray_context, gbuffer.normal);
#else
    const ViewRayContext view_ray_context = vrc_from_uv_and_depth(globals, uv, depth);
    const vec3 refl_ray_origin_ws = biased_secondary_ray_origin_ws(view_ray_context);
#endif

    vec3 wo = (-ray_dir_ws(view_ray_context)) * tangent_to_world;

    // Hack for shading normals facing away from the outgoing ray's direction:
    // We flip the outgoing ray along the shading normal, so that the reflection's curvature
    // continues, albeit at a lower rate.
    if (wo.z < 0.0) {
        wo.z *= -0.25;
        wo = normalize(wo);
    }

    SpecularBrdf specular_brdf;
    specular_brdf.albedo = mix(vec3(0.04), gbuffer.albedo, vec3(gbuffer.metalness));
    specular_brdf.roughness = gbuffer.roughness;

    const uint noise_offset = deref(gpu_input).frame_index * select(USE_TEMPORAL_JITTER, 1, 0);
    uint rng = hash3(uvec3(px, noise_offset));

#if 1
    // Note: since this is pre-baked for various SPP, can run into undersampling
    vec2 urand = vec2(
        blue_noise_sampler(int(px.x), int(px.y), int(noise_offset), 0,
                           ranking_tile_buf,
                           scambling_tile_buf,
                           sobol_buf),
        blue_noise_sampler(int(px.x), int(px.y), int(noise_offset), 1,
                           ranking_tile_buf,
                           scambling_tile_buf,
                           sobol_buf));
#else
    vec2 urand = blue_noise_for_pixel(px, noise_offset).xy;
#endif

    const float sampling_bias = SAMPLING_BIAS;
    urand.x = mix(urand.x, 0.0, sampling_bias);

    BrdfSample brdf_sample = sample_brdf(specular_brdf, wo, urand);

// VNDF still returns a lot of invalid samples on rough surfaces at F0 angles!
// TODO: move this to a separate sample preparation compute shader
#if USE_TEMPORAL_JITTER // && !USE_GGX_VNDF_SAMPLING
    for (uint retry_i = 0; retry_i < 4 && !is_valid(brdf_sample); ++retry_i) {
        urand = vec2(
            uint_to_u01_float(hash1_mut(rng)),
            uint_to_u01_float(hash1_mut(rng)));
        urand.x = max(urand.x, 0.0, sampling_bias);

        brdf_sample = sample_brdf(specular_brdf, wo, urand);
    }
#endif

    const float cos_theta = normalize(wo + brdf_sample.wi).z;

    if (is_valid(brdf_sample)) {
        // const bool use_short_ray = gbuffer.roughness > 0.55 && USE_SHORT_RAYS_FOR_ROUGH;

        RayDesc outgoing_ray;
        outgoing_ray.Direction = tangent_to_world * brdf_sample.wi;
        outgoing_ray.Origin = refl_ray_origin_ws;
        outgoing_ray.TMin = 0;
        outgoing_ray.TMax = SKY_DIST;

        // uint rng = hash2(px);
        safeImageStore(rng_out_tex, ivec2(px), uvec4(rng));
        RtrTraceResult result = do_the_thing(px, gbuffer.normal, gbuffer.roughness, rng, outgoing_ray);

        const vec3 direction_vs = direction_world_to_view(globals, outgoing_ray.Direction);
        const float to_surface_area_measure =
#if RTR_APPROX_MEASURE_CONVERSION
            1
#else
            abs(brdf_sample.wi.z * dot(result.hit_normal_vs, -direction_vs))
#endif
            / max(1e-10, result.hit_t * result.hit_t);

        const vec3 hit_offset_ws = outgoing_ray.Direction * result.hit_t;

        SpecularBrdfEnergyPreservation brdf_lut = SpecularBrdfEnergyPreservation_from_brdf_ndotv(specular_brdf, wo.z);

        const float pdf =
#if RTR_PDF_STORED_WITH_SURFACE_AREA_METRIC
            to_surface_area_measure *
#endif
            brdf_sample.pdf /
            // When sampling the BRDF in a path tracer, a certain fraction of samples
            // taken will be invalid. In the specular filtering pipe we force them all to be valid
            // in order to get the most out of our kernels. We also simply discard any rays
            // going in the wrong direction when reusing neighborhood samples, and renormalize,
            // whereas in a regular integration loop, we'd still count them.
            // Here we adjust the value back to what it would be if a fraction was returned invalid.
            brdf_lut.valid_sample_fraction;

        safeImageStore(out0_tex, ivec2(px), vec4(result.total_radiance, rtr_encode_cos_theta_for_fp16(cos_theta)));
        safeImageStore(out1_tex, ivec2(px), vec4(hit_offset_ws, pdf));
        safeImageStore(out2_tex, ivec2(px), vec4(result.hit_normal_vs, 0));
    } else {
        safeImageStore(out0_tex, ivec2(px), vec4(vec3(1, 0, 1), 0));
        safeImageStore(out1_tex, ivec2(px), 0.0.xxxx);
    }
}
