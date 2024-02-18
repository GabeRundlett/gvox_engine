#include <renderer/kajiya/rtdgi.inl>

#include <utilities/gpu/math.glsl>
// #include <utilities/gpu/uv.glsl>
// #include <utilities/gpu/pack_unpack.glsl>
// #include "../inc/frame_constants.glsl"
#include "../inc/gbuffer.glsl"
#include "../inc/brdf.glsl"
#include "../inc/brdf_lut.glsl"
#include "../inc/layered_brdf.glsl"
// #include <utilities/gpu/blue_noise.glsl>
#include "../inc/rt.glsl"
// #include <utilities/gpu/atmosphere.glsl>
// #include <utilities/gpu/sun.glsl>
// #include <utilities/gpu/lights/triangle.glsl>
#include "../inc/reservoir.glsl"
// #include "../ircache/bindings.hlsl"
// #include "../wrc/bindings.glsl"
#include "rtdgi_restir_settings.glsl"

// #define IRCACHE_LOOKUP_DONT_KEEP_ALIVE
// #define IRCACHE_LOOKUP_KEEP_ALIVE_PROB 0.125

DAXA_DECL_PUSH_CONSTANT(RtdgiValidateComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewIndex half_view_normal_tex = push.uses.half_view_normal_tex;
daxa_ImageViewIndex depth_tex = push.uses.depth_tex;
daxa_ImageViewIndex reprojected_gi_tex = push.uses.reprojected_gi_tex;
daxa_ImageViewIndex reservoir_tex = push.uses.reservoir_tex;
daxa_ImageViewIndex reservoir_ray_history_tex = push.uses.reservoir_ray_history_tex;
daxa_ImageViewIndex blue_noise_vec2 = push.uses.blue_noise_vec2;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_BufferPtr)
IRCACHE_USE_BUFFERS_PUSH_USES()
daxa_ImageViewIndex sky_cube_tex = push.uses.sky_cube_tex;
daxa_ImageViewIndex transmittance_lut = push.uses.transmittance_lut;
daxa_ImageViewIndex irradiance_history_tex = push.uses.irradiance_history_tex;
daxa_ImageViewIndex ray_orig_history_tex = push.uses.ray_orig_history_tex;
daxa_ImageViewIndex rt_history_invalidity_out_tex = push.uses.rt_history_invalidity_out_tex;

#include "../ircache/lookup.glsl"
// #include "../wrc/lookup.hlsl"
#include "candidate_ray_dir.glsl"

#include "diffuse_trace_common.inc.glsl"
#include "../inc/downscale.glsl"
#include "../inc/safety.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    const uvec2 px = gl_GlobalInvocationID.xy;
    const ivec2 hi_px_offset = ivec2(HALFRES_SUBSAMPLE_OFFSET);
    const uvec2 hi_px = px * 2 + hi_px_offset;

    if (0.0 == safeTexelFetch(depth_tex, ivec2(hi_px), 0).r) {
        safeImageStore(rt_history_invalidity_out_tex, ivec2(px), vec4(1.0));
        return;
    }

    float invalidity = 0.0;

    // NOTE(grundlett): Here we fix the ray origin for when the player causes the world to wrap.
    // We technically don't need to write out if the world does not wrap in this frame, but IDC for now.
    vec3 prev_ray_orig = safeTexelFetch(ray_orig_history_tex, ivec2(px), 0).xyz;
    prev_ray_orig -= vec3(deref(gpu_input).player.player_unit_offset - deref(gpu_input).player.prev_unit_offset);
    safeImageStore(ray_orig_history_tex, ivec2(px), vec4(prev_ray_orig, 0));

    if (RESTIR_USE_PATH_VALIDATION && is_rtdgi_validation_frame(deref(gpu_input).frame_index)) {
        const vec3 normal_vs = safeTexelFetch(half_view_normal_tex, ivec2(px), 0).xyz;
        const vec3 normal_ws = direction_view_to_world(gpu_input, normal_vs);

        const vec3 prev_ray_delta = safeTexelFetch(reservoir_ray_history_tex, ivec2(px), 0).xyz;

        const vec4 prev_radiance_packed = safeTexelFetch(irradiance_history_tex, ivec2(px), 0);
        const vec3 prev_radiance = max(0.0.xxx, prev_radiance_packed.rgb);

        RayDesc prev_ray;
        prev_ray.Direction = normalize(prev_ray_delta);
        prev_ray.Origin = prev_ray_orig;
        prev_ray.TMin = 0;
        prev_ray.TMax = SKY_DIST;

        // TODO: frame index
        uint rng = hash3(uvec3(px, 0));

        TraceResult result = do_the_thing(px, normal_ws, rng, prev_ray);
        const vec3 new_radiance = max(0.0.xxx, result.out_value);

        const float rad_diff = length(abs(prev_radiance - new_radiance) / max(vec3(1e-3), prev_radiance + new_radiance));
        invalidity = smoothstep(0.1, 0.5, rad_diff / length(1.0.xxx));

        const float prev_hit_dist = length(prev_ray_delta);

        // If we hit more or less the same point, replace the hit radiance.
        // If the hit is different, it's possible that the previous origin point got obscured
        // by something, in which case we want M-clamping to take care of it instead.
        if (abs(result.hit_t - prev_hit_dist) / (prev_hit_dist + prev_hit_dist) < 0.2) {
            safeImageStore(irradiance_history_tex, ivec2(px), vec4(new_radiance, prev_radiance_packed.a));

            // When we update the radiance, we might end up with fairly low probability
            // rays hitting the bright spots by chance. The PDF division compounded
            // by the increase in radiance causes fireflies to appear.
            // As a HACK, we will clamp that by scaling down the `M` factor then.
            Reservoir1spp r = Reservoir1spp_from_raw(safeTexelFetchU(reservoir_tex, ivec2(px), 0).xy);
            const float lum_old = sRGB_to_luminance(prev_radiance);
            const float lum_new = sRGB_to_luminance(new_radiance);
            r.M *= clamp(lum_old / max(1e-8, lum_new), 0.03, 1.0);

            // Allow the new value to be greater than the old one up to a certain scale,
            // then dim it down by reducing `W`. It will recover over time.
            const float allowed_luminance_increment = 10.0;
            r.W *= clamp(lum_old / max(1e-8, lum_new) * allowed_luminance_increment, 0.01, 1.0);

            safeImageStoreU(reservoir_tex, ivec2(px), uvec4(as_raw(r), 0, 0));
        }
    }

    safeImageStore(rt_history_invalidity_out_tex, ivec2(px), vec4(invalidity, 0, 0, 0));
}
