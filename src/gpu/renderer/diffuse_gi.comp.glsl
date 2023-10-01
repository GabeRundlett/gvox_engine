#include <shared/app.inl>

#include <utils/reservoir.glsl>
#include <utils/downscale.glsl>
#include <utils/occlusion_raymarch.glsl>

#define FRAME_CONSTANTS_PRE_EXPOSURE_DELTA 1.0
#define M_PI PI
#define M_TAU (PI * 2.0)
#define GOLDEN_ANGLE 2.39996323
#define SKY_DIST MAX_DIST

// Restir settings

#define DIFFUSE_GI_USE_RESTIR 1
#define RESTIR_TEMPORAL_M_CLAMP 20.0

// Reduces fireflies, but causes darkening in corners
#define RESTIR_RESERVOIR_W_CLAMP 10.0
// RTDGI_RESTIR_USE_JACOBIAN_BASED_REJECTION covers the same niche.
// #define RESTIR_RESERVOIR_W_CLAMP 1e5

#define RESTIR_USE_SPATIAL true
#define RESTIR_TEMPORAL_USE_PERMUTATIONS true
#define RESTIR_USE_PATH_VALIDATION true

// Narrow down the spatial resampling kernel when M is already high.
#define RTDGI_RESTIR_SPATIAL_USE_KERNEL_NARROWING true

#define RTDGI_RESTIR_SPATIAL_USE_RAYMARCH true
#define RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE false
#define RTDGI_RESTIR_USE_JACOBIAN_BASED_REJECTION false
#define RTDGI_RESTIR_JACOBIAN_BASED_REJECTION_VALUE 8

#define RTDGI_RESTIR_USE_RESOLVE_SPATIAL_FILTER 1

// If `1`, every RTDGI_INTERLEAVED_VALIDATION_PERIOD-th frame is a validation one,
// where new candidates are not suggested, but the reservoir picks are validated instead.
// This hides the cost of validation, but reduces responsiveness.
#define RTDGI_INTERLEAVE_TRACING_AND_VALIDATION 1

// How often should validation happen in interleaved mode. Lower values result
// in more frequent validation, but less frequent candidate generation.
// Note that validation also updates irradiance values, so those frames are not useless
// for the purpose of integration either.
#define RTDGI_INTERLEAVED_VALIDATION_PERIOD 3

// If `1`, we will always trace candidate rays, but keep them short on frames where
// `is_rtdgi_tracing_frame` yields false.
// This preserves contact lighting, but introduces some frame jitter.
// New traces are a bit more expensive than validations, so the jitter is not terrible.
#define RTDGI_INTERLEAVED_VALIDATION_ALWAYS_TRACE_NEAR_FIELD 1
// ---

#define SSGI_NEAR_FIELD_RADIUS 80.0
#define USE_SPLIT_RT_NEAR_FIELD 1
const bool USE_SHARPENING_HISTORY_FETCH = true;

f32vec4 decode_hit_normal_and_dot(f32vec4 val) {
    return f32vec4(val.xyz * 2 - 1, val.w);
}

f32vec4 encode_hit_normal_and_dot(f32vec4 val) {
    return f32vec4(val.xyz * 0.5 + 0.5, val.w);
}

struct TemporalReservoirOutput {
    float depth;
    f32vec3 ray_hit_offset_ws;
    float luminance;
    f32vec3 hit_normal_ws;
};

TemporalReservoirOutput TemporalReservoirOutput_from_raw(u32vec4 raw) {
    f32vec4 ray_hit_offset_and_luminance = f32vec4(
        unpackHalf2x16(raw.y),
        unpackHalf2x16(raw.z));

    TemporalReservoirOutput res;
    res.depth = uintBitsToFloat(raw.x);
    res.ray_hit_offset_ws = ray_hit_offset_and_luminance.xyz;
    res.luminance = ray_hit_offset_and_luminance.w;
    res.hit_normal_ws = unpack_normal_11_10_11(uintBitsToFloat(raw.w));
    return res;
}

u32vec4 TemporalReservoirOutput_as_raw(inout TemporalReservoirOutput self) {
    u32vec4 raw;
    raw.x = floatBitsToUint(self.depth);
    raw.y = packHalf2x16(self.ray_hit_offset_ws.xy);
    raw.z = packHalf2x16(f32vec2(self.ray_hit_offset_ws.z, self.luminance));
    raw.w = floatBitsToUint(pack_normal_11_10_11(self.hit_normal_ws));
    return raw;
}

// f32vec2 r2_sequence(uint i) {
//     const float a1 = 1.0 / M_PLASTIC;
//     const float a2 = 1.0 / (M_PLASTIC * M_PLASTIC);
//     return fract(f32vec2(a1, a2) * f32vec2(i) + 0.5);
// }
f32vec4 blue_noise_for_pixel(daxa_ImageViewId blue_noise_image, u32vec2 px, uint n) {
    // const u32vec2 tex_dims = u32vec2(128, 128);
    // const u32vec2 offset = u32vec2(r2_sequence(n) * tex_dims);
    f32vec2 blue_noise = texelFetch(daxa_texture3D(blue_noise_image), ivec3(px, n) & ivec3(127, 127, 63), 0).xy;
    return blue_noise.xyxy;
}

struct RayDesc {
    f32vec3 Origin;
    float TMin;
    f32vec3 Direction;
    float TMax;
};

bool is_rtdgi_validation_frame() {
#if ENABLE_RESTIR
#if RTDGI_INTERLEAVE_TRACING_AND_VALIDATION
    return deref(gpu_input).frame_index % RTDGI_INTERLEAVED_VALIDATION_PERIOD == 0;
#else
    return true;
#endif
#else
    return false;
#endif
}

bool is_rtdgi_tracing_frame() {
#if ENABLE_RESTIR
#if RTDGI_INTERLEAVE_TRACING_AND_VALIDATION
    return !is_rtdgi_validation_frame();
#else
    return true;
#endif
#else
    return true;
#endif
}

#define select(cond, a, b) ((cond) ? (a) : (b))

#if RTDGI_TRACE_COMPUTE || RTDGI_VALIDATE_COMPUTE
#include <voxels/core.glsl>
#include <utils/sky.glsl>

struct TraceResult {
    f32vec3 out_value;
    f32vec3 hit_normal_ws;
    float hit_t;
    float pdf;
    bool is_hit;
};

TraceResult do_the_thing(u32vec2 px, f32vec3 normal_ws, inout uint rng, RayDesc outgoing_ray) {
    f32vec3 total_radiance = 0.0.xxx;
    f32vec3 hit_normal_ws = -outgoing_ray.Direction;

    float hit_t = outgoing_ray.TMax;
    float pdf = max(0.0, 1.0 / (dot(normal_ws, outgoing_ray.Direction) * 2.0 * PI));

    f32vec3 ray_pos = outgoing_ray.Origin;
    VoxelTraceResult trace_result = voxel_trace(VoxelTraceInfo(VOXELS_BUFFER_PTRS, outgoing_ray.Direction, MAX_STEPS, outgoing_ray.TMax, 0.0, true), ray_pos);

    TraceResult result;
    result.is_hit = (trace_result.dist != outgoing_ray.TMax);

    if (result.is_hit) {
        // total_radiance += f32vec3(0);
        hit_t = trace_result.dist;
        hit_normal_ws = trace_result.nrm;
        f32vec3 hit_albedo = uint_rgba8_to_f32vec4(trace_result.voxel_data).rgb;

        // Project the sample into clip space, and check if it's on-screen
        const f32vec3 primary_hit_cs = position_world_to_sample(globals, ray_pos);
        const f32vec2 primary_hit_uv = cs_to_uv(primary_hit_cs.xy);
        const float primary_hit_screen_depth = textureLod(daxa_sampler2D(depth_tex, deref(gpu_input).sampler_nnc), primary_hit_uv, 0).r;
        bool is_on_screen =
            true &&
            all(bvec2(abs(primary_hit_cs.x) < 1.0, abs(primary_hit_cs.y) < 1.0)) &&
            inverse_depth_relative_diff(primary_hit_cs.z, primary_hit_screen_depth) < 5e-3
            // TODO
            //&& dot(primary_hit_screen_normal_ws, -outgoing_ray.Direction) > 0.0
            //&& dot(primary_hit_screen_normal_ws, gbuffer.normal) > 0.7
            ;

        // If it is on-screen, we'll try to use its reprojected radiance from the previous frame
        f32vec4 reprojected_radiance = f32vec4(0);
        if (is_on_screen) {
            reprojected_radiance =
                textureLod(daxa_sampler2D(reprojected_gi_tex, deref(gpu_input).sampler_nnc), primary_hit_uv, 0) * FRAME_CONSTANTS_PRE_EXPOSURE_DELTA;

            // Check if the temporal reprojection is valid.
            is_on_screen = reprojected_radiance.w > 0;
        }

        // gbuffer.roughness = lerp(gbuffer.roughness, 1.0, ROUGHNESS_BIAS);
        const f32mat3x3 tangent_to_world = tbn_from_normal(hit_normal_ws);
        const f32vec3 wo = (-outgoing_ray.Direction * tangent_to_world);
        // const LayeredBrdf brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);

        f32vec3 sun_radiance = f32vec3(1); // SUN_COL;
        {
            const f32vec3 to_light_norm = SUN_DIR;
            ray_pos += to_light_norm * 1.0e-4;
            VoxelTraceResult sun_trace_result = voxel_trace(VoxelTraceInfo(VOXELS_BUFFER_PTRS, to_light_norm, MAX_STEPS, MAX_DIST, 0.0, true), ray_pos);
            const bool is_shadowed = (sun_trace_result.dist != outgoing_ray.TMax);

            const f32vec3 wi = (to_light_norm * tangent_to_world);
            const f32vec3 brdf_value = max(f32vec3(0.0), dot(hit_normal_ws, to_light_norm)); // brdf.evaluate(wo, wi) * max(0.0, wi.z);
            const f32vec3 light_radiance = select(is_shadowed, f32vec3(0.0), sun_radiance);
            total_radiance += brdf_value * light_radiance;
        }

        // total_radiance += trace_result.emissive;

        if (is_on_screen) {
#if ENABLE_RESTIR
            total_radiance += reprojected_radiance.rgb * hit_albedo;
#endif
        } else {
            // if (USE_IRCACHE) {
            //     const f32vec3 gi = IrcacheLookupParams::create(
            //         outgoing_ray.Origin,
            //         primary_hit.position,
            //         gbuffer.normal)
            //         .with_query_rank(1)
            //         .lookup(rng);
            //     total_radiance += gi * hit_albedo;
            // }
        }
    } else {
        // total_radiance += sample_sky_ambient(outgoing_ray.Direction);
    }

    f32vec3 out_value = total_radiance;

    result.out_value = out_value;
    result.hit_t = hit_t;
    result.hit_normal_ws = hit_normal_ws;
    result.pdf = pdf;
    return result;
}

#endif

#if RTDGI_TEMPORAL_COMPUTE
DAXA_DECL_PUSH_CONSTANT(RtdgiTemporalPush, push)

#define USE_BBOX_CLAMP 1

#define linear_to_working linear_rgb_to_crunched_luma_chroma
#define working_to_linear crunched_luma_chroma_to_linear_rgb
float working_luma(f32vec3 v) { return v.x; }

#define USE_TEMPORAL_FILTER 1

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 px = gl_GlobalInvocationID.xy;
#if !USE_TEMPORAL_FILTER
    imageStore(daxa_image2D(output_tex), i32vec2(px), max(0.0, texelFetch(daxa_texture2D(input_tex), i32vec2(px), 0)));
    imageStore(daxa_image2D(history_output_tex), i32vec2(px), f32vec4(max(0.0, texelFetch(daxa_texture2D(input_tex), i32vec2(px), 0).rgb), 32));
    return;
#endif

    f32vec2 uv = get_uv(px, push.output_tex_size);

    f32vec4 center = linear_to_working(texelFetch(daxa_texture2D(input_tex), i32vec2(px), 0));
    f32vec4 reproj = texelFetch(daxa_texture2D(reprojection_tex), i32vec2(px), 0);

    const f32vec4 history_mult = f32vec4((FRAME_CONSTANTS_PRE_EXPOSURE_DELTA).xxx, 1);
    f32vec4 history = linear_to_working(texelFetch(daxa_texture2D(history_tex), i32vec2(px), 0) * history_mult);

    // imageStore(daxa_image2D(output_tex), i32vec2(px), center);
    // return;

#if 1
    f32vec4 vsum = 0.0.xxxx;
    f32vec4 vsum2 = 0.0.xxxx;
    float wsum = 0.0;
    float hist_diff = 0.0;
    float hist_vsum = 0.0;
    float hist_vsum2 = 0.0;

    // float dev_sum = 0.0;

    const int k = 2;
    {
        for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                f32vec4 neigh = linear_to_working(texelFetch(daxa_texture2D(input_tex), i32vec2(px + i32vec2(x, y)), 0));
                f32vec4 hist_neigh = linear_to_working(texelFetch(daxa_texture2D(history_tex), i32vec2(px + i32vec2(x, y)), 0) * history_mult);

                float neigh_luma = working_luma(neigh.rgb);
                float hist_luma = working_luma(hist_neigh.rgb);

                float w = exp(-3.0 * float(x * x + y * y) / float((k + 1.) * (k + 1.)));
                vsum += neigh * w;
                vsum2 += neigh * neigh * w;
                wsum += w;

                // dev_sum += neigh.a * neigh.a * w;

                // hist_diff += (neigh_luma - hist_luma) * (neigh_luma - hist_luma) * w;
                hist_diff += abs(neigh_luma - hist_luma) / max(1e-5, neigh_luma + hist_luma) * w;
                hist_vsum += hist_luma * w;
                hist_vsum2 += hist_luma * hist_luma * w;
            }
        }
    }

    f32vec4 ex = vsum / wsum;
    f32vec4 ex2 = vsum2 / wsum;
    f32vec4 dev = sqrt(max(0.0.xxxx, ex2 - ex * ex));

    hist_diff /= wsum;
    hist_vsum /= wsum;
    hist_vsum2 /= wsum;
    // dev_sum /= wsum;

    const f32vec2 moments_history =
        textureLod(daxa_sampler2D(variance_history_tex, deref(gpu_input).sampler_lnc), uv + reproj.xy, 0).xy * f32vec2(FRAME_CONSTANTS_PRE_EXPOSURE_DELTA, FRAME_CONSTANTS_PRE_EXPOSURE_DELTA * FRAME_CONSTANTS_PRE_EXPOSURE_DELTA);

    // const float center_luma = working_luma(center.rgb);
    const float center_luma = working_luma(center.rgb) + (hist_vsum - working_luma(ex.rgb)); // - 0.5 * working_luma(control_variate.rgb));
    const f32vec2 current_moments = f32vec2(center_luma, center_luma * center_luma);
    imageStore(daxa_image2D(variance_history_output_tex), i32vec2(px), f32vec4(max(f32vec2(0.0), mix(moments_history, current_moments, f32vec2(0.25))), 0.0, 0.0));
    const float center_temporal_dev = sqrt(max(0.0, moments_history.y - moments_history.x * moments_history.x));

    float center_dev = center.a;

    // Spatial-only variance estimate (dev.rgb) has halos around edges (lighting discontinuities)

    // Temporal variance estimate with a spatial boost
    // TODO: this version reduces flicker in pica and on skeletons in battle, but has halos in cornell_box
    // dev.rgb = center_dev * dev.rgb / max(1e-8, clamp(working_luma(dev.rgb), center_dev * 0.1, center_dev * 3.0));

    // Spatiotemporal variance estimate
    // TODO: this version seems to work best, but needs to take care near sky
    // TODO: also probably needs to be rgb :P
    // dev.rgb = sqrt(dev_sum);

    // Temporal variance estimate with spatial colors
    // dev.rgb *= center_dev / max(1e-8, working_luma(dev.rgb));

    f32vec3 hist_dev = f32vec3(sqrt(abs(hist_vsum2 - hist_vsum * hist_vsum)));
    // dev.rgb *= 0.1 / max(1e-5, clamp(hist_dev, dev.rgb * 0.1, dev.rgb * 10.0));

    // float temporal_change = abs(hist_vsum - working_luma(ex.rgb)) / max(1e-8, hist_vsum + working_luma(ex.rgb));
    float temporal_change = abs(hist_vsum - working_luma(ex.rgb)) / max(1e-8, hist_vsum + working_luma(ex.rgb));
    // float temporal_change = 0.1 * abs(hist_vsum - working_luma(ex.rgb)) / max(1e-5, working_luma(dev.rgb));
    // temporal_change = 0.02 * temporal_change / max(1e-5, working_luma(dev.rgb));
    // temporal_change = WaveActiveSum(temporal_change) / WaveActiveSum(1);
#endif

    const float rt_invalid = saturate(sqrt(texelFetch(daxa_texture2D(rt_history_invalidity_tex), i32vec2(px / 2), 0).x) * 4);
    const float current_sample_count = history.a;

    float clamp_box_size = 1 * mix(0.25, 2.0, 1.0 - rt_invalid) * mix(0.333, 1.0, saturate(reproj.w)) * 2;
    clamp_box_size = max(clamp_box_size, 0.5);

    f32vec4 nmin = center - dev * clamp_box_size;
    f32vec4 nmax = center + dev * clamp_box_size;

#if 0
    {
    	f32vec4 nmin2 = center;
    	f32vec4 nmax2 = center;

    	{const int k = 2;
        for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                f32vec4 neigh = linear_to_working(texelFetch(daxa_texture2D(input_tex), i32vec2(px + i32vec2(x, y)), 0));
    			nmin2 = min(nmin2, neigh);
                nmax2 = max(nmax2, neigh);
            }
        }}

        f32vec3 nmid = mix(nmin2.rgb, nmax2.rgb, 0.5);
        nmin2.rgb = mix(nmid, nmin2.rgb, 1.0);
        nmax2.rgb = mix(nmid, nmax2.rgb, 1.0);

        nmin = max(nmin, nmin2);
        nmax = min(nmax, nmax2);
    }
#endif

#if 1
    f32vec4 clamped_history = f32vec4(clamp(history.rgb, nmin.rgb, nmax.rgb), history.a);
#else
    f32vec4 clamped_history = f32vec4(
        soft_color_clamp(center.rgb, history.rgb, ex.rgb, clamp_box_size * dev.rgb),
        history.a);
#endif

    /*const f32vec3 history_dist = abs(history.rgb - ex.rgb) / max(0.1, dev.rgb * 0.5);
    const f32vec3 closest_pt = clamp(history.rgb, center.rgb - dev.rgb * 0.5, center.rgb + dev.rgb * 0.5);
    clamped_history = f32vec4(
        mix(history.rgb, closest_pt, mix(0.1, 1.0, smoothstep(1.0, 3.0, history_dist))),
        history.a
    );*/

#if !USE_BBOX_CLAMP
    clamped_history = history;
#endif

    const float variance_adjusted_temporal_change = smoothstep(0.1, 1.0, 0.05 * temporal_change / center_temporal_dev);

    float max_sample_count = 32;
    max_sample_count = mix(max_sample_count, 4, variance_adjusted_temporal_change);
    // max_sample_count = mix(max_sample_count, 1, smoothstep(0.01, 0.6, 10 * temporal_change * (center_dev / max(1e-5, center_luma))));
    max_sample_count *= mix(1.0, 0.5, rt_invalid);

    // hax
    // max_sample_count = 32;

    f32vec3 res = mix(clamped_history.rgb, center.rgb, 1.0 / (1.0 + min(max_sample_count, current_sample_count)));
    // f32vec3 res = mix(clamped_history.rgb, center.rgb, 1.0 / 32);

    const float output_sample_count = min(current_sample_count, max_sample_count) + 1;
    f32vec4 output_ = working_to_linear(f32vec4(res, output_sample_count));
    imageStore(daxa_image2D(history_output_tex), i32vec2(px), output_);

    // output_ = smoothstep(1.0, 3.0, history_dist);
    // output_ = abs(history.rgb - ex.rgb);
    // output_ = dev.rgb;

    // output_ *= reproj.z;    // debug validity
    // output_ *= light_stability;
    // output_ = smoothstep(0.0, 0.05, history_dist);
    // output_ = length(dev.rgb);
    // output_ = 1-light_stability;
    // output_ = control_variate_luma;
    // output_ = abs(rdiff);
    // output_ = abs(dev.rgb);
    // output_ = abs(hist_dev.rgb);
    // output_ = smoothed_dev;

    // TODO: adaptively sample according to abs(res)
    // output_ = abs(res);
    // output_ = WaveActiveSum(center.rgb) / WaveActiveSum(1);
    // output_ = WaveActiveSum(history.rgb) / WaveActiveSum(1);
    // output_.rgb = 0.1 * temporal_change / max(1e-5, working_luma(dev.rgb));
    // output_ = pow(smoothstep(0.1, 1, temporal_change), 1.0);
    // output_.rgb = center_temporal_dev;
    // output_ = center_dev / max(1e-5, center_luma);
    // output_ = 1 - smoothstep(0.01, 0.6, temporal_change);
    // output_ = pow(smoothstep(0.02, 0.6, 0.01 * temporal_change / center_temporal_dev), 0.25);
    // output_ = max_sample_count / 32.0;
    // output_.rgb = temporal_change * 0.1;
    // output_.rgb = variance_adjusted_temporal_change * 0.1;
    // output_.rgb = texelFetch(daxa_texture2D(rt_history_invalidity_tex), i32vec2(px / 2), 0);
    // output_.rgb = mix(output_.rgb, rt_invalid, 0.9);
    // output_.rgb = mix(output_.rgb, pow(output_sample_count / 32.0, 4), 0.9);
    // output_.r = 1-reproj.w;

    float temp_var = saturate(
        output_sample_count * mix(1.0, 0.5, rt_invalid) * smoothstep(0.3, 0, temporal_change) / 32.0);

    imageStore(daxa_image2D(output_tex), i32vec2(px), f32vec4(output_.rgb, temp_var));

    // imageStore(daxa_image2D(output_tex), i32vec2(px), f32vec4(output_.rgb, output_sample_count));
    // imageStore(daxa_image2D(output_tex), i32vec2(px), f32vec4(output_.rgb, 1.0 - rt_invalid));
}
#endif

#if RTDGI_SPATIAL_COMPUTE
DAXA_DECL_PUSH_CONSTANT(RtdgiPush, push)

float square(float x) { return x * x; }
float max_3(float x, float y, float z) { return max(x, max(y, z)); }
float rcp(float x) { return 1.0 / x; }

// Bias towards dimmer input -- we don't care about energy loss here
// since this does not feed into subsequent passes, but want to minimize noise.
//
// https://gpuopen.com/learn/optimized-reversible-tonemapper-for-resolve/
f32vec3 crunch(f32vec3 v) {
    return v * rcp(max_3(v.r, v.g, v.b) + 1.0);
}
f32vec3 uncrunch(f32vec3 v) {
    return v * rcp(1.0 - max_3(v.r, v.g, v.b));
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 px = gl_GlobalInvocationID.xy;
#if 0
        imageStore(daxa_image2D(output_tex), i32vec2(px), texelFetch(daxa_texture2D(input_tex), i32vec2(px), 0));
        return;
#endif

    f32vec4 sum = f32vec4(0);

    const float center_validity = texelFetch(daxa_texture2D(input_tex), i32vec2(px), 0).a;
    const float center_depth = texelFetch(daxa_texture2D(depth_tex), i32vec2(px), 0).r;
    const float center_ssao = texelFetch(daxa_texture2D(ssao_tex), i32vec2(px), 0).r;
    const f32vec3 center_value = texelFetch(daxa_texture2D(input_tex), i32vec2(px), 0).rgb;
    const f32vec3 center_normal_vs = texelFetch(daxa_texture2D(geometric_normal_tex), i32vec2(px), 0).xyz * 2.0 - 1.0;

    if (center_validity == 1) {
        imageStore(daxa_image2D(output_tex), i32vec2(px), f32vec4(center_value, 1.0));
        return;
    }

    const float ang_off = (deref(gpu_input).frame_index * 23) % 32 * M_TAU + interleaved_gradient_noise(px) * M_PI;

    const uint MAX_SAMPLE_COUNT = 8;
    const float MAX_RADIUS_PX = sqrt(mix(16.0 * 16.0, 2.0 * 2.0, center_validity));

    // Feeds into the `pow` to remap sample index to radius.
    // At 0.5 (sqrt), it's proper circle sampling, with higher values becoming conical.
    // Must be constants, so the `pow` can be const-folded.
    const float KERNEL_SHARPNESS = 0.666;

    const uint sample_count = clamp(uint(exp2(4.0 * square(1.0 - center_validity))), 2, MAX_SAMPLE_COUNT);

    {
        sum += f32vec4(crunch(center_value), 1);

        const float RADIUS_SAMPLE_MULT = MAX_RADIUS_PX / pow(float(MAX_SAMPLE_COUNT - 1), KERNEL_SHARPNESS);

        // Note: faster on RTX2080 than a dynamic loop
        for (uint sample_i = 1; sample_i < MAX_SAMPLE_COUNT; ++sample_i) {
            const float ang = (sample_i + ang_off) * GOLDEN_ANGLE;

            float radius = pow(float(sample_i), KERNEL_SHARPNESS) * RADIUS_SAMPLE_MULT;
            f32vec2 sample_offset = f32vec2(cos(ang), sin(ang)) * radius;
            const i32vec2 sample_px = i32vec2(f32vec2(px) + sample_offset);

            const float sample_depth = texelFetch(daxa_texture2D(depth_tex), i32vec2(sample_px), 0).r;
            const f32vec3 sample_val = texelFetch(daxa_texture2D(input_tex), i32vec2(sample_px), 0).rgb;
            const float sample_ssao = texelFetch(daxa_texture2D(ssao_tex), i32vec2(sample_px), 0).r;
            const f32vec3 sample_normal_vs = texelFetch(daxa_texture2D(geometric_normal_tex), i32vec2(sample_px), 0).xyz * 2.0 - 1.0;

            if (sample_depth != 0 && sample_i < sample_count) {
                float wt = 1;
                // wt *= pow(saturate(dot(center_normal_vs, sample_normal_vs)), 20);
                wt *= exp2(-100.0 * abs(center_normal_vs.z * (center_depth / sample_depth - 1.0)));

#if USE_SSAO_STEERING
                wt *= exp2(-20.0 * abs(sample_ssao - center_ssao));
#endif

                sum += f32vec4(crunch(sample_val), 1.0) * wt;
            }
        }
    }

    float norm_factor = 1.0 / max(1e-5, sum.a);
    f32vec3 filtered = uncrunch(sum.rgb * norm_factor);

    imageStore(daxa_image2D(output_tex), i32vec2(px), f32vec4(filtered, 1.0));
}
#endif

#if RTDGI_REPROJECT_COMPUTE
DAXA_DECL_PUSH_CONSTANT(RtdgiPush, push)

// For `image_sample_catmull_rom`. Not applying the actual color remap here to reduce cost.
// struct HistoryRemap {
// };
// static HistoryRemap create() {
//     HistoryRemap res;
//     return res;
// }

f32vec4 HistoryRemap_remap(f32vec4 v) {
    return v;
}

f32vec4 cubic_hermite(f32vec4 A, f32vec4 B, f32vec4 C, f32vec4 D, float t) {
    float t2 = t * t;
    float t3 = t * t * t;
    f32vec4 a = -A / 2.0 + (3.0 * B) / 2.0 - (3.0 * C) / 2.0 + D / 2.0;
    f32vec4 b = A - (5.0 * B) / 2.0 + 2.0 * C - D / 2.0;
    f32vec4 c = -A / 2.0 + C / 2.0;
    f32vec4 d = B;

    return a * t3 + b * t2 + c * t + d;
}

#define REMAP_FUNC HistoryRemap_remap
f32vec4 image_sample_catmull_rom(daxa_ImageViewId img, f32vec2 P, f32vec4 img_size) {
    // https://www.shadertoy.com/view/MllSzX

    f32vec2 pixel = P * img_size.xy + 0.5;
    f32vec2 c_onePixel = img_size.zw;
    f32vec2 c_twoPixels = c_onePixel * 2.0;

    f32vec2 frc = fract(pixel);
    // pixel = floor(pixel) / output_tex_size.xy - f32vec2(c_onePixel/2.0);
    i32vec2 ipixel = i32vec2(pixel) - 1;

    f32vec4 C00 = REMAP_FUNC(texelFetch(daxa_texture2D(img), ipixel + i32vec2(-1, -1), 0));
    f32vec4 C10 = REMAP_FUNC(texelFetch(daxa_texture2D(img), ipixel + i32vec2(0, -1), 0));
    f32vec4 C20 = REMAP_FUNC(texelFetch(daxa_texture2D(img), ipixel + i32vec2(1, -1), 0));
    f32vec4 C30 = REMAP_FUNC(texelFetch(daxa_texture2D(img), ipixel + i32vec2(2, -1), 0));

    f32vec4 C01 = REMAP_FUNC(texelFetch(daxa_texture2D(img), ipixel + i32vec2(-1, 0), 0));
    f32vec4 C11 = REMAP_FUNC(texelFetch(daxa_texture2D(img), ipixel + i32vec2(0, 0), 0));
    f32vec4 C21 = REMAP_FUNC(texelFetch(daxa_texture2D(img), ipixel + i32vec2(1, 0), 0));
    f32vec4 C31 = REMAP_FUNC(texelFetch(daxa_texture2D(img), ipixel + i32vec2(2, 0), 0));

    f32vec4 C02 = REMAP_FUNC(texelFetch(daxa_texture2D(img), ipixel + i32vec2(-1, 1), 0));
    f32vec4 C12 = REMAP_FUNC(texelFetch(daxa_texture2D(img), ipixel + i32vec2(0, 1), 0));
    f32vec4 C22 = REMAP_FUNC(texelFetch(daxa_texture2D(img), ipixel + i32vec2(1, 1), 0));
    f32vec4 C32 = REMAP_FUNC(texelFetch(daxa_texture2D(img), ipixel + i32vec2(2, 1), 0));

    f32vec4 C03 = REMAP_FUNC(texelFetch(daxa_texture2D(img), ipixel + i32vec2(-1, 2), 0));
    f32vec4 C13 = REMAP_FUNC(texelFetch(daxa_texture2D(img), ipixel + i32vec2(0, 2), 0));
    f32vec4 C23 = REMAP_FUNC(texelFetch(daxa_texture2D(img), ipixel + i32vec2(1, 2), 0));
    f32vec4 C33 = REMAP_FUNC(texelFetch(daxa_texture2D(img), ipixel + i32vec2(2, 2), 0));

    f32vec4 CP0X = cubic_hermite(C00, C10, C20, C30, frc.x);
    f32vec4 CP1X = cubic_hermite(C01, C11, C21, C31, frc.x);
    f32vec4 CP2X = cubic_hermite(C02, C12, C22, C32, frc.x);
    f32vec4 CP3X = cubic_hermite(C03, C13, C23, C33, frc.x);

    return cubic_hermite(CP0X, CP1X, CP2X, CP3X, frc.y);
}
#undef REMAP_FUNC

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 px = gl_GlobalInvocationID.xy;
    f32vec2 uv = get_uv(px, push.output_tex_size);

    f32vec4 center = texelFetch(daxa_texture2D(input_tex), i32vec2(px), 0);
    f32vec4 reproj = texelFetch(daxa_texture2D(reprojection_tex), i32vec2(px), 0);
    f32vec2 prev_uv = uv + reproj.xy;

    uint quad_reproj_valid_packed = uint(reproj.z * 15.0 + 0.5);

    // For the sharpening (4x4) kernel, we need to know whether our neighbors are valid too,
    // as otherwise we end up over-sharpening with fake history (moving edges rather than scene features).
    f32vec2 uv_offset = 0.5 * sign(prev_uv) * push.output_tex_size.zw;
    const u32vec4 reproj_valid_neigh =
        u32vec4(textureGather(daxa_sampler2D(reprojection_tex, deref(gpu_input).sampler_nnc), uv, 2) * 15.0 + 0.5);

    f32vec4 history = 0.0.xxxx;

    if (0 == quad_reproj_valid_packed) {
        // Everything invalid
    } else if (15 == quad_reproj_valid_packed) {
        if (USE_SHARPENING_HISTORY_FETCH && all(bvec4(reproj_valid_neigh == u32vec4(15)))) {
            history = max(0.0.xxxx, image_sample_catmull_rom(
                                        input_tex,
                                        prev_uv,
                                        push.output_tex_size));
        } else {
            history = textureLod(daxa_sampler2D(input_tex, deref(gpu_input).sampler_lnc), prev_uv, 0);
        }
    } else {
        // Only some samples are valid. Only include those, and don't do a sharpening fetch here.

        f32vec4 quad_reproj_valid = f32vec4((quad_reproj_valid_packed & u32vec4(1, 2, 4, 8)) != f32vec4(0.0));

        const Bilinear bilinear = get_bilinear_filter(prev_uv, push.output_tex_size.xy);
        f32vec4 s00 = texelFetch(daxa_texture2D(input_tex), i32vec2(i32vec2(bilinear.origin) + i32vec2(0, 0)), 0);
        f32vec4 s10 = texelFetch(daxa_texture2D(input_tex), i32vec2(i32vec2(bilinear.origin) + i32vec2(1, 0)), 0);
        f32vec4 s01 = texelFetch(daxa_texture2D(input_tex), i32vec2(i32vec2(bilinear.origin) + i32vec2(0, 1)), 0);
        f32vec4 s11 = texelFetch(daxa_texture2D(input_tex), i32vec2(i32vec2(bilinear.origin) + i32vec2(1, 1)), 0);

        f32vec4 weights = get_bilinear_custom_weights(bilinear, quad_reproj_valid);

        if (dot(weights, f32vec4(1.0)) > 1e-5) {
            history = apply_bilinear_custom_weights(s00, s10, s01, s11, weights, true);
        }
    }

    imageStore(daxa_image2D(output_tex), i32vec2(px), history);
}
#endif

#if RTDGI_VALIDATE_COMPUTE
DAXA_DECL_PUSH_CONSTANT(RtdgiRestirTemporalPush, push)

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 px = gl_GlobalInvocationID.xy;
    const u32vec2 HALFRES_SUBSAMPLE_OFFSET = get_downscale_offset(gpu_input);
    const i32vec2 hi_px_offset = i32vec2(HALFRES_SUBSAMPLE_OFFSET);
    const u32vec2 hi_px = px * SHADING_SCL + hi_px_offset;

    if (0.0 == texelFetch(daxa_texture2D(depth_tex), i32vec2(hi_px), 0).r) {
        imageStore(daxa_image2D(rt_history_invalidity_out_tex), i32vec2(px), f32vec4(1.0));
        return;
    }

    float invalidity = 0.0;

    if (RESTIR_USE_PATH_VALIDATION && is_rtdgi_validation_frame()) {
        const f32vec3 normal_vs = texelFetch(daxa_texture2D(half_view_normal_tex), i32vec2(px), 0).xyz;
        const f32vec3 normal_ws = direction_view_to_world(globals, normal_vs);

        const f32vec3 prev_ray_orig = texelFetch(daxa_texture2D(ray_orig_history_tex), i32vec2(px), 0).xyz;
        const f32vec3 prev_hit_pos = texelFetch(daxa_texture2D(reservoir_ray_history_tex), i32vec2(px), 0).xyz + prev_ray_orig;

        const f32vec4 prev_radiance_packed = texelFetch(daxa_texture2D(irradiance_history_tex), i32vec2(px), 0);
        const f32vec3 prev_radiance = max(0.0.xxx, prev_radiance_packed.rgb);

        RayDesc prev_ray;
        prev_ray.Direction = normalize(prev_hit_pos - prev_ray_orig);
        prev_ray.Origin = prev_ray_orig;
        prev_ray.TMin = 0;
        prev_ray.TMax = SKY_DIST;

        // TODO: frame index
        uint rng = hash3(u32vec3(px, 0));

        TraceResult result = do_the_thing(px, normal_ws, rng, prev_ray);
        const f32vec3 new_radiance = max(0.0.xxx, result.out_value);

        const float rad_diff = length(abs(prev_radiance - new_radiance) / max(f32vec3(1e-3), prev_radiance + new_radiance));
        invalidity = smoothstep(0.1, 0.5, rad_diff / length(1.0.xxx));

        const float prev_hit_dist = length(prev_hit_pos - prev_ray_orig);

        // If we hit more or less the same point, replace the hit radiance.
        // If the hit is different, it's possible that the previous origin point got obscured
        // by something, in which case we want M-clamping to take care of it instead.
        if (abs(result.hit_t - prev_hit_dist) / (prev_hit_dist + prev_hit_dist) < 0.2) {
            imageStore(daxa_image2D(irradiance_history_tex), i32vec2(px), f32vec4(new_radiance, prev_radiance_packed.a));

            // When we update the radiance, we might end up with fairly low probability
            // rays hitting the bright spots by chance. The PDF division compounded
            // by the increase in radiance causes fireflies to appear.
            // As a HACK, we will clamp that by scaling down the `M` factor then.
            Reservoir1spp r = Reservoir1spp_from_raw(texelFetch(daxa_utexture2D(reservoir_tex), i32vec2(px), 0).xy);
            const float lum_old = sRGB_to_luminance(prev_radiance);
            const float lum_new = sRGB_to_luminance(new_radiance);
            r.M *= clamp(lum_old / max(1e-8, lum_new), 0.03, 1.0);

            // Allow the new value to be greater than the old one up to a certain scale,
            // then dim it down by reducing `W`. It will recover over time.
            const float allowed_luminance_increment = 10.0;
            r.W *= clamp(lum_old / max(1e-8, lum_new) * allowed_luminance_increment, 0.01, 1.0);

            imageStore(daxa_uimage2D(reservoir_tex), i32vec2(px), u32vec4(Reservoir1spp_as_raw(r), 0, 0));
        }
    }

    imageStore(daxa_image2D(rt_history_invalidity_out_tex), i32vec2(px), f32vec4(invalidity, 0, 0, 0));
}
#endif

#if RTDGI_TRACE_COMPUTE

DAXA_DECL_PUSH_CONSTANT(RtdgiRestirTemporalPush, push)

f32vec3 rtdgi_candidate_ray_dir(u32vec2 px, f32mat3x3 tangent_to_world) {
    f32vec2 urand = blue_noise_for_pixel(blue_noise_vec2, px, deref(gpu_input).frame_index).xy;
    f32vec3 wi = uniform_sample_hemisphere(urand);
    return tangent_to_world * wi;
}

void rtdgi_trace_job(u32vec2 px) {
    const u32vec2 HALFRES_SUBSAMPLE_OFFSET = get_downscale_offset(gpu_input);
    const i32vec2 hi_px_offset = i32vec2(HALFRES_SUBSAMPLE_OFFSET);
    const u32vec2 hi_px = px * SHADING_SCL + hi_px_offset;

    float depth = texelFetch(daxa_texture2D(depth_tex), i32vec2(hi_px), 0).r;

    if (0.0 == depth) {
        imageStore(daxa_image2D(candidate_irradiance_out_tex), i32vec2(px), f32vec4(0));
        imageStore(daxa_image2D(candidate_normal_out_tex), i32vec2(px), f32vec4(0, 0, 1, 0));
        imageStore(daxa_image2D(rt_history_invalidity_out_tex), i32vec2(px), f32vec4(0));
        return;
    }

    const f32vec2 uv = get_uv(hi_px, push.gbuffer_tex_size);
    const ViewRayContext view_ray_context = vrc_from_uv_and_biased_depth(globals, uv_to_ss(gpu_input, uv, push.gbuffer_tex_size), depth);

    const float NEAR_FIELD_FADE_OUT_END = -ray_hit_vs(view_ray_context).z * (SSGI_NEAR_FIELD_RADIUS * push.gbuffer_tex_size.w * 0.5);

    // #if RTDGI_INTERLEAVED_VALIDATION_ALWAYS_TRACE_NEAR_FIELD
    //     if (true) {
    // #else
    //     if (is_rtdgi_tracing_frame()) {
    // #endif
    {
        const f32vec3 normal_vs = texelFetch(daxa_texture2D(half_view_normal_tex), i32vec2(px), 0).xyz;
        const f32vec3 normal_ws = direction_view_to_world(globals, normal_vs);
        const f32mat3x3 tangent_to_world = tbn_from_normal(normal_ws);
        const f32vec3 outgoing_dir = rtdgi_candidate_ray_dir(px, tangent_to_world);

        RayDesc outgoing_ray;
        outgoing_ray.Direction = outgoing_dir;
        outgoing_ray.Origin = biased_secondary_ray_origin_ws_with_normal(view_ray_context, normal_ws);
        outgoing_ray.TMin = 0;

        if (is_rtdgi_tracing_frame()) {
            outgoing_ray.TMax = SKY_DIST;
        } else {
            outgoing_ray.TMax = NEAR_FIELD_FADE_OUT_END;
        }

        uint rng = hash3(u32vec3(px, deref(gpu_input).frame_index & 31));
        TraceResult result = do_the_thing(px, normal_ws, rng, outgoing_ray);

#if RTDGI_INTERLEAVED_VALIDATION_ALWAYS_TRACE_NEAR_FIELD
        if (!is_rtdgi_tracing_frame() && !result.is_hit) {
            // If we were only tracing short rays, make sure we don't try to output
            // sky color upon misses.
            result.out_value = f32vec3(0.0);
            result.hit_t = SKY_DIST;
        }
#endif

        const f32vec3 hit_offset_ws = outgoing_ray.Direction * result.hit_t;

        const float cos_theta = dot(normalize(outgoing_dir - ray_dir_ws(view_ray_context)), normal_ws);
        imageStore(daxa_image2D(candidate_irradiance_out_tex), i32vec2(px), f32vec4(result.out_value, rtr_encode_cos_theta_for_fp16(cos_theta)));
        imageStore(daxa_image2D(candidate_hit_out_tex), i32vec2(px), f32vec4(hit_offset_ws, result.pdf * select(is_rtdgi_tracing_frame(), 1, -1)));
        imageStore(daxa_image2D(candidate_normal_out_tex), i32vec2(px), f32vec4(direction_world_to_view(globals, result.hit_normal_ws), 0));
    }
    //  else {
    //     const f32vec4 reproj = texelFetch(daxa_texture2D(reprojection_tex), i32vec2(hi_px), 0);
    //     const i32vec2 reproj_px = floor(px + gbuffer_tex_size.xy * reproj.xy / 2 + 0.5);
    //     imageStore(daxa_image2D(candidate_irradiance_out_tex), i32vec2(px), 0.0);
    //     imageStore(daxa_image2D(candidate_hit_out_tex), i32vec2(px), 0.0);
    //     imageStore(daxa_image2D(candidate_normal_out_tex), i32vec2(px), 0.0);
    // }

    const f32vec4 reproj = texelFetch(daxa_texture2D(reprojection_tex), i32vec2(hi_px), 0);
    const i32vec2 reproj_px = i32vec2(floor(f32vec2(px) + push.gbuffer_tex_size.xy * reproj.xy / 2.0 + 0.5));
    imageStore(daxa_image2D(rt_history_invalidity_out_tex), i32vec2(px), texelFetch(daxa_texture2D(rt_history_invalidity_in_tex), i32vec2(reproj_px), 0));
}

#define ONE_JOB_PER_THREAD 1

#if ONE_JOB_PER_THREAD
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 px = gl_GlobalInvocationID.xy;
    rtdgi_trace_job(px);
}
#elif FOUR_JOBS_PER_THREAD
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 px = gl_GlobalInvocationID.xy;
    for (u32 yi = 0; yi < 2; ++yi) {
        for (u32 xi = 0; xi < 2; ++xi) {
            rtdgi_trace_job(px * 2 + u32vec2(xi, yi));
        }
    }
}
#elif VARIABLE_JOBS_PER_THREAD
shared u32 job_counter;
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    if (gl_LocalInvocationIndex == 0) {
        job_counter = 0;
    }
    barrier();
    memoryBarrierShared();
    while (true) {
        u32 job_index = atomicAdd(job_counter, 1);
        if (job_index >= 256) {
            break;
        }
        u32vec2 px = gl_WorkGroupID.xy * 16 + u32vec2(job_index % 16, job_index / 16);
        rtdgi_trace_job(px);
    }
}
#endif

#endif

#if RTDGI_VALIDITY_INTEGRATE_COMPUTE
DAXA_DECL_PUSH_CONSTANT(RtdgiRestirResolvePush, push)

#define USE_SSAO_STEERING 1
#define USE_DYNAMIC_KERNEL_RADIUS 0

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 px = gl_GlobalInvocationID.xy;
    f32vec2 invalid_blurred = f32vec2(0);

    const float center_depth = texelFetch(daxa_texture2D(half_depth_tex), i32vec2(px), 0).r;
    const f32vec4 reproj = texelFetch(daxa_texture2D(reprojection_tex), i32vec2(px * 2), 0);

    if (RESTIR_USE_PATH_VALIDATION) {
        {
            const int k = 2;
            for (int y = -k; y <= k; ++y) {
                for (int x = -k; x <= k; ++x) {
                    const i32vec2 offset = i32vec2(x, y);
                    // float w = 1;
                    float w = exp2(-0.1 * dot(f32vec2(offset), f32vec2(offset)));
                    invalid_blurred += f32vec2(texelFetch(daxa_texture2D(input_tex), i32vec2(px + offset), 0).x, 1) * w;
                }
            }
        }
        invalid_blurred /= invalid_blurred.y;
        invalid_blurred.x = mix(invalid_blurred.x, subgroupBroadcast(invalid_blurred.x, gl_SubgroupInvocationID ^ 2), 0.5);
        invalid_blurred.x = mix(invalid_blurred.x, subgroupBroadcast(invalid_blurred.x, gl_SubgroupInvocationID ^ 16), 0.5);
        // invalid_blurred.x = mix(invalid_blurred.x, WaveActiveSum(invalid_blurred.x) / 64.0, 0.25);

        const f32vec2 reproj_rand_offset = f32vec2(0.0);

        invalid_blurred.x = smoothstep(0.0, 1.0, invalid_blurred.x);
    }

    /*if (reproj.z == 0) {
        invalid_blurred.x += 1;
    }*/

    float edge = 1;

    {
        const int k = 2;
        for (int y = 0; y <= k; ++y) {
            for (int x = 1; x <= k; ++x) {
                const i32vec2 offset = i32vec2(x, y);
                const i32vec2 sample_px = i32vec2(px) * 2 + offset;
                const i32vec2 sample_px_half = i32vec2(px) + offset / 2;
                const f32vec4 reproj = texelFetch(daxa_texture2D(reprojection_tex), i32vec2(sample_px), 0);
                const float sample_depth = texelFetch(daxa_texture2D(half_depth_tex), i32vec2(sample_px_half), 0).r;

                if (reproj.w < 0 || inverse_depth_relative_diff(center_depth, sample_depth) > 0.1) {
                    edge = 0;
                    break;
                }

                edge *= float(reproj.z == 0 && sample_depth != 0);
            }
        }
    }

    edge = max(edge, subgroupBroadcast(edge, gl_SubgroupInvocationID ^ 1));
    edge = max(edge, subgroupBroadcast(edge, gl_SubgroupInvocationID ^ 8));
    /*edge = max(edge, WaveReadLaneAt(edge, WaveGetLaneIndex() ^ 4));
    edge = max(edge, WaveReadLaneAt(edge, WaveGetLaneIndex() ^ 32));*/

    invalid_blurred.x += edge;

    invalid_blurred = saturate(invalid_blurred);

    // invalid_blurred.x = smoothstep(0.1, 1.0, invalid_blurred.x);
    // invalid_blurred.x = pow(invalid_blurred.x, 4);

    // invalid_blurred.x = 1;;

    const f32vec2 reproj_px = px + push.gbuffer_tex_size.xy * reproj.xy / 2 + 0.5;
    float history = 0;

    const int sample_count = 8;
    float ang_off = uint_to_u01_float(hash3(u32vec3(px, deref(gpu_input).frame_index))) * M_PI * 2;

    for (uint sample_i = 0; sample_i < sample_count; ++sample_i) {
        float ang = (sample_i + ang_off) * GOLDEN_ANGLE;
        float radius = float(sample_i) * 1.0;
        f32vec2 sample_offset = f32vec2(cos(ang), sin(ang)) * radius;
        const i32vec2 sample_px = i32vec2(reproj_px + sample_offset);
        history += texelFetch(daxa_texture2D(history_tex), i32vec2(sample_px), 0).x;
    }

    history /= sample_count;

    /*float history = (
        texelFetch(daxa_texture2D(history_tex), i32vec2(reproj_px), 0) +
        texelFetch(daxa_texture2D(history_tex), i32vec2(reproj_px + i32vec2(-4, 0)), 0) +
        texelFetch(daxa_texture2D(history_tex), i32vec2(reproj_px + i32vec2(4, 0)), 0) +
        texelFetch(daxa_texture2D(history_tex), i32vec2(reproj_px + i32vec2(0, 4)), 0) +
        texelFetch(daxa_texture2D(history_tex), i32vec2(reproj_px + i32vec2(0, -4)), 0)
    ) / 5;*/

    // float history = texelFetch(daxa_texture2D(history_tex), i32vec2(reproj_px), 0);

    imageStore(daxa_image2D(output_tex), i32vec2(px), f32vec4(max(history * 0.75, invalid_blurred.x),
                                                              // invalid_blurred.x,
                                                              texelFetch(daxa_texture2D(input_tex), i32vec2(px), 0).x, 0.0, 0.0));
}
#endif

#if RTDGI_RESTIR_TEMPORAL_COMPUTE
DAXA_DECL_PUSH_CONSTANT(RtdgiRestirTemporalPush, push)

u32vec2 reservoir_payload_to_px(uint payload) {
    return u32vec2(payload & 0xffff, payload >> 16);
}

struct TraceResult {
    f32vec3 out_value;
    f32vec3 hit_normal_ws;
    float inv_pdf;
    // bool prev_sample_valid;
};

TraceResult do_the_thing(u32vec2 px, inout uint rng, RayDesc outgoing_ray, f32vec3 primary_hit_normal) {
    const f32vec4 candidate_radiance_inv_pdf = texelFetch(daxa_texture2D(candidate_radiance_tex), i32vec2(px), 0);
    TraceResult result;
    result.out_value = candidate_radiance_inv_pdf.rgb;
    result.inv_pdf = 1;
    result.hit_normal_ws = direction_view_to_world(globals, texelFetch(daxa_texture2D(candidate_normal_tex), i32vec2(px), 0).xyz);
    return result;
}

i32vec2 get_rpx_offset(uint sample_i, uint frame_index) {
    const i32vec2 offsets[4] = {
        i32vec2(-1, -1),
        i32vec2(1, 1),
        i32vec2(-1, 1),
        i32vec2(1, -1),
    };

    const i32vec2 reservoir_px_offset_base =
        offsets[frame_index & 3] + offsets[(sample_i + (frame_index ^ 1)) & 3];

    return select(sample_i == 0, i32vec2(0), i32vec2(reservoir_px_offset_base));
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 px = gl_GlobalInvocationID.xy;
    const i32vec2 hi_px_offset = i32vec2(get_downscale_offset(gpu_input));
    const u32vec2 hi_px = px * 2 + hi_px_offset;

    float depth = texelFetch(daxa_texture2D(depth_tex), i32vec2(hi_px), 0).r;

    if (0.0 == depth) {
        imageStore(daxa_image2D(radiance_out_tex), i32vec2(px), f32vec4(0.0.xxx, -SKY_DIST));
        imageStore(daxa_image2D(hit_normal_output_tex), i32vec2(px), 0.0.xxxx);
        imageStore(daxa_uimage2D(reservoir_out_tex), i32vec2(px), u32vec4(0));
        return;
    }

    const f32vec2 uv = get_uv(hi_px, push.gbuffer_tex_size);
    const ViewRayContext view_ray_context = vrc_from_uv_and_biased_depth(globals, uv, depth);
    const f32vec3 normal_vs = texelFetch(daxa_texture2D(half_view_normal_tex), i32vec2(px), 0).xyz;
    const f32vec3 normal_ws = direction_view_to_world(globals, normal_vs);
    const f32mat3x3 tangent_to_world = tbn_from_normal(normal_ws);
    const f32vec3 refl_ray_origin_ws = biased_secondary_ray_origin_ws_with_normal(view_ray_context, normal_ws);

    const f32vec3 hit_offset_ws = texelFetch(daxa_texture2D(candidate_hit_tex), i32vec2(px), 0).xyz;
    f32vec3 outgoing_dir = normalize(hit_offset_ws);

    uint rng = hash3(u32vec3(px, deref(gpu_input).frame_index));

    u32vec2 src_px_sel = px;
    f32vec3 radiance_sel = f32vec3(0);
    f32vec3 ray_orig_sel_ws = f32vec3(0);
    f32vec3 ray_hit_sel_ws = f32vec3(1);
    f32vec3 hit_normal_sel = f32vec3(1);
    // bool prev_sample_valid = false;

    Reservoir1sppStreamState stream_state = Reservoir1sppStreamState_create();
    Reservoir1spp reservoir = Reservoir1spp_create();
    const uint reservoir_payload = px.x | (px.y << 16);

    if (is_rtdgi_tracing_frame()) {
        RayDesc outgoing_ray;
        outgoing_ray.Direction = outgoing_dir;
        outgoing_ray.Origin = refl_ray_origin_ws;
        outgoing_ray.TMin = 0;
        outgoing_ray.TMax = SKY_DIST;

        const float hit_t = length(hit_offset_ws);

        TraceResult result = do_the_thing(px, rng, outgoing_ray, normal_ws);

        /*if (USE_SPLIT_RT_NEAR_FIELD) {
            const float NEAR_FIELD_FADE_OUT_END = -view_ray_context.ray_hit_vs().z * (SSGI_NEAR_FIELD_RADIUS * push.gbuffer_tex_size.w * 0.5);
            const float NEAR_FIELD_FADE_OUT_START = NEAR_FIELD_FADE_OUT_END * 0.5;
            float infl = hit_t / (SSGI_NEAR_FIELD_RADIUS * push.gbuffer_tex_size.w * 0.5) / -view_ray_context.ray_hit_vs().z;
            result.out_value *= smoothstep(0.0, 1.0, infl);
        }*/

        const float p_q = 1.0 * max(0, sRGB_to_luminance(result.out_value))
                          // Note: using max(0, dot) reduces noise in easy areas,
                          // but then increases it in corners by undersampling grazing angles.
                          // Effectively over time the sampling turns cosine-distributed, which
                          // we avoided doing in the first place.
                          * step(0, dot(outgoing_dir, normal_ws));

        const float inv_pdf_q = result.inv_pdf;

        radiance_sel = result.out_value;
        ray_orig_sel_ws = outgoing_ray.Origin;
        ray_hit_sel_ws = outgoing_ray.Origin + outgoing_ray.Direction * hit_t;
        hit_normal_sel = result.hit_normal_ws;
        // prev_sample_valid = result.prev_sample_valid;

        Reservoir1spp_init_with_stream(reservoir, p_q, inv_pdf_q, stream_state, reservoir_payload);

        float rl = mix(texelFetch(daxa_texture2D(candidate_history_tex), i32vec2(px), 0).y, sqrt(hit_t), 0.05);
        imageStore(daxa_image2D(candidate_out_tex), i32vec2(px), f32vec4(sqrt(hit_t), rl, 0, 0));
    }

    const float rt_invalidity = sqrt(saturate(texelFetch(daxa_texture2D(rt_invalidity_tex), i32vec2(px), 0).y));

    const bool use_resampling = DIFFUSE_GI_USE_RESTIR != 0;
    // const bool use_resampling = false;

    // 1 (center) plus offset samples
    const uint MAX_RESOLVE_SAMPLE_COUNT =
        select(RESTIR_TEMPORAL_USE_PERMUTATIONS, 5, 1);

    float center_M = 0;

    if (use_resampling) {
        for (
            uint sample_i = 0;
            sample_i < MAX_RESOLVE_SAMPLE_COUNT
            // Use permutation sampling, but only up to a certain M; those are lower quality,
            // so we want to be rather conservative.
            && stream_state.M_sum < 1.25 * RESTIR_TEMPORAL_M_CLAMP;
            ++sample_i) {
            const i32vec2 rpx_offset = get_rpx_offset(sample_i, deref(gpu_input).frame_index);
            if (sample_i > 0 && all(bvec2(rpx_offset == i32vec2(0)))) {
                // No point using the center sample twice
                continue;
            }

            const f32vec4 reproj = texelFetch(daxa_texture2D(reprojection_tex), i32vec2(hi_px + rpx_offset * 2), 0);

            // Can't use linear interpolation, but we can interpolate stochastically instead
            // const f32vec2 reproj_rand_offset = f32vec2(uint_to_u01_float(hash1_mut(rng)), uint_to_u01_float(hash1_mut(rng))) - 0.5;
            // Or not at all.
            const f32vec2 reproj_rand_offset = f32vec2(0.0);

            const u32vec2 xor_seq[4] = {
                u32vec2(3, 3),
                u32vec2(2, 1),
                u32vec2(1, 2),
                u32vec2(3, 3),
            };
            const u32vec2 permutation_xor_val =
                xor_seq[deref(gpu_input).frame_index & 3];

            const i32vec2 permuted_reproj_px = i32vec2(floor(
                select(sample_i == 0, px
                       // My poor approximation of permutation sampling.
                       // https://twitter.com/more_fps/status/1457749362025459715
                       //
                       // When applied everywhere, it does nicely reduce noise, but also makes the GI less reactive
                       // since we're effectively increasing the lifetime of the most attractive samples.
                       // Where it does come in handy though is for boosting convergence rate for newly revealed
                       // locations.
                       ,
                       ((px + rpx_offset) ^ permutation_xor_val)) +
                push.gbuffer_tex_size.xy * reproj.xy * 0.5 + reproj_rand_offset + 0.5));

            const i32vec2 rpx = permuted_reproj_px + rpx_offset;
            const u32vec2 rpx_hi = rpx * 2 + hi_px_offset;

            const i32vec2 permuted_neighbor_px = i32vec2(floor(
                select(sample_i == 0, px
                       // ditto
                       ,
                       ((px + rpx_offset) ^ permutation_xor_val)) +
                0.5));

            const i32vec2 neighbor_px = permuted_neighbor_px + rpx_offset;
            const u32vec2 neighbor_px_hi = neighbor_px * 2 + hi_px_offset;

            // WRONG. needs previous normal
            // const f32vec3 sample_normal_vs = texelFetch(daxa_texture2D(half_view_normal_tex), i32vec2(rpx), 0);
            // // Note: also doing this for sample 0, as under extreme aliasing,
            // // we can easily get bad samples in.
            // if (dot(sample_normal_vs, normal_vs) < 0.7) {
            //     continue;
            // }

            Reservoir1spp r = Reservoir1spp_from_raw(texelFetch(daxa_utexture2D(reservoir_history_tex), i32vec2(rpx), 0).xy);
            const u32vec2 spx = reservoir_payload_to_px(r.payload);

            float visibility = 1;
            // float relevance = select(sample_i == 0, 1, 0.5);
            float relevance = 1;

            // const f32vec2 sample_uv = get_uv(rpx_hi, push.gbuffer_tex_size);
            const float sample_depth = texelFetch(daxa_texture2D(depth_tex), i32vec2(neighbor_px_hi), 0).r;

            // WRONG: needs previous depth
            // if (length(prev_ray_orig_and_dist.xyz - refl_ray_origin_ws) > 0.1 * -view_ray_context.ray_hit_vs().z) {
            //     // Reject disocclusions
            //     continue;
            // }

            const f32vec3 prev_ray_orig = texelFetch(daxa_texture2D(ray_orig_history_tex), i32vec2(spx), 0).xyz;
            if (length(prev_ray_orig - refl_ray_origin_ws) > 0.1 * -ray_hit_vs(view_ray_context).z) {
                // Reject disocclusions
                continue;
            }

            // Note: also doing this for sample 0, as under extreme aliasing,
            // we can easily get bad samples in.
            if (0 == sample_depth) {
                continue;
            }

            // TODO: some more rejection based on the reprojection map.
            // This one is not enough ("battle", buttom of tower).
            if (reproj.z == 0) {
                continue;
            }

#if 1
            relevance *= 1 - smoothstep(0.0, 0.1, inverse_depth_relative_diff(depth, sample_depth));
#else
            if (inverse_depth_relative_diff(depth, sample_depth) > 0.2) {
                continue;
            }
#endif

            const f32vec3 sample_normal_vs = texelFetch(daxa_texture2D(half_view_normal_tex), i32vec2(neighbor_px), 0).rgb;
            const float normal_similarity_dot = max(0.0, dot(sample_normal_vs, normal_vs));

// Increases noise, but prevents leaking in areas of geometric complexity
#if 1
            // High cutoff seems unnecessary. Had it at 0.9 before.
            const float normal_cutoff = 0.2;
            if (sample_i != 0 && normal_similarity_dot < normal_cutoff) {
                continue;
            }
#endif

            relevance *= pow(normal_similarity_dot, 4);

            // TODO: this needs fixing with reprojection
            // const ViewRayContext sample_ray_ctx = ViewRayContext::from_uv_and_depth(sample_uv, sample_depth);

            const f32vec4 sample_hit_ws_and_dist = texelFetch(daxa_texture2D(ray_history_tex), i32vec2(spx), 0) + f32vec4(prev_ray_orig, 0.0);
            const f32vec3 sample_hit_ws = sample_hit_ws_and_dist.xyz;
            // const f32vec3 prev_dir_to_sample_hit_unnorm_ws = sample_hit_ws - sample_ray_ctx.ray_hit_ws();
            // const f32vec3 prev_dir_to_sample_hit_ws = normalize(prev_dir_to_sample_hit_unnorm_ws);
            const float prev_dist = sample_hit_ws_and_dist.w;

            // Note: `hit_normal_history_tex` is not reprojected.
            const f32vec4 sample_hit_normal_ws_dot = decode_hit_normal_and_dot(texelFetch(daxa_texture2D(hit_normal_history_tex), i32vec2(spx), 0));

            /*if (sample_i > 0 && !(prev_dist > 1e-4)) {
                continue;
            }*/

            const f32vec3 dir_to_sample_hit_unnorm = sample_hit_ws - refl_ray_origin_ws;
            const float dist_to_sample_hit = length(dir_to_sample_hit_unnorm);
            const f32vec3 dir_to_sample_hit = normalize(dir_to_sample_hit_unnorm);

            const float center_to_hit_vis = -dot(sample_hit_normal_ws_dot.xyz, dir_to_sample_hit);
            // const float prev_to_hit_vis = -dot(sample_hit_normal_ws_dot.xyz, prev_dir_to_sample_hit_ws);

            const f32vec4 prev_rad =
                texelFetch(daxa_texture2D(radiance_history_tex), i32vec2(spx), 0) * f32vec4((FRAME_CONSTANTS_PRE_EXPOSURE_DELTA).xxx, 1);

            // From the ReSTIR paper:
            // With temporal reuse, the number of candidates M contributing to the
            // pixel can in theory grow unbounded, as each frame always combines
            // its reservoir with the previous frame’s. This causes (potentially stale)
            // temporal samples to be weighted disproportionately high during
            // resampling. To fix this, we simply clamp the previous frame’s M
            // to at most 20× of the current frame’s reservoir’s M

            r.M = max(0, min(r.M, exp2(log2(RESTIR_TEMPORAL_M_CLAMP) * (1.0 - rt_invalidity))));
            // r.M = min(r.M, RESTIR_TEMPORAL_M_CLAMP);
            // r.M = min(r.M, 0.1);

            const float p_q = 1 * max(0, sRGB_to_luminance(prev_rad.rgb))
                              // Note: using max(0, dot) reduces noise in easy areas,
                              // but then increases it in corners by undersampling grazing angles.
                              // Effectively over time the sampling turns cosine-distributed, which
                              // we avoided doing in the first place.
                              * step(0, dot(dir_to_sample_hit, normal_ws));

            float jacobian = 1;

            // Note: needed for sample 0 too due to temporal jitter.
            {
                // Distance falloff. Needed to avoid leaks.
                jacobian *= clamp(prev_dist / dist_to_sample_hit, 1e-4, 1e4);
                jacobian *= jacobian;

                // N of hit dot -L. Needed to avoid leaks. Without it, light "hugs" corners.
                //
                jacobian *= clamp(center_to_hit_vis / sample_hit_normal_ws_dot.w, 0, 1e4);
            }

            // Fixes boiling artifacts near edges. Unstable jacobians,
            // but also effectively reduces reliance on reservoir exchange
            // in tight corners, which is desirable since the well-distributed
            // raw samples thrown at temporal filters will do better.
            if (RTDGI_RESTIR_USE_JACOBIAN_BASED_REJECTION) {
                // Clamp neighbors give us a hit point that's considerably easier to sample
                // from our own position than from the neighbor. This can cause some darkening,
                // but prevents fireflies.
                //
                // The darkening occurs in corners, where micro-bounce should be happening instead.

#if 1
                // Doesn't over-darken corners as much
                jacobian = min(jacobian, RTDGI_RESTIR_JACOBIAN_BASED_REJECTION_VALUE);
#else
                // Slightly less noise
                if (jacobian > RTDGI_RESTIR_JACOBIAN_BASED_REJECTION_VALUE) {
                    continue;
                }
#endif
            }

            r.M *= relevance;

            if (0 == sample_i) {
                center_M = r.M;
            }

            if (Reservoir1spp_update_with_stream(reservoir,
                                                 r, p_q, jacobian * visibility,
                                                 stream_state, reservoir_payload, rng)) {
                outgoing_dir = dir_to_sample_hit;
                src_px_sel = rpx;
                radiance_sel = prev_rad.rgb;
                ray_orig_sel_ws = prev_ray_orig;
                ray_hit_sel_ws = sample_hit_ws;
                hit_normal_sel = sample_hit_normal_ws_dot.xyz;
            }
        }

        Reservoir1spp_finish_stream(reservoir, stream_state);
        reservoir.W = min(reservoir.W, RESTIR_RESERVOIR_W_CLAMP);
    }

    // TODO: this results in M being accumulated at a slower rate, although finally reaching
    // the limit we're after. What it does is practice is slow down the kernel tightening
    // in the subsequent spatial reservoir resampling.
    reservoir.M = center_M + 0.5;
    // reservoir.M = center_M + 1;

    RayDesc outgoing_ray;
    outgoing_ray.Direction = outgoing_dir;
    outgoing_ray.Origin = refl_ray_origin_ws;
    outgoing_ray.TMin = 0;

    const f32vec4 hit_normal_ws_dot = f32vec4(hit_normal_sel, -dot(hit_normal_sel, outgoing_ray.Direction));

    imageStore(daxa_image2D(radiance_out_tex), i32vec2(px), f32vec4(radiance_sel, dot(normal_ws, outgoing_ray.Direction)));
    imageStore(daxa_image2D(ray_orig_output_tex), i32vec2(px), f32vec4(ray_orig_sel_ws, 0.0));
    imageStore(daxa_image2D(hit_normal_output_tex), i32vec2(px), encode_hit_normal_and_dot(hit_normal_ws_dot));
    imageStore(daxa_image2D(ray_output_tex), i32vec2(px), f32vec4(ray_hit_sel_ws - ray_orig_sel_ws, length(ray_hit_sel_ws - refl_ray_origin_ws)));
    imageStore(daxa_uimage2D(reservoir_out_tex), i32vec2(px), u32vec4(Reservoir1spp_as_raw(reservoir), 0, 0));

    TemporalReservoirOutput res_packed;
    res_packed.depth = depth;
    res_packed.ray_hit_offset_ws = ray_hit_sel_ws - ray_hit_ws(view_ray_context);
    res_packed.luminance = max(0.0, sRGB_to_luminance(radiance_sel));
    res_packed.hit_normal_ws = hit_normal_ws_dot.xyz;
    imageStore(daxa_uimage2D(temporal_reservoir_packed_tex), i32vec2(px), TemporalReservoirOutput_as_raw(res_packed));
}
#endif

#if RTDGI_RESTIR_SPATIAL_COMPUTE
DAXA_DECL_PUSH_CONSTANT(RtdgiRestirSpatialPush, push)

#define USE_SSAO_WEIGHING 1
#define ALLOW_REUSE_OF_BACKFACING 1

u32vec2 reservoir_payload_to_px(uint payload) {
    return u32vec2(payload & 0xffff, payload >> 16);
}

// Two-thirds of SmeLU
float normal_inluence_nonlinearity(float x, float b) {
    return select(x < -b, 0, (x + b) * (x + b) / (4 * b));
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 px = gl_GlobalInvocationID.xy;
    const u32vec2 HALFRES_SUBSAMPLE_OFFSET = get_downscale_offset(gpu_input);
    const u32vec2 hi_px = px * 2 + HALFRES_SUBSAMPLE_OFFSET;
    float depth = texelFetch(daxa_texture2D(half_depth_tex), i32vec2(px), 0).r;

    const uint seed = deref(gpu_input).frame_index + push.spatial_reuse_pass_idx * 123;
    uint rng = hash3(u32vec3(px, seed));

    const f32vec2 uv = get_uv(hi_px, push.gbuffer_tex_size);
    const ViewRayContext view_ray_context = vrc_from_uv_and_depth(globals, uv, depth);

    const f32vec3 center_normal_vs = texelFetch(daxa_texture2D(half_view_normal_tex), i32vec2(px), 0).rgb;
    const f32vec3 center_normal_ws = direction_view_to_world(globals, center_normal_vs);
    const float center_depth = texelFetch(daxa_texture2D(half_depth_tex), i32vec2(px), 0).r;
    const float center_ssao = texelFetch(daxa_texture2D(half_ssao_tex), i32vec2(px), 0).r;

    Reservoir1sppStreamState stream_state = Reservoir1sppStreamState_create();
    Reservoir1spp reservoir = Reservoir1spp_create();

    f32vec3 dir_sel = f32vec3(1);

    float sample_radius_offset = uint_to_u01_float(hash1_mut(rng));

    Reservoir1spp center_r = Reservoir1spp_from_raw(texelFetch(daxa_utexture2D(reservoir_input_tex), i32vec2(px), 0).xy);

    float kernel_tightness = 1.0 - center_ssao;

    const uint SAMPLE_COUNT_PASS0 = 8;
    const uint SAMPLE_COUNT_PASS1 = 5;

    const float MAX_INPUT_M_IN_PASS0 = RESTIR_TEMPORAL_M_CLAMP;
    const float MAX_INPUT_M_IN_PASS1 = MAX_INPUT_M_IN_PASS0 * SAMPLE_COUNT_PASS0;
    const float MAX_INPUT_M_IN_PASS = select(push.spatial_reuse_pass_idx == 0, MAX_INPUT_M_IN_PASS0, MAX_INPUT_M_IN_PASS1);

    // TODO: consider keeping high in areas of high variance.
    if (RTDGI_RESTIR_SPATIAL_USE_KERNEL_NARROWING) {
        kernel_tightness = mix(
            kernel_tightness, 1.0,
            0.5 * smoothstep(MAX_INPUT_M_IN_PASS * 0.5, MAX_INPUT_M_IN_PASS, center_r.M));
    }

    float max_kernel_radius =
        select(push.spatial_reuse_pass_idx == 0, mix(32.0, 12.0, kernel_tightness), mix(16.0, 6.0, kernel_tightness));

    // TODO: only run more passes where absolutely necessary (dispatch in tiles)
    if (push.spatial_reuse_pass_idx >= 2) {
        max_kernel_radius = 8;
    }

    const f32vec2 dist_to_edge_xy = min(f32vec2(px), push.output_tex_size.xy - px);
    const float allow_edge_overstep = select(center_r.M < 10, 100.0, 1.25);
    // const float allow_edge_overstep = 1.25;
    const f32vec2 kernel_radius = min(f32vec2(max_kernel_radius), dist_to_edge_xy * allow_edge_overstep);
    // const f32vec2 kernel_radius = max_kernel_radius;

    uint sample_count = select(DIFFUSE_GI_USE_RESTIR != 0, select(push.spatial_reuse_pass_idx == 0, SAMPLE_COUNT_PASS0, SAMPLE_COUNT_PASS1), 1);

#if 1
    // Scrambling angles here would be nice, but results in bad cache thrashing.
    // Quantizing the offsets results in mild cache abuse, and fixes most of the artifacts
    // (flickering near edges, e.g. under sofa in the UE5 archviz apartment scene).
    const u32vec2 ang_offset_seed = select(push.spatial_reuse_pass_idx == 0, (px >> 3), (px >> 2));
#else
    // Haha, cache go brrrrrrr.
    const u32vec2 ang_offset_seed = px;
#endif

    float ang_offset = uint_to_u01_float(hash3(
                           u32vec3(ang_offset_seed, deref(gpu_input).frame_index * 2 + push.spatial_reuse_pass_idx))) *
                       M_PI * 2;

    if (!RESTIR_USE_SPATIAL) {
        sample_count = 1;
    }

    f32vec3 radiance_output = f32vec3(0);

    for (uint sample_i = 0; sample_i < sample_count; ++sample_i) {
        // float ang = M_PI / 2;
        float ang = (sample_i + ang_offset) * GOLDEN_ANGLE;
        f32vec2 radius =
            select(0 == sample_i, f32vec2(0), (pow(float(sample_i + sample_radius_offset) / sample_count, 0.5) * kernel_radius));
        i32vec2 rpx_offset = i32vec2(f32vec2(cos(ang), sin(ang)) * radius);

        const bool is_center_sample = sample_i == 0;
        // const bool is_center_sample = all(rpx_offset == 0);

        const i32vec2 rpx = i32vec2(px) + rpx_offset;

        const u32vec2 reservoir_raw = texelFetch(daxa_utexture2D(reservoir_input_tex), i32vec2(rpx), 0).xy;
        if (0 == reservoir_raw.x) {
            // Invalid reprojectoin
            continue;
        }

        Reservoir1spp r = Reservoir1spp_from_raw(reservoir_raw);

        r.M = min(r.M, 500);

        const u32vec2 spx = reservoir_payload_to_px(r.payload);

        const TemporalReservoirOutput spx_packed = TemporalReservoirOutput_from_raw(texelFetch(daxa_utexture2D(temporal_reservoir_packed_tex), i32vec2(spx), 0));
        const float reused_luminance = spx_packed.luminance;

        float visibility = 1;
        float relevance = 1;

        // Note: we're using `rpx` (neighbor reservoir px) here instead of `spx` (original ray px),
        // since we're merging with the stream of the neighbor and not the original ray.
        //
        // The distinction is in jacobians -- during every exchange, they get adjusted so that the target
        // pixel has correctly distributed rays. If we were to merge with the original pixel's stream,
        // we'd be applying the reservoirs several times.
        //
        // Consider for example merging a pixel with itself (no offset) multiple times over; we want
        // the jacobian to be 1.0 in that case, and not to reflect wherever its ray originally came from.

        const i32vec2 sample_offset = i32vec2(px) - i32vec2(rpx);
        const float sample_dist2 = dot(f32vec2(sample_offset), f32vec2(sample_offset));
        const f32vec3 sample_normal_vs = texelFetch(daxa_texture2D(half_view_normal_tex), i32vec2(rpx), 0).rgb;

        f32vec3 sample_radiance;
        if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
            sample_radiance = texelFetch(daxa_texture2D(bounced_radiance_input_tex), i32vec2(rpx), 0).rgb;
        }

        const float normal_similarity_dot = dot(sample_normal_vs, center_normal_vs);
#if ALLOW_REUSE_OF_BACKFACING
        // Allow reuse even with surfaces that face away, but weigh them down.
        relevance *= normal_inluence_nonlinearity(normal_similarity_dot, 0.5) / normal_inluence_nonlinearity(1.0, 0.5);
#else
        relevance *= max(0, normal_similarity_dot);
#endif

        const float sample_ssao = texelFetch(daxa_texture2D(half_ssao_tex), i32vec2(rpx), 0).r;

#if USE_SSAO_WEIGHING
        relevance *= 1 - abs(sample_ssao - center_ssao);
#endif

        const f32vec2 rpx_uv = get_uv(
            rpx * 2 + HALFRES_SUBSAMPLE_OFFSET,
            push.gbuffer_tex_size);
        const float rpx_depth = texelFetch(daxa_texture2D(half_depth_tex), i32vec2(rpx), 0).r;

        if (rpx_depth == 0.0) {
            continue;
        }

        const ViewRayContext rpx_ray_ctx = vrc_from_uv_and_depth(globals, rpx_uv, rpx_depth);

        const f32vec2 spx_uv = get_uv(
            spx * 2 + HALFRES_SUBSAMPLE_OFFSET,
            push.gbuffer_tex_size);
        const ViewRayContext spx_ray_ctx = vrc_from_uv_and_depth(globals, spx_uv, spx_packed.depth);
        const f32vec3 sample_hit_ws = spx_packed.ray_hit_offset_ws + ray_hit_ws(spx_ray_ctx);

        const f32vec3 reused_dir_to_sample_hit_unnorm_ws = sample_hit_ws - ray_hit_ws(rpx_ray_ctx);

        // const float reused_luminance = sample_hit_ws_and_luminance.a;

        // Note: we want the neighbor's sample, which might have been resampled already.
        const float reused_dist = length(reused_dir_to_sample_hit_unnorm_ws);
        const f32vec3 reused_dir_to_sample_hit_ws = reused_dir_to_sample_hit_unnorm_ws / reused_dist;

        const f32vec3 dir_to_sample_hit_unnorm = sample_hit_ws - ray_hit_ws(view_ray_context);
        const float dist_to_sample_hit = length(dir_to_sample_hit_unnorm);
        const f32vec3 dir_to_sample_hit = normalize(dir_to_sample_hit_unnorm);

        // Reject neighbors with vastly different depths
        if (!is_center_sample) {
            // Clamp the normal_vs.z so that we don't get arbitrarily loose depth comparison at grazing angles.
            const float depth_diff = abs(max(0.3, center_normal_vs.z) * (center_depth / rpx_depth - 1.0));

            const float depth_threshold =
                select(push.spatial_reuse_pass_idx == 0, 0.15, 0.1);

            relevance *= 1 - smoothstep(0.0, depth_threshold, depth_diff);
        }

        // Raymarch to check occlusion
        if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH && (push.perform_occlusion_raymarch != 0)) {
            const f32vec2 ray_orig_uv = spx_uv;

            // const float surface_offset_len = length(spx_ray_ctx.ray_hit_vs() - view_ray_context.ray_hit_vs());
            const float surface_offset_len = length(
                // Use the center depth for simplicity; this doesn't need to be exact.
                // Faster, looks about the same.
                ray_hit_vs(vrc_from_uv_and_depth(globals, ray_orig_uv, depth)) - ray_hit_vs(view_ray_context));

            // Multiplier over the surface offset from the center to the neighbor
            const float MAX_RAYMARCH_DIST_MULT = 3.0;

            // Trace towards the hit point.

            const f32vec3 raymarch_dir_unnorm_ws = sample_hit_ws - ray_hit_ws(view_ray_context);
            const f32vec3 raymarch_end_ws =
                ray_hit_ws(view_ray_context)
                // TODO: what's a good max distance to raymarch?
                + raymarch_dir_unnorm_ws * min(1.0, MAX_RAYMARCH_DIST_MULT * surface_offset_len / length(raymarch_dir_unnorm_ws));

            OcclusionScreenRayMarch raymarch = OcclusionScreenRayMarch_create(
                uv, view_ray_context.ray_hit_cs.xyz, ray_hit_ws(view_ray_context),
                raymarch_end_ws,
                push.gbuffer_tex_size.xy);

            OcclusionScreenRayMarch_with_max_sample_count(raymarch, 6);
            OcclusionScreenRayMarch_with_halfres_depth(raymarch, push.output_tex_size.xy, half_depth_tex);
            //.with_fullres_depth(depth_tex);

            if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
                OcclusionScreenRayMarch_with_color_bounce(raymarch, reprojected_gi_tex);
            }

            OcclusionScreenRayMarch_march(gpu_input, globals, raymarch, visibility, sample_radiance);
        }

        const f32vec3 sample_hit_normal_ws = spx_packed.hit_normal_ws;

        // phi_2^r in the ReSTIR GI paper
        const float center_to_hit_vis = -dot(sample_hit_normal_ws, dir_to_sample_hit);

        // phi_2^q
        const float reused_to_hit_vis = -dot(sample_hit_normal_ws, reused_dir_to_sample_hit_ws);

        float p_q = 1;
        if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
            p_q *= sRGB_to_luminance(sample_radiance);
        } else {
            p_q *= reused_luminance;
        }

        // Unlike in temporal reuse, here we can (and should) be running this.
        p_q *= max(0, dot(dir_to_sample_hit, center_normal_ws));

        float jacobian = 1;

        // Distance falloff. Needed to avoid leaks.
        jacobian *= reused_dist / dist_to_sample_hit;
        jacobian *= jacobian;

        // N of hit dot -L. Needed to avoid leaks. Without it, light "hugs" corners.
        //
        // Note: importantly, using the neighbor's data, not the original ray.
        jacobian *= clamp(center_to_hit_vis / reused_to_hit_vis, 0, 1e4);

        // Clearly wrong, but!:
        // The Jacobian introduces additional noise in corners, which is difficult to filter.
        // We still need something _resembling_ the jacobian in order to get directional cutoff,
        // and avoid leaks behind surfaces, but we don't actually need the precise Jacobian.
        // This causes us to lose some energy very close to corners, but with the near field split,
        // we don't need it anyway -- and it's better not to have the larger dark halos near corners,
        // which fhe full jacobian can cause due to imperfect integration (color bbox filters, etc).
        jacobian = sqrt(jacobian);

        if (is_center_sample) {
            jacobian = 1;
        }

        // Clamp neighbors give us a hit point that's considerably easier to sample
        // from our own position than from the neighbor. This can cause some darkening,
        // but prevents fireflies.
        //
        // The darkening occurs in corners, where micro-bounce should be happening instead.

        if (RTDGI_RESTIR_USE_JACOBIAN_BASED_REJECTION) {
#if 1
            // Doesn't over-darken corners as much
            jacobian = min(jacobian, RTDGI_RESTIR_JACOBIAN_BASED_REJECTION_VALUE);
#else
            // Slightly less noise
            if (jacobian > RTDGI_RESTIR_JACOBIAN_BASED_REJECTION_VALUE) {
                continue;
            }
#endif
        }

        if (!(p_q >= 0)) {
            continue;
        }

        r.M *= relevance;

        if (push.occlusion_raymarch_importance_only != 0) {
            // This is used with ray-traced reservoir visibility which happens after
            // the last spatial resampling. We don't _need_ to perform the raymarch
            // for it, but importance sampling based on unshadowed contribution
            // could end up choosing occluded areas, which then get turned black
            // by the ray-traced check. This then creates extra variance.
            //
            // We can instead try to use the ray-marched visibility as an estimator
            // of real visibility.

            p_q *= mix(0.25, 1.0, visibility);
            visibility = 1;
        }

        if (Reservoir1spp_update_with_stream(reservoir,
                                             r, p_q, visibility * jacobian,
                                             stream_state, r.payload, rng)) {
            dir_sel = dir_to_sample_hit;
            radiance_output = sample_radiance;
        }
    }

    Reservoir1spp_finish_stream(reservoir, stream_state);
    reservoir.W = min(reservoir.W, RESTIR_RESERVOIR_W_CLAMP);

    imageStore(daxa_uimage2D(reservoir_output_tex), i32vec2(px), u32vec4(Reservoir1spp_as_raw(reservoir), 0, 0));

    if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
        imageStore(daxa_image2D(bounced_radiance_output_tex), i32vec2(px), f32vec4(radiance_output, 0.0));
    }
}
#endif

#if RTDGI_RESTIR_RESOLVE_COMPUTE
DAXA_DECL_PUSH_CONSTANT(RtdgiRestirResolvePush, push)

float ggx_ndf_unnorm(float a2, float cos_theta) {
    float denom_sqrt = cos_theta * cos_theta * (a2 - 1.0) + 1.0;
    return a2 / (denom_sqrt * denom_sqrt);
}

u32vec2 reservoir_payload_to_px(uint payload) {
    return u32vec2(payload & 0xffff, payload >> 16);
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec2 px = gl_GlobalInvocationID.xy;
    const u32vec2 HALFRES_SUBSAMPLE_OFFSET = get_downscale_offset(gpu_input);
    const i32vec2 hi_px_offset = i32vec2(HALFRES_SUBSAMPLE_OFFSET);

    float depth = texelFetch(daxa_texture2D(depth_tex), i32vec2(px), 0).r;
    if (0 == depth) {
        imageStore(daxa_image2D(irradiance_output_tex), i32vec2(px), f32vec4(0));
        return;
    }

    const uint seed = deref(gpu_input).frame_index;
    uint rng = hash3(u32vec3(px, seed));

    const f32vec2 uv = get_uv(px, push.gbuffer_tex_size);
    const ViewRayContext view_ray_context = vrc_from_uv_and_depth(globals, uv, depth);

    u32vec4 g_buffer_value = texelFetch(daxa_utexture2D(gbuffer_tex), i32vec2(px), 0);
    f32vec3 gbuffer_nrm = u16_to_nrm(g_buffer_value.y);

    const f32vec3 center_normal_ws = gbuffer_nrm;
    const f32vec3 center_normal_vs = direction_world_to_view(globals, center_normal_ws);
    const float center_depth = depth;
    const float center_ssao = texelFetch(daxa_texture2D(ssao_tex), i32vec2(px), 0).r;

    // const f32vec3 center_bent_normal_ws = normalize(direction_view_to_world(globals, texelFetch(daxa_texture2D(ssao_tex), i32vec2(px * 2), 0).gba));

    const uint frame_hash = hash1(deref(gpu_input).frame_index);
    const uint px_idx_in_quad = (((px.x & 1) | (px.y & 1) * 2) + frame_hash) & 3;
    const f32vec4 blue = blue_noise_for_pixel(blue_noise_vec2, px, deref(gpu_input).frame_index) * M_TAU;

    const float NEAR_FIELD_FADE_OUT_END = -ray_hit_vs(view_ray_context).z * (SSGI_NEAR_FIELD_RADIUS * push.output_tex_size.w * 0.5);
    const float NEAR_FIELD_FADE_OUT_START = NEAR_FIELD_FADE_OUT_END * 0.5;

#if RTDGI_INTERLEAVED_VALIDATION_ALWAYS_TRACE_NEAR_FIELD
    // The near field cannot be fully trusted in tight corners because our irradiance cache
    // has limited resolution, and is likely to create artifacts. Opt on the side of shadowing.
    const float near_field_influence = center_ssao;
#else
    const float near_field_influence = select(is_rtdgi_tracing_frame(), center_ssao, 0);
#endif

    f32vec3 total_irradiance = f32vec3(0.0);
    bool sharpen_gi_kernel = false;

    {
        float w_sum = 0;
        f32vec3 weighted_irradiance = f32vec3(0.0);

        for (uint sample_i = 0; sample_i < select(RTDGI_RESTIR_USE_RESOLVE_SPATIAL_FILTER != 0, 4, 1); ++sample_i) {
            const float ang = (sample_i + blue.x) * GOLDEN_ANGLE + (px_idx_in_quad / 4.0) * M_TAU;
            const float radius =
                select(RTDGI_RESTIR_USE_RESOLVE_SPATIAL_FILTER != 0, (pow(float(sample_i), 0.666) * 1.0 + 0.4), 0.0);
            const f32vec2 reservoir_px_offset = f32vec2(cos(ang), sin(ang)) * radius;
            const i32vec2 rpx = i32vec2(floor(f32vec2(px) * 0.5 + reservoir_px_offset));

            const f32vec2 rpx_uv = get_uv(
                rpx * 2 + HALFRES_SUBSAMPLE_OFFSET,
                push.gbuffer_tex_size);
            const float rpx_depth = texelFetch(daxa_texture2D(half_depth_tex), i32vec2(rpx), 0).r;
            const ViewRayContext rpx_ray_ctx = vrc_from_uv_and_depth(globals, rpx_uv, rpx_depth);

            if (USE_SPLIT_RT_NEAR_FIELD != 0) {
                const f32vec3 hit_ws = texelFetch(daxa_texture2D(candidate_hit_tex), i32vec2(rpx), 0).xyz + ray_hit_ws(rpx_ray_ctx);
                const f32vec3 sample_offset = hit_ws - ray_hit_ws(view_ray_context);
                const float sample_dist = length(sample_offset);
                const f32vec3 sample_dir = sample_offset / sample_dist;

                const float geometric_term =
                    // TODO: fold the 2 into the PDF
                    2 * max(0.0, dot(center_normal_ws, sample_dir));

                const float atten = smoothstep(NEAR_FIELD_FADE_OUT_END, NEAR_FIELD_FADE_OUT_START, sample_dist);
                sharpen_gi_kernel = sharpen_gi_kernel || (atten > 0.9);

                f32vec3 contribution = texelFetch(daxa_texture2D(candidate_radiance_tex), i32vec2(rpx), 0).rgb * geometric_term;
                contribution *= mix(0.0, atten, near_field_influence);

                f32vec3 sample_normal_vs = texelFetch(daxa_texture2D(half_view_normal_tex), i32vec2(rpx), 0).rgb;
                const float sample_ssao = texelFetch(daxa_texture2D(ssao_tex), i32vec2(rpx * 2 + HALFRES_SUBSAMPLE_OFFSET), 0).r;

                float w = 1;
                w *= ggx_ndf_unnorm(0.01, saturate(dot(center_normal_vs, sample_normal_vs)));
                w *= exp2(-200.0 * abs(center_normal_vs.z * (center_depth / rpx_depth - 1.0)));

                weighted_irradiance += contribution * w;
                w_sum += w;
            }
        }

        total_irradiance += weighted_irradiance / max(1e-20, w_sum);
    }

    {
        float w_sum = 0;
        f32vec3 weighted_irradiance = f32vec3(0);

        const float kernel_scale = select(sharpen_gi_kernel, 0.5, 1.0);

        for (uint sample_i = 0; sample_i < select(RTDGI_RESTIR_USE_RESOLVE_SPATIAL_FILTER != 0, 4, 1); ++sample_i) {
            const float ang = (sample_i + blue.x) * GOLDEN_ANGLE + (px_idx_in_quad / 4.0) * M_TAU;
            const float radius =
                select(RTDGI_RESTIR_USE_RESOLVE_SPATIAL_FILTER != 0, (pow(float(sample_i), 0.666) * 1.0 * kernel_scale + 0.4 * kernel_scale), 0.0);

            const f32vec2 reservoir_px_offset = f32vec2(cos(ang), sin(ang)) * radius;
            const i32vec2 rpx = i32vec2(floor(f32vec2(px) * 0.5 + reservoir_px_offset));

            Reservoir1spp r = Reservoir1spp_from_raw(texelFetch(daxa_utexture2D(reservoir_input_tex), i32vec2(rpx), 0).xy);
            const u32vec2 spx = reservoir_payload_to_px(r.payload);

            const TemporalReservoirOutput spx_packed = TemporalReservoirOutput_from_raw(texelFetch(daxa_utexture2D(temporal_reservoir_packed_tex), i32vec2(spx), 0));

            const f32vec2 spx_uv = get_uv(
                spx * 2 + HALFRES_SUBSAMPLE_OFFSET,
                push.gbuffer_tex_size);
            const ViewRayContext spx_ray_ctx = vrc_from_uv_and_depth(globals, spx_uv, spx_packed.depth);

            {
                const float spx_depth = spx_packed.depth;
                const float rpx_depth = texelFetch(daxa_texture2D(half_depth_tex), i32vec2(rpx), 0).r;

                const f32vec3 hit_ws = spx_packed.ray_hit_offset_ws + ray_hit_ws(spx_ray_ctx);
                const f32vec3 sample_offset = hit_ws - ray_hit_ws(view_ray_context);
                const float sample_dist = length(sample_offset);
                const f32vec3 sample_dir = sample_offset / sample_dist;

                const float geometric_term =
                    // TODO: fold the 2 into the PDF
                    2 * max(0.0, dot(center_normal_ws, sample_dir));

                f32vec3 radiance;
                if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
                    radiance = texelFetch(daxa_texture2D(bounced_radiance_input_tex), i32vec2(rpx), 0).rgb;
                } else {
                    radiance = texelFetch(daxa_texture2D(radiance_tex), i32vec2(spx), 0).rgb;
                }

                if (USE_SPLIT_RT_NEAR_FIELD != 0) {
                    const float atten = smoothstep(NEAR_FIELD_FADE_OUT_START, NEAR_FIELD_FADE_OUT_END, sample_dist);
                    radiance *= mix(1.0, atten, near_field_influence);
                }

                const f32vec3 contribution = radiance * geometric_term * r.W;

                f32vec3 sample_normal_vs = texelFetch(daxa_texture2D(half_view_normal_tex), i32vec2(spx), 0).rgb;
                const float sample_ssao = texelFetch(daxa_texture2D(ssao_tex), i32vec2(rpx * 2 + HALFRES_SUBSAMPLE_OFFSET), 0).r;

                float w = 1;
                w *= ggx_ndf_unnorm(0.01, saturate(dot(center_normal_vs, sample_normal_vs)));
                w *= exp2(-200.0 * abs(center_normal_vs.z * (center_depth / rpx_depth - 1.0)));
                w *= exp2(-20.0 * abs(center_ssao - sample_ssao));

                weighted_irradiance += contribution * w;
                w_sum += w;
            }
        }

        total_irradiance += weighted_irradiance / max(1e-20, w_sum);
    }

    imageStore(daxa_image2D(irradiance_output_tex), i32vec2(px), f32vec4(total_irradiance, 1));
}
#endif
