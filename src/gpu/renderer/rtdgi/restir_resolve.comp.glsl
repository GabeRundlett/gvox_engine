#include <shared/app.inl>

#include <utils/math.glsl>
#include <utils/color.glsl>
// #include <utils/samplers.glsl>
// #include <utils/frame_constants.glsl>
// #include <utils/pack_unpack.glsl>
#include <utils/brdf.glsl>
#include <utils/brdf_lut.glsl>
#include <utils/layered_brdf.glsl>
// #include <utils/uv.glsl>
// #include <utils/hash.glsl>
#include <utils/reservoir.glsl>
// #include <utils/blue_noise.glsl>
#include "near_field_settings.glsl"
#include "rtdgi_restir_settings.glsl"
#include "rtdgi_common.glsl"
#include "candidate_ray_dir.glsl"

#include <utils/safety.glsl>
#include <utils/downscale.glsl>

float ggx_ndf_unnorm(float a2, float cos_theta) {
    float denom_sqrt = cos_theta * cos_theta * (a2 - 1.0) + 1.0;
    return a2 / (denom_sqrt * denom_sqrt);
}

uvec2 reservoir_payload_to_px(uint payload) {
    return uvec2(payload & 0xffff, payload >> 16);
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    uvec2 px = gl_GlobalInvocationID.xy;
    const ivec2 hi_px_offset = ivec2(HALFRES_SUBSAMPLE_OFFSET);

    float depth = safeTexelFetch(depth_tex, ivec2(px), 0).r;
    if (0 == depth) {
        safeImageStore(irradiance_output_tex, ivec2(px), vec4(0));
        return;
    }

    const uint seed = deref(gpu_input).frame_index;
    uint rng = hash3(uvec3(px, seed));

    const vec2 uv = get_uv(px, push.gbuffer_tex_size);
    const ViewRayContext view_ray_context = vrc_from_uv_and_depth(globals, uv, depth);

    GbufferData gbuffer = unpack(GbufferDataPacked(safeTexelFetchU(gbuffer_tex, ivec2(px), 0)));

    const vec3 center_normal_ws = gbuffer.normal;
    const vec3 center_normal_vs = direction_world_to_view(globals, center_normal_ws);
    const float center_depth = depth;
    const float center_ssao = safeTexelFetch(ssao_tex, ivec2(px), 0).r;

    // const vec3 center_bent_normal_ws = normalize(direction_view_to_world(ssao_tex[px * 2].gba));

    const uint frame_hash = hash1(deref(gpu_input).frame_index);
    const uint px_idx_in_quad = (((px.x & 1) | (px.y & 1) * 2) + frame_hash) & 3;
    const vec4 blue = blue_noise_for_pixel(px, deref(gpu_input).frame_index) * M_TAU;

    const float NEAR_FIELD_FADE_OUT_END = -ray_hit_vs(view_ray_context).z * (SSGI_NEAR_FIELD_RADIUS * push.output_tex_size.w * 0.5);
    const float NEAR_FIELD_FADE_OUT_START = NEAR_FIELD_FADE_OUT_END * 0.5;

#if RTDGI_INTERLEAVED_VALIDATION_ALWAYS_TRACE_NEAR_FIELD
    // The near field cannot be fully trusted in tight corners because our irradiance cache
    // has limited resolution, and is likely to create artifacts. Opt on the side of shadowing.
    const float near_field_influence = center_ssao;
#else
    const float near_field_influence = select(is_rtdgi_tracing_frame(), center_ssao, 0);
#endif

    vec3 total_irradiance = vec3(0.0);
    bool sharpen_gi_kernel = false;

    {
        float w_sum = 0;
        vec3 weighted_irradiance = vec3(0.0);

        for (uint sample_i = 0; sample_i < select(RTDGI_RESTIR_USE_RESOLVE_SPATIAL_FILTER != 0, 4, 1); ++sample_i) {
            const float ang = (sample_i + blue.x) * GOLDEN_ANGLE + (px_idx_in_quad / 4.0) * M_TAU;
            const float radius =
                select(RTDGI_RESTIR_USE_RESOLVE_SPATIAL_FILTER != 0, (pow(float(sample_i), 0.666) * 1.0 + 0.4), 0.0);
            const vec2 reservoir_px_offset = vec2(cos(ang), sin(ang)) * radius;
            const ivec2 rpx = ivec2(floor(vec2(px) * 0.5 + reservoir_px_offset));

            const vec2 rpx_uv = get_uv(
                rpx * 2 + HALFRES_SUBSAMPLE_OFFSET,
                push.gbuffer_tex_size);
            const float rpx_depth = safeTexelFetch(half_depth_tex, ivec2(rpx), 0).r;
            const ViewRayContext rpx_ray_ctx = vrc_from_uv_and_depth(globals, rpx_uv, rpx_depth);

            if (USE_SPLIT_RT_NEAR_FIELD != 0) {
                const vec3 hit_ws = safeTexelFetch(candidate_hit_tex, ivec2(rpx), 0).xyz + ray_hit_ws(rpx_ray_ctx);
                const vec3 sample_offset = hit_ws - ray_hit_ws(view_ray_context);
                const float sample_dist = length(sample_offset);
                const vec3 sample_dir = sample_offset / sample_dist;

                const float geometric_term =
                    // TODO: fold the 2 into the PDF
                    2 * max(0.0, dot(center_normal_ws, sample_dir));

                const float atten = smoothstep(NEAR_FIELD_FADE_OUT_END, NEAR_FIELD_FADE_OUT_START, sample_dist);
                sharpen_gi_kernel = sharpen_gi_kernel || (atten > 0.9);

                vec3 contribution = safeTexelFetch(candidate_radiance_tex, ivec2(rpx), 0).rgb * geometric_term;
                contribution *= mix(0.0, atten, near_field_influence);

                vec3 sample_normal_vs = safeTexelFetch(half_view_normal_tex, ivec2(rpx), 0).rgb;
                const float sample_ssao = safeTexelFetch(ssao_tex, ivec2(rpx * 2 + HALFRES_SUBSAMPLE_OFFSET), 0).r;

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
        vec3 weighted_irradiance = vec3(0);

        const float kernel_scale = select(sharpen_gi_kernel, 0.5, 1.0);

        for (uint sample_i = 0; sample_i < select(RTDGI_RESTIR_USE_RESOLVE_SPATIAL_FILTER != 0, 4, 1); ++sample_i) {
            const float ang = (sample_i + blue.x) * GOLDEN_ANGLE + (px_idx_in_quad / 4.0) * M_TAU;
            const float radius =
                select(RTDGI_RESTIR_USE_RESOLVE_SPATIAL_FILTER != 0, (pow(float(sample_i), 0.666) * 1.0 * kernel_scale + 0.4 * kernel_scale), 0.0);

            const vec2 reservoir_px_offset = vec2(cos(ang), sin(ang)) * radius;
            const ivec2 rpx = ivec2(floor(vec2(px) * 0.5 + reservoir_px_offset));

            Reservoir1spp r = Reservoir1spp_from_raw(safeTexelFetchU(reservoir_input_tex, ivec2(rpx), 0).xy);
            const uvec2 spx = reservoir_payload_to_px(r.payload);

            const TemporalReservoirOutput spx_packed = TemporalReservoirOutput_from_raw(safeTexelFetchU(temporal_reservoir_packed_tex, ivec2(spx), 0));

            const vec2 spx_uv = get_uv(
                spx * 2 + HALFRES_SUBSAMPLE_OFFSET,
                push.gbuffer_tex_size);
            const ViewRayContext spx_ray_ctx = vrc_from_uv_and_depth(globals, spx_uv, spx_packed.depth);

            {
                const float spx_depth = spx_packed.depth;
                const float rpx_depth = safeTexelFetch(half_depth_tex, ivec2(rpx), 0).r;

                const vec3 hit_ws = spx_packed.ray_hit_offset_ws + ray_hit_ws(spx_ray_ctx);
                const vec3 sample_offset = hit_ws - ray_hit_ws(view_ray_context);
                const float sample_dist = length(sample_offset);
                const vec3 sample_dir = sample_offset / sample_dist;

                const float geometric_term =
                    // TODO: fold the 2 into the PDF
                    2 * max(0.0, dot(center_normal_ws, sample_dir));

                vec3 radiance;
                if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
                    // radiance = safeTexelFetch(bounced_radiance_input_tex, ivec2(rpx), 0).rgb;
                } else {
                    radiance = safeTexelFetch(radiance_tex, ivec2(spx), 0).rgb;
                }

                if (USE_SPLIT_RT_NEAR_FIELD != 0) {
                    const float atten = smoothstep(NEAR_FIELD_FADE_OUT_START, NEAR_FIELD_FADE_OUT_END, sample_dist);
                    radiance *= mix(1.0, atten, near_field_influence);
                }

                const vec3 contribution = radiance * geometric_term * r.W;

                vec3 sample_normal_vs = safeTexelFetch(half_view_normal_tex, ivec2(spx), 0).rgb;
                const float sample_ssao = safeTexelFetch(ssao_tex, ivec2(rpx * 2 + HALFRES_SUBSAMPLE_OFFSET), 0).r;

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

    safeImageStore(irradiance_output_tex, ivec2(px), vec4(total_irradiance, 1));
}
