#pragma once

#include <utils/camera.glsl>
#include <utils/downscale.glsl>

struct OcclusionScreenRayMarch {
    uint max_sample_count;
    daxa_f32vec2 raymarch_start_uv;
    daxa_f32vec3 raymarch_start_cs;
    daxa_f32vec3 raymarch_start_ws;
    daxa_f32vec3 raymarch_end_ws;
    daxa_f32vec2 fullres_depth_tex_size;

    bool use_halfres_depth;
    daxa_f32vec2 halfres_depth_tex_size;
    daxa_ImageViewIndex halfres_depth_tex;

    daxa_ImageViewIndex fullres_depth_tex;

    bool use_color_bounce;
    daxa_ImageViewIndex fullres_color_bounce_tex;
};

void with_color_bounce(inout OcclusionScreenRayMarch self, daxa_ImageViewIndex _fullres_color_bounce_tex) {
    self.use_color_bounce = true;
    self.fullres_color_bounce_tex = _fullres_color_bounce_tex;
}

void with_max_sample_count(OcclusionScreenRayMarch self, uint _max_sample_count) {
    self.max_sample_count = _max_sample_count;
}

void with_halfres_depth(
    inout OcclusionScreenRayMarch self,
    daxa_f32vec2 _halfres_depth_tex_size,
    daxa_ImageViewIndex _halfres_depth_tex) {
    self.use_halfres_depth = true;
    self.halfres_depth_tex_size = _halfres_depth_tex_size;
    self.halfres_depth_tex = _halfres_depth_tex;
}

void with_fullres_depth(
    inout OcclusionScreenRayMarch self,
    daxa_ImageViewIndex _fullres_depth_tex) {
    self.use_halfres_depth = false;
    self.fullres_depth_tex = _fullres_depth_tex;
}

OcclusionScreenRayMarch OcclusionScreenRayMarch_create(
    daxa_f32vec2 raymarch_start_uv, daxa_f32vec3 raymarch_start_cs, daxa_f32vec3 raymarch_start_ws,
    daxa_f32vec3 raymarch_end_ws,
    daxa_f32vec2 fullres_depth_tex_size) {
    OcclusionScreenRayMarch res;
    res.max_sample_count = 4;
    res.raymarch_start_uv = raymarch_start_uv;
    res.raymarch_start_cs = raymarch_start_cs;
    res.raymarch_start_ws = raymarch_start_ws;
    res.raymarch_end_ws = raymarch_end_ws;

    res.fullres_depth_tex_size = fullres_depth_tex_size;

    res.use_color_bounce = false;
    return res;
}

void march(
    daxa_BufferPtr(GpuInput) gpu_input,
    daxa_RWBufferPtr(GpuGlobals) globals,
    OcclusionScreenRayMarch self,
    inout float visibility,
    inout daxa_f32vec3 sample_radiance) {
    const vec2 raymarch_end_uv = cs_to_uv(position_world_to_clip(globals, self.raymarch_end_ws).xy);
    const vec2 raymarch_uv_delta = raymarch_end_uv - self.raymarch_start_uv;
    const vec2 raymarch_len_px = raymarch_uv_delta * select(bvec2(self.use_halfres_depth), self.halfres_depth_tex_size, self.fullres_depth_tex_size);

    const uint MIN_PX_PER_STEP = 2;

    const int k_count = min(int(self.max_sample_count), int(floor(length(raymarch_len_px) / MIN_PX_PER_STEP)));

    // Depth values only have the front; assume a certain thickness.
    const float Z_LAYER_THICKNESS = 0.05;

    // const vec3 raymarch_start_cs = view_ray_context.ray_hit_cs.xyz;
    const vec3 raymarch_end_cs = position_world_to_clip(globals, self.raymarch_end_ws).xyz;
    const float depth_step_per_px = (raymarch_end_cs.z - self.raymarch_start_cs.z) / length(raymarch_len_px);
    const float depth_step_per_z = (raymarch_end_cs.z - self.raymarch_start_cs.z) / length(raymarch_end_cs.xy - self.raymarch_start_cs.xy);

    float t_step = 1.0 / k_count;
    float t = 0.5 * t_step;
    for (int k = 0; k < k_count; ++k) {
        const vec3 interp_pos_cs = mix(self.raymarch_start_cs, raymarch_end_cs, t);

        // The point-sampled UV could end up with a quite different depth value
        // than the one interpolated along the ray (which is not quantized).
        // This finds a conservative bias for the comparison.
        const vec2 uv_at_interp = cs_to_uv(interp_pos_cs.xy);

        uvec2 px_at_interp;
        float depth_at_interp;

        if (self.use_halfres_depth) {
            px_at_interp = (uvec2(floor(uv_at_interp * self.fullres_depth_tex_size - HALFRES_SUBSAMPLE_OFFSET)) & ~1u) + HALFRES_SUBSAMPLE_OFFSET;
            depth_at_interp = texelFetch(daxa_texture2D(self.halfres_depth_tex), daxa_i32vec2(px_at_interp >> 1u), 0).r;
        } else {
            px_at_interp = uvec2(floor(uv_at_interp * self.fullres_depth_tex_size));
            depth_at_interp = texelFetch(daxa_texture2D(self.fullres_depth_tex), daxa_i32vec2(px_at_interp), 0).r;
        }

        const vec2 quantized_cs_at_interp = uv_to_cs((px_at_interp + 0.5) / self.fullres_depth_tex_size);

        const float biased_interp_z = self.raymarch_start_cs.z + depth_step_per_z * length(quantized_cs_at_interp - self.raymarch_start_cs.xy);

        if (depth_at_interp > biased_interp_z) {
            const float depth_diff = inverse_depth_relative_diff(interp_pos_cs.z, depth_at_interp);

            float hit = smoothstep(
                Z_LAYER_THICKNESS,
                Z_LAYER_THICKNESS * 0.5,
                depth_diff);

            // if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
            //     const vec3 hit_radiance = fullres_color_bounce_tex.SampleLevel(sampler_llc, cs_to_uv(interp_pos_cs.xy), 0).rgb;
            //     const vec3 prev_sample_radiance = sample_radiance;

            //     sample_radiance = max(sample_radiance, hit_radiance, hit);

            //     // Heuristic: don't allow getting _brighter_ from accidental
            //     // hits reused from neighbors. This can cause some darkening,
            //     // but also fixes reduces noise (expecting to hit dark, hitting bright),
            //     // and improves a few cases that otherwise look unshadowed.
            //     visibility *= min(1.0, sRGB_to_luminance(prev_sample_radiance) / sRGB_to_luminance(sample_radiance));
            // } else {
            visibility *= 1.0 - hit;
            // }

            if (depth_diff > Z_LAYER_THICKNESS) {
                // Going behind an object; could be sketchy.
                // Note: maybe nuke.. causes bias around foreground objects.
                // relevance *= 0.2;
            }
        }

        t += t_step;
    }
}
