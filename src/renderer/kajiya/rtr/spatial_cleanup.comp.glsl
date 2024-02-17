#include <renderer/kajiya/rtr.inl>

#include <renderer/kajiya/inc/camera.glsl>
// #include <utilities/gpu/uv.glsl>
// #include "../inc/frame_constants.glsl"
#include "../inc/working_color_space.glsl"
#define linear_to_working linear_rgb_to_crunched_rgb
#define working_to_linear crunched_rgb_to_linear_rgb
// #define linear_to_working linear_rgb_to_linear_rgb
// #define working_to_linear linear_rgb_to_linear_rgb
#include "../inc/safety.glsl"

DAXA_DECL_PUSH_CONSTANT(RtrSpatialFilterComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_BufferPtr(ivec2) spatial_resolve_offsets = push.uses.spatial_resolve_offsets;
daxa_ImageViewIndex input_tex = push.uses.input_tex;
daxa_ImageViewIndex depth_tex = push.uses.depth_tex;
daxa_ImageViewIndex geometric_normal_tex = push.uses.geometric_normal_tex;
daxa_ImageViewIndex output_tex = push.uses.output_tex;

const bool SHUFFLE_SUBPIXELS = true;

float depth_to_view_z(float depth) {
    // NOTE(grundlett): In Kajiya, this is `clip_to_view._43`, which
    // I have no clue what corresponds to in GLSL.
    return rcp(depth * -deref(globals).player.cam.clip_to_view[2][3]);
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    uvec2 px = gl_GlobalInvocationID.xy;
    const vec4 center = safeTexelFetch(input_tex, ivec2(px), 0);
    const float center_depth = safeTexelFetch(depth_tex, ivec2(px), 0).r;
    const float center_sample_count = center.w;

    const float min_sample_count = 8;

    if (center_sample_count >= min_sample_count || center_depth == 0.0) {
        safeImageStore(output_tex, ivec2(px), center);
        return;
    }

    const vec3 center_normal_vs = safeTexelFetch(geometric_normal_tex, ivec2(px), 0).xyz * 2.0 - 1.0;

    // TODO: project the BRDF lobe footprint; this only works for certain roughness ranges
    const float filter_radius_ss = 0.5 * deref(globals).player.cam.view_to_clip[1][1] / -depth_to_view_z(center_depth);
    const uint filter_idx = uint(clamp(filter_radius_ss * 7.0, 0.0, 7.0));

    vec3 vsum = 0.0.xxx;
    float wsum = 0.0;

    const uint sample_count = clamp(uint(8 - center_sample_count / 2), uint(min_sample_count / 4), uint(min_sample_count));
    // const uint sample_count = 8;
    const int kernel_scale = select(center_sample_count < 4, 2, 1);
    const uint px_idx_in_quad = (((px.x & 1) | (px.y & 1) * 2) + select(SHUFFLE_SUBPIXELS, 1, 0) * deref(gpu_input).frame_index) & 3;

    for (uint sample_i = 0; sample_i < sample_count; ++sample_i) {
        // TODO: precalculate temporal variants
        ivec2 sample_px = ivec2(px) + kernel_scale * deref(spatial_resolve_offsets[(px_idx_in_quad * 16 + sample_i) + 64 * filter_idx]).xy;

        const vec3 neigh = linear_to_working(safeTexelFetch(input_tex, sample_px, 0)).rgb;
        const float sample_depth = safeTexelFetch(depth_tex, sample_px, 0).r;
        const vec3 sample_normal_vs = safeTexelFetch(geometric_normal_tex, sample_px, 0).xyz * 2.0 - 1.0;

        float w = 1;
        // TODO: BRDF-based weights
        w *= exp2(-50.0 * abs(center_normal_vs.z * (center_depth / sample_depth - 1.0)));
        float dp = saturate(dot(center_normal_vs, sample_normal_vs));
        w *= dp * dp * dp;

        vsum += neigh * w;
        wsum += w;
    }

    safeImageStore(output_tex, ivec2(px), working_to_linear(vec4(vsum / wsum, 1.0)));
}
