#include <renderer/kajiya/taa.inl>

#include <g_samplers>
#include "../inc/camera.glsl"
#include "../inc/color.glsl"
#include "../inc/image.glsl"
#include "taa_common.glsl"

DAXA_DECL_PUSH_CONSTANT(TaaReprojectComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewIndex history_tex = push.uses.history_tex;
daxa_ImageViewIndex reprojection_map = push.uses.reprojection_map;
daxa_ImageViewIndex depth_image = push.uses.depth_image;
daxa_ImageViewIndex reprojected_history_img = push.uses.reprojected_history_img;
daxa_ImageViewIndex closest_velocity_img = push.uses.closest_velocity_img;

// Optimization: Try to skip velocity dilation if velocity diff is small
// around the pixel.
#define APPROX_SKIP_DILATION 1

vec4 fetch_history(vec2 uv) {
    vec4 h = textureLod(daxa_sampler2D(history_tex, g_sampler_lnc), uv, 0);
    return vec4(decode_rgb(h.xyz * deref(gpu_input).pre_exposure_delta), h.w);
}

vec4 HistoryRemap_remap(vec4 v) {
    return vec4(decode_rgb(v.rgb * deref(gpu_input).pre_exposure_delta), v.a);
}
image_sample_catmull_rom_TEMPLATE(HistoryRemap_remap);

float fetch_depth(ivec2 px) {
    return safeTexelFetch(depth_image, px, 0).r;
}
vec4 fetch_reproj(ivec2 px) {
    return safeTexelFetch(reprojection_map, px, 0);
}

layout(local_size_x = TAA_WG_SIZE_X, local_size_y = TAA_WG_SIZE_Y, local_size_z = 1) in;
void main() {
    vec2 px = gl_GlobalInvocationID.xy;

    vec4 input_tex_size = vec4(push.input_tex_size.xy, 0, 0);
    input_tex_size.zw = 1.0 / input_tex_size.xy;
    vec4 output_tex_size = vec4(push.output_tex_size.xy, 0, 0);
    output_tex_size.zw = 1.0 / output_tex_size.xy;

    const vec2 input_resolution_scale = input_tex_size.xy / output_tex_size.xy;
    const uvec2 reproj_px = uvec2((px + 0.5) * input_resolution_scale);

    vec2 uv = get_uv(px, output_tex_size);
    uvec2 closest_px = reproj_px;

#if APPROX_SKIP_DILATION
    // Find the bounding box of velocities around this 3x3 region
    vec2 vel_min;
    vec2 vel_max;
    {
        vec2 v = fetch_reproj(ivec2(reproj_px) + ivec2(-1, -1)).xy;
        vel_min = v;
        vel_max = v;
    }
    {
        vec2 v = fetch_reproj(ivec2(reproj_px) + ivec2(1, -1)).xy;
        vel_min = min(vel_min, v);
        vel_max = max(vel_max, v);
    }
    {
        vec2 v = fetch_reproj(ivec2(reproj_px) + ivec2(-1, 1)).xy;
        vel_min = min(vel_min, v);
        vel_max = max(vel_max, v);
    }
    {
        vec2 v = fetch_reproj(ivec2(reproj_px) + ivec2(1, 1)).xy;
        vel_min = min(vel_min, v);
        vel_max = max(vel_max, v);
    }

    uint should_dilate = uint(any(greaterThan(vel_max - vel_min, 0.1 * max(input_tex_size.zw, abs(vel_max + vel_min)))));

    // Since we're only checking a few pixels, there's a chance we'll miss something.
    // Dilate in the wave to reduce the chance of that happening.
    // should_dilate |= subgroupBroadcast(should_dilate, gl_SubgroupInvocationID ^ 1);
    should_dilate |= subgroupBroadcast(should_dilate, gl_SubgroupInvocationID ^ 2);
    // should_dilate |= subgroupBroadcast(should_dilate, gl_SubgroupInvocationID ^ 8);
    should_dilate |= subgroupBroadcast(should_dilate, gl_SubgroupInvocationID ^ 16);

    // We want to find the velocity of the pixel which is closest to the camera,
    // which is critical to anti-aliased moving edges.
    // At the same time, when everything moves with roughly the same velocity
    // in the neighborhood of the pixel, we'd be performing this depth-based kernel
    // only to return the same value.
    // Therefore, we predicate the search on there being any appreciable
    // velocity difference around the target pixel. This ends up being faster on average.
    if (should_dilate != 0)
#endif
    {
        float reproj_depth = fetch_depth(ivec2(reproj_px));
        int k = 1;
        for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                float d = fetch_depth(ivec2(reproj_px) + ivec2(x, y));
                if (d > reproj_depth) {
                    reproj_depth = d;
                    closest_px = reproj_px + ivec2(x, y);
                }
            }
        }
    }

    const vec2 reproj_xy = fetch_reproj(ivec2(closest_px)).xy;
    safeImageStore(closest_velocity_img, ivec2(px), vec4(reproj_xy, 0, 0));
    vec2 history_uv = uv + reproj_xy;

#if 0
    vec4 history_packed = image_sample_catmull_rom(
        TextureImage::from_parts(history_tex, output_tex_size.xy),
        history_uv,
        HistoryRemap::create()
    );
#elif 1
    vec4 history_packed = image_sample_catmull_rom_5tap(HistoryRemap_remap)(
        history_tex, g_sampler_llc, history_uv, output_tex_size.xy);
#else
    vec4 history_packed = fetch_history(history_uv);
#endif

    vec3 history = history_packed.rgb;
    float history_coverage = max(0.0, history_packed.a);

    safeImageStore(reprojected_history_img, ivec2(gl_GlobalInvocationID.xy), vec4(history, history_coverage));
}
