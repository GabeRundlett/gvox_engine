#include <shared/app.inl>
#include <utils/safety.glsl>

#if ShadowBitPackComputeShader

uvec2 FFX_DNSR_Shadows_GetBufferDimensions() {
    return uvec2(push.input_tex_size.xy);
}

bool FFX_DNSR_Shadows_HitsLight(uvec2 px, uvec2 gtid, uvec2 gid) {
    float shadow_value = safeTexelFetch(input_tex, daxa_i32vec2(px), 0).r;
    return shadow_value != 0.0;
}

void FFX_DNSR_Shadows_WriteMask(uint linear_tile_index, uint value) {
    const ivec2 tile = ivec2(
        linear_tile_index % push.bitpacked_shadow_mask_extent.x,
        linear_tile_index / push.bitpacked_shadow_mask_extent.x);

    safeImageStoreU(output_tex, tile, uvec4(value, 0, 0, 0));
}

#include "ffx/ffx_denoiser_shadows_prepare.glsl"

layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;
void main() {
    FFX_DNSR_Shadows_PrepareShadowMask(gl_LocalInvocationID.xy, gl_WorkGroupID.xy);
}

#endif

#if ShadowTemporalFilterComputeShader

uvec2 FFX_DNSR_Shadows_GetBufferDimensions() {
    return uvec2(push.input_tex_size.xy);
}

vec2 FFX_DNSR_Shadows_GetInvBufferDimensions() {
    return push.input_tex_size.zw;
}

mat4 FFX_DNSR_Shadows_GetProjectionInverse() {
    return deref(globals).player.cam.clip_to_view;
}

mat4 FFX_DNSR_Shadows_GetReprojectionMatrix() {
    return deref(globals).player.cam.clip_to_prev_clip;
}

mat4 FFX_DNSR_Shadows_GetViewProjectionInverse() {
    // TODO: replace the temporal component in the denoiser
    return deref(globals).player.cam.clip_to_view;
}

vec3 get_eye_position() {
    vec4 eye_pos_h = deref(globals).player.cam.view_to_world * vec4(0, 0, 0, 1);
    return eye_pos_h.xyz / eye_pos_h.w;
}

vec3 FFX_DNSR_Shadows_GetEye() {
    return get_eye_position();
}

vec3 FFX_DNSR_Shadows_ReadNormals(uvec2 px) {
    // TODO
    return vec3(0, 0, 1);
}

float FFX_DNSR_Shadows_ReadDepth(uvec2 px) {
    // TODO
    return 0.5;
}

float FFX_DNSR_Shadows_ReadPreviousDepth(uvec2 px) {
    // TODO
    return 0.5;
}

bool FFX_DNSR_Shadows_IsShadowReciever(uvec2 px) {
    // TODO
    return true;
}

vec2 FFX_DNSR_Shadows_ReadVelocity(uvec2 px) {
    // TODO
    return 0.0.xx;
}

void FFX_DNSR_Shadows_WriteMetadata(uint linear_tile_index, uint mask) {
    const ivec2 tile = ivec2(
        linear_tile_index % push.bitpacked_shadow_mask_extent.x,
        linear_tile_index / push.bitpacked_shadow_mask_extent.x);

    safeImageStoreU(meta_output_tex, tile, uvec4(mask, 0, 0, 0));
}

uint FFX_DNSR_Shadows_ReadRaytracedShadowMask(uint linear_tile_index) {
    const ivec2 tile = ivec2(
        linear_tile_index % push.bitpacked_shadow_mask_extent.x,
        linear_tile_index / push.bitpacked_shadow_mask_extent.x);

    return safeTexelFetchU(bitpacked_shadow_mask_tex, tile, 0).r;
}

void FFX_DNSR_Shadows_WriteReprojectionResults(uvec2 px, vec2 shadow_clamped_variance) {
    safeImageStore(temporal_output_tex, ivec2(px), vec4(shadow_clamped_variance, 0, 0));
}

void FFX_DNSR_Shadows_WriteMoments(uvec2 px, vec4 moments) {
    // Don't accumulate more samples than a certain number,
    // so that our variance estimate is quick, and contact shadows turn crispy sooner.
    moments.z = min(moments.z, 32);

    safeImageStore(output_moments_tex, ivec2(px), moments);
}

float FFX_DNSR_Shadows_HitsLight(uvec2 px) {
    return safeTexelFetch(shadow_mask_tex, daxa_i32vec2(px), 0).r;
}

float soft_color_clamp(float center, float history, float ex, float dev) {
    // Sort of like the color bbox clamp, but with a twist. In noisy surrounds, the bbox becomes
    // very large, and then the clamp does nothing, especially with a high multiplier on std. deviation.
    //
    // Instead of a hard clamp, this will smoothly bring the value closer to the center,
    // thus over time reducing disocclusion artifacts.
    float history_dist = abs(history - ex) / max(abs(history * 0.1), dev);

    float closest_pt = clamp(history, center - dev, center + dev);
    return mix(history, closest_pt, smoothstep((1.0), (3.0), history_dist));
}

daxa_f32vec4 HistoryRemap_remap(daxa_f32vec4 v) {
    return v;
}
daxa_f32vec4 cubic_hermite(daxa_f32vec4 A, daxa_f32vec4 B, daxa_f32vec4 C, daxa_f32vec4 D, float t) {
    float t2 = t * t;
    float t3 = t * t * t;
    daxa_f32vec4 a = -A / 2.0 + (3.0 * B) / 2.0 - (3.0 * C) / 2.0 + D / 2.0;
    daxa_f32vec4 b = A - (5.0 * B) / 2.0 + 2.0 * C - D / 2.0;
    daxa_f32vec4 c = -A / 2.0 + C / 2.0;
    daxa_f32vec4 d = B;

    return a * t3 + b * t2 + c * t + d;
}

#define REMAP_FUNC HistoryRemap_remap
daxa_f32vec4 image_sample_catmull_rom(daxa_ImageViewId img, daxa_f32vec2 P, daxa_f32vec4 img_size) {
    // https://www.shadertoy.com/view/MllSzX

    daxa_f32vec2 pixel = P * img_size.xy + 0.5;
    daxa_f32vec2 c_onePixel = img_size.zw;
    daxa_f32vec2 c_twoPixels = c_onePixel * 2.0;

    daxa_f32vec2 frc = fract(pixel);
    // pixel = floor(pixel) / output_tex_size.xy - daxa_f32vec2(c_onePixel/2.0);
    daxa_i32vec2 ipixel = daxa_i32vec2(pixel) - 1;

    daxa_f32vec4 C00 = REMAP_FUNC(safeTexelFetch(img, ipixel + daxa_i32vec2(-1, -1), 0));
    daxa_f32vec4 C10 = REMAP_FUNC(safeTexelFetch(img, ipixel + daxa_i32vec2(0, -1), 0));
    daxa_f32vec4 C20 = REMAP_FUNC(safeTexelFetch(img, ipixel + daxa_i32vec2(1, -1), 0));
    daxa_f32vec4 C30 = REMAP_FUNC(safeTexelFetch(img, ipixel + daxa_i32vec2(2, -1), 0));

    daxa_f32vec4 C01 = REMAP_FUNC(safeTexelFetch(img, ipixel + daxa_i32vec2(-1, 0), 0));
    daxa_f32vec4 C11 = REMAP_FUNC(safeTexelFetch(img, ipixel + daxa_i32vec2(0, 0), 0));
    daxa_f32vec4 C21 = REMAP_FUNC(safeTexelFetch(img, ipixel + daxa_i32vec2(1, 0), 0));
    daxa_f32vec4 C31 = REMAP_FUNC(safeTexelFetch(img, ipixel + daxa_i32vec2(2, 0), 0));

    daxa_f32vec4 C02 = REMAP_FUNC(safeTexelFetch(img, ipixel + daxa_i32vec2(-1, 1), 0));
    daxa_f32vec4 C12 = REMAP_FUNC(safeTexelFetch(img, ipixel + daxa_i32vec2(0, 1), 0));
    daxa_f32vec4 C22 = REMAP_FUNC(safeTexelFetch(img, ipixel + daxa_i32vec2(1, 1), 0));
    daxa_f32vec4 C32 = REMAP_FUNC(safeTexelFetch(img, ipixel + daxa_i32vec2(2, 1), 0));

    daxa_f32vec4 C03 = REMAP_FUNC(safeTexelFetch(img, ipixel + daxa_i32vec2(-1, 2), 0));
    daxa_f32vec4 C13 = REMAP_FUNC(safeTexelFetch(img, ipixel + daxa_i32vec2(0, 2), 0));
    daxa_f32vec4 C23 = REMAP_FUNC(safeTexelFetch(img, ipixel + daxa_i32vec2(1, 2), 0));
    daxa_f32vec4 C33 = REMAP_FUNC(safeTexelFetch(img, ipixel + daxa_i32vec2(2, 2), 0));

    daxa_f32vec4 CP0X = cubic_hermite(C00, C10, C20, C30, frc.x);
    daxa_f32vec4 CP1X = cubic_hermite(C01, C11, C21, C31, frc.x);
    daxa_f32vec4 CP2X = cubic_hermite(C02, C12, C22, C32, frc.x);
    daxa_f32vec4 CP3X = cubic_hermite(C03, C13, C23, C33, frc.x);

    return cubic_hermite(CP0X, CP1X, CP2X, CP3X, frc.y);
}
#undef REMAP_FUNC

vec4 FFX_DNSR_Shadows_ReadPreviousMomentsBuffer(vec2 uv) {
#if 1
    vec4 moments = image_sample_catmull_rom(
        prev_moments_tex,
        uv,
        push.input_tex_size);
    // Clamp EX2 and sample count
    moments.yz = max(vec2(0), moments.yz);
    return moments;
#else
    return prev_moments_tex.SampleLevel(sampler_lnc, uv, 0);
#endif
}

float FFX_DNSR_Shadows_ReadHistory(vec2 uv) {
#if 1
    return image_sample_catmull_rom(
               prev_accum_tex,
               uv,
               push.input_tex_size)
        .x;
#else
    return prev_accum_tex.SampleLevel(sampler_lnc, uv, 0).x;
#endif
}

bool FFX_DNSR_Shadows_IsFirstFrame() {
    return deref(gpu_input).frame_index == 0;
}

#include "ffx/ffx_denoiser_shadows_tileclassification.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    FFX_DNSR_Shadows_TileClassification(gl_LocalInvocationIndex, gl_WorkGroupID.xy);
}

#endif

#if ShadowSpatialFilterComputeShader

uvec2 FFX_DNSR_Shadows_GetBufferDimensions() {
    return uvec2(push.input_tex_size.xy);
}

vec2 FFX_DNSR_Shadows_GetInvBufferDimensions() {
    return push.input_tex_size.zw;
}

mat4 FFX_DNSR_Shadows_GetProjectionInverse() {
    return deref(globals).player.cam.clip_to_view;
}

float FFX_DNSR_Shadows_GetDepthSimilaritySigma() {
    return 0.01;
}

bool FFX_DNSR_Shadows_IsShadowReciever(uvec2 px) {
    return safeTexelFetch(depth_tex, ivec2(px), 0).r != 0;
}

vec3 FFX_DNSR_Shadows_ReadNormals(uvec2 px) {
    vec3 normal_vs = safeTexelFetch(geometric_normal_tex, ivec2(px), 0).xyz * 2.0 - 1.0;
    return vec3(normal_vs);
}

float FFX_DNSR_Shadows_ReadDepth(uvec2 px) {
    return safeTexelFetch(depth_tex, ivec2(px), 0).r;
}

vec2 FFX_DNSR_Shadows_ReadInput(uvec2 px) {
    return safeTexelFetch(input_tex, ivec2(px), 0).xy;
}

uint FFX_DNSR_Shadows_ReadTileMetaData(uint linear_tile_index) {
    const ivec2 tile = ivec2(
        linear_tile_index % push.bitpacked_shadow_mask_extent.x,
        linear_tile_index / push.bitpacked_shadow_mask_extent.x);

    return safeTexelFetchU(meta_tex, tile, 0).x;
}

#include "ffx/ffx_denoiser_shadows_filter.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    const uint pass_idx = 0;

    bool write_results = true;
    vec2 filter_output = FFX_DNSR_Shadows_FilterSoftShadowsPass(gl_WorkGroupID.xy, gl_LocalInvocationID.xy, gl_GlobalInvocationID.xy, write_results, pass_idx, push.step_size);

    if (write_results) {
        safeImageStore(output_tex, ivec2(gl_GlobalInvocationID.xy), vec4(max(vec2(0.0), filter_output), 0, 0));
    }
}
#endif
