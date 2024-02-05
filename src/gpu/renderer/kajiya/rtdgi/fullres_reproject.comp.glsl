#include <shared/renderer/rtdgi.inl>

// #include <utils/samplers.glsl>
#include <utils/color.glsl>
// #include <utils/uv.glsl>
#include <utils/bilinear.glsl>
// #include <utils/frame_constants.glsl>
#include <utils/image.glsl>
#include <utils/camera.glsl>
#include <utils/safety.glsl>

DAXA_DECL_PUSH_CONSTANT(RtdgiFullresReprojectComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewIndex input_tex = push.uses.input_tex;
daxa_ImageViewIndex reprojection_tex = push.uses.reprojection_tex;
daxa_ImageViewIndex output_tex = push.uses.output_tex;

const bool USE_SHARPENING_HISTORY_FETCH = true;

// For `image_sample_catmull_rom`. Not applying the actual color remap here to reduce cost.
vec4 HistoryRemap_remap(vec4 v) {
    return v;
}
image_sample_catmull_rom_TEMPLATE(HistoryRemap_remap);

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    uvec2 px = gl_GlobalInvocationID.xy;
    vec2 uv = get_uv(px, push.output_tex_size);

    vec4 center = safeTexelFetch(input_tex, ivec2(px), 0);
    vec4 reproj = safeTexelFetch(reprojection_tex, ivec2(px), 0);
    vec2 prev_uv = uv + reproj.xy;

    uint quad_reproj_valid_packed = uint(reproj.z * 15.0 + 0.5);

    // For the sharpening (4x4) kernel, we need to know whether our neighbors are valid too,
    // as otherwise we end up over-sharpening with fake history (moving edges rather than scene features).
    const uvec4 reproj_valid_neigh =
        uvec4(textureGather(daxa_sampler2D(reprojection_tex, deref(gpu_input).sampler_nnc), uv + 0.5 * sign(prev_uv) * push.output_tex_size.zw) * 15.0 + 0.5);

    vec4 history = 0.0.xxxx;

    if (0 == quad_reproj_valid_packed) {
        // Everything invalid
    } else if (15 == quad_reproj_valid_packed) {
        if (USE_SHARPENING_HISTORY_FETCH && all(equal(reproj_valid_neigh, uvec4(15)))) {
            history = max(0.0.xxxx, image_sample_catmull_rom(HistoryRemap_remap)(
                                        input_tex,
                                        prev_uv,
                                        push.output_tex_size));
        } else {
            history = textureLod(daxa_sampler2D(input_tex, deref(gpu_input).sampler_lnc), prev_uv, 0);
        }
    } else {
        // Only some samples are valid. Only include those, and don't do a sharpening fetch here.

        vec4 quad_reproj_valid = vec4(notEqual((quad_reproj_valid_packed & uvec4(1, 2, 4, 8)), uvec4(0)));

        const Bilinear bilinear = get_bilinear_filter(prev_uv, push.output_tex_size.xy);
        vec4 s00 = safeTexelFetch(input_tex, ivec2(ivec2(bilinear.origin) + ivec2(0, 0)), 0);
        vec4 s10 = safeTexelFetch(input_tex, ivec2(ivec2(bilinear.origin) + ivec2(1, 0)), 0);
        vec4 s01 = safeTexelFetch(input_tex, ivec2(ivec2(bilinear.origin) + ivec2(0, 1)), 0);
        vec4 s11 = safeTexelFetch(input_tex, ivec2(ivec2(bilinear.origin) + ivec2(1, 1)), 0);

        vec4 weights = get_bilinear_custom_weights(bilinear, quad_reproj_valid);

        if (dot(weights, vec4(1.0)) > 1e-5) {
            history = apply_bilinear_custom_weights(s00, s10, s01, s11, weights, true);
        }
    }

    safeImageStore(output_tex, ivec2(px), history);
}
