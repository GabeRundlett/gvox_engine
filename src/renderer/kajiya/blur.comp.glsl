#include <renderer/kajiya/blur.inl>
#include "inc/safety.glsl"

#if BlurComputeShader

DAXA_DECL_PUSH_CONSTANT(BlurComputePush, push)
daxa_ImageViewIndex input_tex = push.uses.input_tex;
daxa_ImageViewIndex output_tex = push.uses.output_tex;

const uint kernel_radius = 5;
const uint group_width = 64;
const uint vblur_window_size = (group_width + kernel_radius) * 2;

#define Color vec4
#define Color_swizzle xyzw

float gaussian_wt(float dst_px, float src_px) {
    float px_off = (dst_px + 0.5) * 2 - (src_px + 0.5);
    float sigma = kernel_radius * 0.5;
    return exp(-px_off * px_off / (sigma * sigma));
}

Color vblur(ivec2 dst_px, ivec2 src_px) {
    Color res = Color(0);
    float wt_sum = 0;

    for (uint y = 0; y <= kernel_radius * 2; ++y) {
        float wt = gaussian_wt(dst_px.y, src_px.y + y);
        res += safeTexelFetch(input_tex, src_px + ivec2(0, y), 0).Color_swizzle * wt;
        wt_sum += wt;
    }

    return res / wt_sum;
}

shared Color vblur_out[vblur_window_size];

void vblur_into_shmem(ivec2 dst_px, uint xfetch, uvec2 group_id) {
    ivec2 src_px = ivec2(group_id) * ivec2(group_width * 2, 2) + ivec2(xfetch - kernel_radius, -kernel_radius);
    vblur_out[xfetch] = vblur(dst_px, src_px);
}

layout(local_size_x = group_width, local_size_y = 1, local_size_z = 1) in;
void main() {
    uvec2 px = gl_GlobalInvocationID.xy;
    uvec2 px_within_group = gl_LocalInvocationID.xy;
    uvec2 group_id = gl_WorkGroupID.xy;

    for (uint xfetch = px_within_group.x; xfetch < vblur_window_size; xfetch += group_width) {
        vblur_into_shmem(ivec2(px), xfetch, group_id);
    }

    barrier();

    vec4 res = vec4(0);
    float wt_sum = 0;

    for (uint x = 0; x <= kernel_radius * 2; ++x) {
        float wt = gaussian_wt(px.x, px.x * 2 + x - kernel_radius);
        res.Color_swizzle += vblur_out[px_within_group.x * 2 + x] * wt;
        wt_sum += wt;
    }
    res /= wt_sum;

    safeImageStore(output_tex, ivec2(px), res);
}

#endif

#if RevBlurComputeShader

#include <g_samplers>

DAXA_DECL_PUSH_CONSTANT(RevBlurComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewIndex input_tail_tex = push.uses.input_tail_tex;
daxa_ImageViewIndex input_tex = push.uses.input_tex;
daxa_ImageViewIndex output_tex = push.uses.output_tex;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    uvec2 px = gl_GlobalInvocationID.xy;
    vec4 pyramid_col = safeTexelFetch(input_tail_tex, ivec2(px), 0);

    vec4 self_col = vec4(0);
    vec2 inv_size = vec2(1.0) / vec2(push.output_extent);

    // TODO: do a small Gaussian blur instead of this nonsense

    const int K = 1;

    for (int y = -K; y < K; ++y) {
        for (int x = -K; x < K; ++x) {
            vec2 uv = (vec2(px) + vec2(0.5) + vec2(x, y)) * inv_size;
            vec4 t_sample = textureLod(daxa_sampler2D(input_tex, g_sampler_lnc), uv, 0);
            self_col += t_sample;
        }
    }

    self_col /= float((2 * K + 1) * (2 * K + 1));
    float exponential_falloff = 0.6;

    safeImageStore(output_tex, ivec2(px), mix(self_col, pyramid_col, push.self_weight * exponential_falloff));
}

#endif
