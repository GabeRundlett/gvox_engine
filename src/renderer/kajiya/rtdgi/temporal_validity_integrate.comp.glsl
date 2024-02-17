#include <renderer/kajiya/rtdgi.inl>

// #include "../inc/frame_constants.glsl"
// #include <utilities/gpu/hash.glsl>
// #include "../inc/quasi_random.glsl"
#include <utilities/gpu/math.glsl>
#include "rtdgi_restir_settings.glsl"
#include "../inc/safety.glsl"

DAXA_DECL_PUSH_CONSTANT(RtdgiValidityIntegrateComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewIndex input_tex = push.uses.input_tex;
daxa_ImageViewIndex history_tex = push.uses.history_tex;
daxa_ImageViewIndex reprojection_tex = push.uses.reprojection_tex;
daxa_ImageViewIndex half_view_normal_tex = push.uses.half_view_normal_tex;
daxa_ImageViewIndex half_depth_tex = push.uses.half_depth_tex;
daxa_ImageViewIndex output_tex = push.uses.output_tex;

#define USE_SSAO_STEERING 1
#define USE_DYNAMIC_KERNEL_RADIUS 0

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    uvec2 px = gl_GlobalInvocationID.xy;
    vec2 invalid_blurred = vec2(0);

    const float center_depth = safeTexelFetch(half_depth_tex, ivec2(px), 0).r;
    const vec4 reproj = safeTexelFetch(reprojection_tex, ivec2(px * 2), 0);

    if (RESTIR_USE_PATH_VALIDATION) {
        {
            const int k = 2;
            for (int y = -k; y <= k; ++y) {
                for (int x = -k; x <= k; ++x) {
                    const ivec2 offset = ivec2(x, y);
                    // float w = 1;
                    float w = exp2(-0.1 * dot(vec2(offset), vec2(offset)));
                    invalid_blurred += vec2(safeTexelFetch(input_tex, ivec2(px + offset), 0).x, 1) * w;
                }
            }
        }
        invalid_blurred /= invalid_blurred.y;
        invalid_blurred.x = mix(invalid_blurred.x, subgroupBroadcast(invalid_blurred.x, gl_SubgroupInvocationID ^ 2), 0.5);
        invalid_blurred.x = mix(invalid_blurred.x, subgroupBroadcast(invalid_blurred.x, gl_SubgroupInvocationID ^ 16), 0.5);
        // invalid_blurred.x = max(invalid_blurred.x, WaveActiveSum(invalid_blurred.x) / 64.0, 0.25);

        const vec2 reproj_rand_offset = vec2(0.0);

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
                const ivec2 offset = ivec2(x, y);
                const ivec2 sample_px = ivec2(px) * 2 + offset;
                const ivec2 sample_px_half = ivec2(px) + offset / 2;
                const vec4 reproj = safeTexelFetch(reprojection_tex, ivec2(sample_px), 0);
                const float sample_depth = safeTexelFetch(half_depth_tex, ivec2(sample_px_half), 0).r;

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

    const vec2 reproj_px = px + push.gbuffer_tex_size.xy * reproj.xy / 2 + 0.5;
    float history = 0;

    const int sample_count = 8;
    float ang_off = uint_to_u01_float(hash3(uvec3(px, deref(gpu_input).frame_index))) * M_PI * 2;

    for (uint sample_i = 0; sample_i < sample_count; ++sample_i) {
        float ang = (sample_i + ang_off) * GOLDEN_ANGLE;
        float radius = float(sample_i) * 1.0;
        vec2 sample_offset = vec2(cos(ang), sin(ang)) * radius;
        const ivec2 sample_px = ivec2(reproj_px + sample_offset);
        history += safeTexelFetch(history_tex, ivec2(sample_px), 0).x;
    }

    history /= sample_count;

    /*float history = (
        history_tex[reproj_px] +
        history_tex[reproj_px + ivec2(-4, 0)] +
        history_tex[reproj_px + ivec2(4, 0)] +
        history_tex[reproj_px + ivec2(0, 4)] +
        history_tex[reproj_px + ivec2(0, -4)]
    ) / 5;*/

    // float history = history_tex[reproj_px];

    safeImageStore(output_tex, ivec2(px), vec4(max(history * 0.75, invalid_blurred.x),
                                               // invalid_blurred.x,
                                               safeTexelFetch(input_tex, ivec2(px), 0).x, 0.0, 0.0));
}
