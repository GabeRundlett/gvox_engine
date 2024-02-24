#include <renderer/kajiya/taa.inl>

#include <utilities/gpu/math.glsl>
#include <g_samplers>
#include "../inc/safety.glsl"

DAXA_DECL_PUSH_CONSTANT(TaaProbFilter2ComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewIndex prob_filtered1_img = push.uses.prob_filtered1_img;
daxa_ImageViewIndex prob_filtered2_img = push.uses.prob_filtered2_img;

float fetch_input(ivec2 px) {
    return safeTexelFetch(prob_filtered1_img, px, 0).r;
}

layout(local_size_x = TAA_WG_SIZE_X, local_size_y = TAA_WG_SIZE_Y, local_size_z = 1) in;
void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    float prob = fetch_input(px);

    vec2 weighted_prob = vec2(0);
    const float SQUISH_STRENGTH = 10;

    const int k = 2;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float neighbor_prob = fetch_input(px + ivec2(x, y) * 2);
            weighted_prob += vec2(exponential_squish(neighbor_prob, SQUISH_STRENGTH), 1);
        }
    }

    prob = exponential_unsquish(weighted_prob.x / weighted_prob.y, SQUISH_STRENGTH);

    // prob = min(prob, WaveReadLaneAt(prob, WaveGetLaneIndex() ^ 1));
    // prob = min(prob, WaveReadLaneAt(prob, WaveGetLaneIndex() ^ 8));

    safeImageStore(prob_filtered2_img, ivec2(px), vec4(prob, 0, 0, 0));
}
