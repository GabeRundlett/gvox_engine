#include <renderer/kajiya/taa.inl>

#include <g_samplers>
#include "../inc/safety.glsl"

DAXA_DECL_PUSH_CONSTANT(TaaProbFilterComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewIndex input_prob_img = push.uses.input_prob_img;
daxa_ImageViewIndex prob_filtered1_img = push.uses.prob_filtered1_img;

float fetch_input(ivec2 px) {
    return safeTexelFetch(input_prob_img, px, 0).r;
}

layout(local_size_x = TAA_WG_SIZE_X, local_size_y = TAA_WG_SIZE_Y, local_size_z = 1) in;
void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    float prob = fetch_input(px);

    const int k = 1;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float neighbor_prob = fetch_input(px + ivec2(x, y));
            prob = max(prob, neighbor_prob);
        }
    }

    safeImageStore(prob_filtered1_img, ivec2(px), vec4(prob, 0, 0, 0));
}
