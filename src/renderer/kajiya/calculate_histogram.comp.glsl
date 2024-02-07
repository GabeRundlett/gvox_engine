#include <renderer/kajiya/calculate_histogram.inl>
#include <utils/camera.glsl>
#include <utils/color.glsl>

DAXA_DECL_PUSH_CONSTANT(CalculateHistogramComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewIndex input_tex = push.uses.input_tex;
daxa_RWBufferPtr(daxa_u32) output_buffer = push.uses.output_buffer;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    uvec2 px = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(px, push.input_extent))) {
        return;
    }

    vec3 input_tex_value = texelFetch(daxa_texture2D(input_tex), ivec2(px), 0).rgb;
    float log_lum = log2(max(1e-20, sRGB_to_luminance(input_tex_value) / deref(gpu_input).pre_exposure));

    const float t = clamp((log_lum - LUMINANCE_HISTOGRAM_MIN_LOG2) / (LUMINANCE_HISTOGRAM_MAX_LOG2 - LUMINANCE_HISTOGRAM_MIN_LOG2), 0, 1);
    const uint bin = min(uint(t * 256), 255);

    const vec2 uv = vec2(px + 0.5) / push.input_extent;
    const float infl = exp(-8 * pow(length(uv - 0.5), 2));
    const uint quantized_infl = uint(infl * 256.0);

    atomicAdd(deref(output_buffer[bin]), quantized_infl);
}
