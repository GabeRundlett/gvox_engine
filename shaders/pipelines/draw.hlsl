#include "shared.inl"

#include "common/buffers.hlsl"

#include "utils/rand.hlsl"
#include "utils/tonemapping.hlsl"

#include "common/impl/game/_drawing.hlsl"

[[vk::push_constant]] const DrawPush p;

#define MIXING_FACTOR 0.5

float3 rgb2hsv(float3 c) {
    float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = lerp(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
    float4 q = lerp(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

float3 hsv2rgb(float3 c) {
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * lerp(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// clang-format off
[numthreads(8, 8, 1)] void main(uint3 pixel_i: SV_DispatchThreadID) {
    // clang-format on

    StructuredBuffer<GpuGlobals> globals = daxa::get_StructuredBuffer<GpuGlobals>(p.globals_buffer_id);
    StructuredBuffer<GpuInput> input = daxa::get_StructuredBuffer<GpuInput>(p.input_buffer_id);

    if (pixel_i.x >= input[0].render_size.x ||
        pixel_i.y >= input[0].render_size.y)
        return;

    RWTexture2D<float4> col_image_out = daxa::get_RWTexture2D<float4>(p.render_color_image_id);
    RWTexture2D<float2> mot_image_out = daxa::get_RWTexture2D<float2>(p.render_motion_image_id);
    RWTexture2D<float> dep_image_out = daxa::get_RWTexture2D<float>(p.render_depth_image_id);

    float start_depth = dep_image_out[uint2(pixel_i.xy / 2) * 2 + uint2(1, 0)];

    const float THETA = globals[0].game.player.camera.fov * 3.14159f / 360.0f / (input[0].render_size.y / 4);
    const float MAX_D = (1.0 / tan(THETA));

    start_depth = clamp(start_depth - 1, 0, MAX_D);

    DrawSample draw_sample = globals[0].game.draw(input[0], pixel_i.xy, start_depth);

    float4 out_col;

    float3 col = draw_sample.col;
    out_col = float4(tonemap<1>(col), 1);
    col_image_out[pixel_i.xy] = out_col;

    // write zero to depth value (instead of draw_sample.depth)
    dep_image_out[pixel_i.xy] = draw_sample.depth;
}
