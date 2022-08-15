#include "common/buffers.hlsl"

#include "utils/rand.hlsl"
#include "utils/tonemapping.hlsl"

#include "common/impl/game.hlsl"

struct Push {
    uint globals_id;
    uint input_id;
    uint col_image_id_in, col_image_id_out;
    uint pos_image_id_in, pos_image_id_out;
    uint nrm_image_id_in, nrm_image_id_out;
};

[[vk::push_constant]] const Push p;

#define MIXING_FACTOR 0.5

float3 rgb2hsv(float3 c)
{
    float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = lerp(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
    float4 q = lerp(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

float3 hsv2rgb(float3 c)
{
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * lerp(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// clang-format off
[numthreads(8, 8, 1)] void main(uint3 pixel_i: SV_DispatchThreadID) {
    // clang-format on

    StructuredBuffer<Globals> globals = daxa::getBuffer<Globals>(p.globals_id);
    StructuredBuffer<Input> input = daxa::getBuffer<Input>(p.input_id);

    if (pixel_i.x >= input[0].frame_dim.x ||
        pixel_i.y >= input[0].frame_dim.y)
        return;

    RWTexture2D<float4> col_image_in = daxa::getRWTexture2D<float4>(p.col_image_id_in);
    RWTexture2D<float4> col_image_out = daxa::getRWTexture2D<float4>(p.col_image_id_out);
    RWTexture2D<float4> pos_image_in = daxa::getRWTexture2D<float4>(p.pos_image_id_in);
    RWTexture2D<float4> pos_image_out = daxa::getRWTexture2D<float4>(p.pos_image_id_out);
    RWTexture2D<float4> nrm_image_in = daxa::getRWTexture2D<float4>(p.nrm_image_id_in);
    RWTexture2D<float4> nrm_image_out = daxa::getRWTexture2D<float4>(p.nrm_image_id_out);

    float start_depth = pos_image_in[uint2(pixel_i.xy / 2) * 2 + uint2(1, 0)].a;

    const float THETA = globals[0].game.player.camera.fov * 3.14159f / 360.0f / (input[0].frame_dim.y / 4);
    const float MAX_D = (1.0 / tan(THETA));

    start_depth = clamp(start_depth - 1, 0, MAX_D);

    DrawSample draw_sample = globals[0].game.draw(input[0], pixel_i.xy, start_depth);
    float3 prev_col = col_image_in[pixel_i.xy].rgb;
    prev_col = tonemap<-1>(prev_col);

    float4 out_col;
    // float fog = draw_sample.depth * 0.01;
    // fog = clamp(pow(1 - exp(-fog), 4), 0, 1);
    // draw_sample.col = lerp(draw_sample.col, float3(1, 1, 1), fog);

    float3 col = draw_sample.col;
    // col = rgb2hsv(col);
    // col.x = frac(col.x + 0.5);
    // col = hsv2rgb(col);
    out_col = float4(tonemap<1>(col * MIXING_FACTOR + prev_col * (1.0 - MIXING_FACTOR)), draw_sample.lifetime);
    // out_col = float4(tonemap<1>(draw_sample.pos), draw_sample.lifetime);
    // out_col = float4(tonemap<1>(draw_sample.nrm), draw_sample.lifetime);
    // out_col = float4(tonemap<1>(fog), draw_sample.lifetime);

    col_image_out[pixel_i.xy] = out_col;
    nrm_image_out[pixel_i.xy] = float4(draw_sample.nrm, 0);

    // write zero to depth value (instead of draw_sample.depth)
    pos_image_out[pixel_i.xy] = float4(draw_sample.pos, 0);
}
