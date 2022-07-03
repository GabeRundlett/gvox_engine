
#include "drawing/common.hlsl"
#include "utils/intersect.hlsl"
#include "utils/noise.hlsl"
#include "utils/tonemapping.hlsl"
#include "chunk.hlsl"

#include "player.hlsl"

DAXA_DEFINE_BA_SAMPLER(void)

#include "drawing/ui.hlsl"
#include "drawing/world.hlsl"

[numthreads(8, 8, 1)] void main(uint3 pixel_i
                                : SV_DispatchThreadID) {
    StructuredBuffer<Globals> globals = daxa::getBuffer<Globals>(p.globals_sb);
    StructuredBuffer<PlayerBuffer> player_buffer = daxa::getBuffer<PlayerBuffer>(p.player_buf_id);

    if (pixel_i.x >= globals[0].frame_dim.x ||
        pixel_i.y >= globals[0].frame_dim.y)
        return;

    const float2 subsamples = float2(SUBSAMPLE_N, SUBSAMPLE_N);
    const float2 inv_subsamples = 1 / subsamples;
    float2 inv_frame_dim = 1 / float2(globals[0].frame_dim);
    float aspect = float(globals[0].frame_dim.x) * inv_frame_dim.y; // * 0.5;
    int2 i_uv = int2(pixel_i.xy); // * int2(2, 1);
    // if (pixel_i.x < globals[0].frame_dim.x / 2) {
    // } else {
    //     i_uv.x -= globals[0].frame_dim.x;
    // }
    float uv_rand_offset = globals[0].time;
    float3 color = float3(0, 0, 0);
    float depth = 100000;

    float2 uv_offset =
#if JITTER_VIEW || ENABLE_TAA
        // (pixel_i.x > globals[0].frame_dim.x / 2) *
        float2(rand(float2(i_uv + uv_rand_offset + 10)),
               rand(float2(i_uv + uv_rand_offset)));
#else
        float2(0, 0);
#endif
    float2 uv = (float2(i_uv) + uv_offset * inv_subsamples) * inv_frame_dim * 2 - 1;

#if SHOW_UI
    if (!draw_ui(globals, uv, aspect, color))
#endif
        draw_world(globals, player_buffer, uv, pixel_i, subsamples, inv_subsamples, inv_frame_dim, aspect, color, depth);

#if SHOW_UI
    draw_rect(pixel_i.xy, color, globals[0].frame_dim.x / 2 - 0, globals[0].frame_dim.y / 2 - 4, 1, 9);
    draw_rect(pixel_i.xy, color, globals[0].frame_dim.x / 2 - 4, globals[0].frame_dim.y / 2 - 0, 9, 1);
#endif

    // draw_rect(pixel_i.xy, color, globals[0].frame_dim.x / 4 - 0, globals[0].frame_dim.y / 2 - 4, 1, 9);
    // draw_rect(pixel_i.xy, color, globals[0].frame_dim.x / 4 - 4, globals[0].frame_dim.y / 2 - 0, 9, 1);
    // draw_rect(pixel_i.xy, color, globals[0].frame_dim.x / 4 + globals[0].frame_dim.x / 2 - 0, globals[0].frame_dim.y / 2 - 4, 1, 9);
    // draw_rect(pixel_i.xy, color, globals[0].frame_dim.x / 4 + globals[0].frame_dim.x / 2 - 4, globals[0].frame_dim.y / 2 - 0, 9, 1);

    RWTexture2D<float4> output_image = daxa::getRWTexture2D<float4>(p.output_image_i);
    float4 prev_val = output_image[pixel_i.xy];
#if ENABLE_TAA
    // if (pixel_i.x > globals[0].frame_dim.x / 2) {
        output_image[pixel_i.xy] = float4(tonemap<1>(tonemap<-1>(prev_val.rgb) * (1.0 - TAA_MIXING) + color * 1 * TAA_MIXING), 1);
    // } else 
    // output_image[pixel_i.xy] = new_val;
#else
    output_image[pixel_i.xy] = float4(max(color, 0), 1);
#endif
#if ENABLE_TAA
    // }
#endif
}
