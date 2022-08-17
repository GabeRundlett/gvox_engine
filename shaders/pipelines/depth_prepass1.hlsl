#include "common/buffers.hlsl"

#include "utils/rand.hlsl"
#include "utils/tonemapping.hlsl"

#include "common/impl/game/_drawing.hlsl"

struct Push {
    uint globals_id;
    uint input_id;
    uint pos_image_id_in, pos_image_id_out;
    uint scl;
};

[[vk::push_constant]] const Push p;

// clang-format off
[numthreads(8, 8, 1)] void main(uint3 pixel_i: SV_DispatchThreadID) {
    // clang-format on

    StructuredBuffer<Globals> globals = daxa::getBuffer<Globals>(p.globals_id);
    StructuredBuffer<Input> input = daxa::getBuffer<Input>(p.input_id);

    if (pixel_i.x * p.scl * 2 >= input[0].frame_dim.x ||
        pixel_i.y * p.scl * 2 >= input[0].frame_dim.y)
        return;

    RWTexture2D<float4> pos_image_in = daxa::getRWTexture2D<float4>(p.pos_image_id_in);
    RWTexture2D<float4> pos_image_out = daxa::getRWTexture2D<float4>(p.pos_image_id_out);

    float d1 = pos_image_out[(pixel_i.xy * 2 + uint2(0, 0)) * p.scl].a;
    float d2 = pos_image_out[(pixel_i.xy * 2 + uint2(2, 0)) * p.scl].a;
    float d3 = pos_image_out[(pixel_i.xy * 2 + uint2(0, 2)) * p.scl].a;
    float d4 = pos_image_out[(pixel_i.xy * 2 + uint2(2, 2)) * p.scl].a;

    float min_d = max(min(d1, min(d2, min(d3, d4))), 0);
    float depth = min_d;

    pos_image_in[(pixel_i.xy * 2 + uint2(1, 0)) * p.scl] = float4(0, 0, 0, depth);
}
