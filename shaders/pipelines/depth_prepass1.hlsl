#include "shared.inl"

#include "common/buffers.hlsl"

#include "utils/rand.hlsl"
#include "utils/tonemapping.hlsl"

#include "common/impl/game/_drawing.hlsl"

[[vk::push_constant]] const DepthPrepassPush p;

// clang-format off
[numthreads(8, 8, 1)] void main(uint3 pixel_i: SV_DispatchThreadID) {
    // clang-format on

    StructuredBuffer<GpuGlobals> globals = daxa::get_StructuredBuffer<GpuGlobals>(p.globals_buffer_id);
    StructuredBuffer<GpuInput> input = daxa::get_StructuredBuffer<GpuInput>(p.input_buffer_id);

    if (pixel_i.x * p.scl * 2 >= input[0].render_size.x ||
        pixel_i.y * p.scl * 2 >= input[0].render_size.y)
        return;

    RWTexture2D<float> dep_image_out = daxa::get_RWTexture2D<float>(p.render_depth_image_id);

    float d1 = dep_image_out[(pixel_i.xy * 2 + uint2(0, 0)) * p.scl];
    float d2 = dep_image_out[(pixel_i.xy * 2 + uint2(2, 0)) * p.scl];
    float d3 = dep_image_out[(pixel_i.xy * 2 + uint2(0, 2)) * p.scl];
    float d4 = dep_image_out[(pixel_i.xy * 2 + uint2(2, 2)) * p.scl];

    float min_d = max(min(d1, min(d2, min(d3, d4))), 0);
    float depth = min_d;

    dep_image_out[(pixel_i.xy * 2 + uint2(1, 0)) * p.scl] = depth;
}
