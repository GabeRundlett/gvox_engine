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

    float start_depth = 0;
    const float THETA = globals[0].game.player.camera.fov * 3.14159f / 360.0f / input[0].render_size.y * p.scl;
    const float MAX_D = (1.0 / tan(THETA));
    start_depth = clamp(start_depth - 1, 0, MAX_D);

    float depth = globals[0].game.draw_depth((pixel_i.xy * 2) * p.scl, 0);
    dep_image_out[(pixel_i.xy * 2) * p.scl] = depth;
}
