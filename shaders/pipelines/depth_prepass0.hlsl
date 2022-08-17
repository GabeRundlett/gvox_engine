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

    float start_depth = pos_image_out[pixel_i.xy * p.scl].a;
    const float THETA = globals[0].game.player.camera.fov * 3.14159f / 360.0f / input[0].frame_dim.y * p.scl;
    const float MAX_D = (1.0 / tan(THETA));
    start_depth = clamp(start_depth - 1, 0, MAX_D);

    float depth = globals[0].game.draw_depth((pixel_i.xy * 2) * p.scl, 0, input[0].max_steps());
    pos_image_out[(pixel_i.xy * 2) * p.scl] = float4(0, 0, 0, depth);
}
