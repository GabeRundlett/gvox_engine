#include "common/buffers.hlsl"

#include "common/impl/voxel_world.hlsl"

struct Push {
    uint globals_id;
    uint input_id;
};
[[vk::push_constant]] const Push p;

// clang-format off
[numthreads(8, 8, 8)] void main(uint3 block_offset: SV_DispatchThreadID) {
    // clang-format on

    StructuredBuffer<Globals> globals = daxa::getBuffer<Globals>(p.globals_id);
    StructuredBuffer<Input> input = daxa::getBuffer<Input>(p.input_id);
    globals[0].game.voxel_world.chunkgen(block_offset, input[0]);
}
