#include "common/buffers.hlsl"

#include "common/impl/voxel_world/_update.hlsl"

struct Push {
    daxa::BufferId globals_id;
    daxa::BufferId input_id;
};
[[vk::push_constant]] const Push p;

// clang-format off
[numthreads(8, 8, 8)] void main(uint3 block_offset: SV_DispatchThreadID) {
    // clang-format on

    StructuredBuffer<Globals> globals = daxa::get_StructuredBuffer<Globals>(p.globals_id);
    StructuredBuffer<Input> input = daxa::get_StructuredBuffer<Input>(p.input_id);
    globals[0].game.voxel_world.chunkgen(block_offset, input[0]);
}
