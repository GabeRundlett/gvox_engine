#include "shared.inl"

#include "common/buffers.hlsl"

#include "common/impl/voxel_world/_update.hlsl"

[[vk::push_constant]] const ChunkgenPush p;

// clang-format off
[numthreads(8, 8, 8)] void main(uint3 block_offset: SV_DispatchThreadID) {
    // clang-format on

    StructuredBuffer<GpuGlobals> globals = daxa::get_StructuredBuffer<GpuGlobals>(p.globals_buffer_id);
    StructuredBuffer<GpuInput> input = daxa::get_StructuredBuffer<GpuInput>(p.input_buffer_id);
    globals[0].game.voxel_world.chunkgen(block_offset, input[0]);
}
