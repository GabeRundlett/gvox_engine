#include "shared.inl"

#include "common/buffers.hlsl"
#include "common/impl/game/_update.hlsl"

[[vk::push_constant]] const PerframePush p;

[numthreads(1, 1, 1)] void main() {
    StructuredBuffer<GpuGlobals> globals = daxa::get_StructuredBuffer<GpuGlobals>(p.globals_buffer_id);
    StructuredBuffer<GpuInput> input = daxa::get_StructuredBuffer<GpuInput>(p.input_buffer_id);
    StructuredBuffer<GpuOutput> output = daxa::get_StructuredBuffer<GpuOutput>(p.output_buffer_id);

    globals[0].game.update(input, output[0]);
}
