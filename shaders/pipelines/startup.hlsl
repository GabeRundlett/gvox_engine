#include "shared.inl"

#include "common/buffers.hlsl"

#include "common/impl/game/_update.hlsl"

[[vk::push_constant]] const StartupPush p;

[numthreads(1, 1, 1)] void main() {
    StructuredBuffer<GpuGlobals> globals = daxa::get_StructuredBuffer<GpuGlobals>(p.globals_buffer_id);
    globals[0].game.default_init();

    globals[0].game.init();
}
