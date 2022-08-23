#include "common/buffers.hlsl"
#include "common/impl/game/_update.hlsl"

struct Push {
    daxa::BufferId globals_id;
    daxa::BufferId input_id;
};
[[vk::push_constant]] const Push p;

[numthreads(1, 1, 1)] void main() {
    StructuredBuffer<Globals> globals = daxa::get_StructuredBuffer<Globals>(p.globals_id);
    StructuredBuffer<Input> input = daxa::get_StructuredBuffer<Input>(p.input_id);

    globals[0].game.update(input[0]);
}
