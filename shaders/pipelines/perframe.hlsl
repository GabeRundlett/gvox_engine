#include "common/buffers.hlsl"
#include "common/impl/game/_update.hlsl"

struct Push {
    uint globals_id;
    uint input_id;
};
[[vk::push_constant]] const Push p;

[numthreads(1, 1, 1)] void main() {
    StructuredBuffer<Globals> globals = daxa::getBuffer<Globals>(p.globals_id);
    StructuredBuffer<Input> input = daxa::getBuffer<Input>(p.input_id);

    globals[0].game.update(input[0]);
}
