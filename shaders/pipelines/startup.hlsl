#include "common/buffers.hlsl"

#include "common/impl/game/_update.hlsl"

struct Push {
    daxa::BufferId globals_id;
};
[[vk::push_constant]] const Push p;

[numthreads(1, 1, 1)] void main() {
    StructuredBuffer<Globals> globals = daxa::get_StructuredBuffer<Globals>(p.globals_id);
    globals[0].game.default_init();

    globals[0].game.init();
}
