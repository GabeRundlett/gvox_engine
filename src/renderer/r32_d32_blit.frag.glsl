#include <renderer/trace_primary.inl>

DAXA_DECL_PUSH_CONSTANT(R32D32BlitPush, push)
daxa_ImageViewIndex input_tex = push.uses.input_tex;

void main() {
    gl_FragDepth = texelFetch(daxa_texture2D(input_tex), ivec2(gl_FragCoord.xy), 0).r;
}
