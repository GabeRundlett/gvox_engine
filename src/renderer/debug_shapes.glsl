#include <debug_shapes.inl>

#if defined(DebugLinesShader)
DAXA_DECL_PUSH_CONSTANT(DebugLinesPush, push)

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

layout(location = 0) out vec3 f_col;

void main() {
    mat4 world_to_clip;

    // if (push.flags != 0) {
    //     world_to_clip = deref(push.uses.gpu_input).observer_cam.view_to_clip * deref(push.uses.gpu_input).observer_cam.world_to_view;
    // } else {
        world_to_clip = deref(push.uses.gpu_input).player.cam.view_to_clip * deref(push.uses.gpu_input).player.cam.world_to_view;
    // }

    uint shape_index = gl_VertexIndex / 2;
    uint vertex_index = gl_VertexIndex % 2;

    vec3 vertex_pos = deref(advance(push.vertex_data, shape_index * 3 + vertex_index));
    vec3 vertex_col = deref(advance(push.vertex_data, shape_index * 3 + 2));

    gl_Position = world_to_clip * vec4(vertex_pos, 1);
    gl_Position.y *= -1;
    f_col = vertex_col;
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT

layout(location = 0) in vec3 f_col;
layout(location = 0) out vec4 f_out;

void main() {
    f_out = vec4(f_col, 1);
}

#endif
#endif

#if defined(DebugPointsShader)
DAXA_DECL_PUSH_CONSTANT(DebugPointsPush, push)

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

layout(location = 0) out vec3 f_col;
layout(location = 1) out vec2 f_uv;

void main() {
    uint shape_index = gl_VertexIndex / 6;
    uint vertex_index = gl_VertexIndex % 6;

    vec3 vertex_pos = deref(advance(push.vertex_data, shape_index * 3 + 0));
    vec3 vertex_col = deref(advance(push.vertex_data, shape_index * 3 + 1));
    vec3 vertex_meta = deref(advance(push.vertex_data, shape_index * 3 + 2));

    vec2 uv = vec2((0x32 >> vertex_index) & 1, (0x2c >> vertex_index) & 1);
    f_uv = uv;
    vec4 pos_cs;

    vec2 pos_offset = uv - 0.5;

    mat4 world_to_view;
    mat4 view_to_clip;

    // if (push.flags != 0) {
    //     world_to_view = deref(push.uses.gpu_input).observer_cam.world_to_view;
    //     view_to_clip = deref(push.uses.gpu_input).observer_cam.view_to_clip;
    // } else {
        world_to_view = deref(push.uses.gpu_input).player.cam.world_to_view;
        view_to_clip = deref(push.uses.gpu_input).player.cam.view_to_clip;
    // }

    switch (int(vertex_meta.z)) {
    case 0: {
        vec4 pos_vs = world_to_view * vec4(vertex_pos, 1);
        pos_cs = view_to_clip * pos_vs;
        pos_cs = pos_cs / pos_cs.w;
        pos_cs.xy += pos_offset * vertex_meta.xy * 2.0 / deref(push.uses.gpu_input).frame_dim;
    } break;
    case 1: {
        vec4 pos_vs = world_to_view * vec4(vertex_pos, 1);
        pos_vs.xy += pos_offset * vertex_meta.xy;
        pos_cs = view_to_clip * pos_vs;
    } break;
    }

    gl_Position = pos_cs;
    f_col = vertex_col;
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT

layout(location = 0) in vec3 f_col;
layout(location = 1) in vec2 f_uv;
layout(location = 0) out vec4 f_out;

void main() {
    if (length(f_uv - 0.5) > 0.5) {
        discard;
    }
    f_out = vec4(f_col, 1);
}

#endif
#endif
