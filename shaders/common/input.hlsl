#pragma once

#include "common/glfw_keycodes.hlsl"

struct MouseInput {
    float2 pos;
    float2 pos_delta;
    float2 scroll_delta;
    uint buttons[GLFW_MOUSE_BUTTON_LAST + 1];
};

struct KeyboardInput {
    uint keys[GLFW_KEY_LAST + 1];
};

struct Input {
    int2 frame_dim;
    float time, delta_time;
    float fov;
    float3 block_color;
    uint flags;
    uint _pad0[3];
    MouseInput mouse;
    KeyboardInput keyboard;

    bool shadows_enabled() {
        return (flags >> 0) & 0x1;
    }

    uint max_steps() {
        return _pad0[0];
    }
};
