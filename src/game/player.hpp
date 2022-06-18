#pragma once

#include <glm/glm.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <numbers>

struct Camera {
    float fov = 98.6f;
};

namespace input {
    struct Keybinds {
        i32 move_pz, move_nz;
        i32 move_px, move_nx;
        i32 move_py, move_ny;
        i32 toggle_pause;
        i32 toggle_sprint;
    };

    static constexpr Keybinds DEFAULT_KEYBINDS{
        .move_pz = GLFW_KEY_W,
        .move_nz = GLFW_KEY_S,
        .move_px = GLFW_KEY_A,
        .move_nx = GLFW_KEY_D,
        .move_py = GLFW_KEY_SPACE,
        .move_ny = GLFW_KEY_LEFT_CONTROL,
        .toggle_pause = GLFW_KEY_ESCAPE,
        .toggle_sprint = GLFW_KEY_LEFT_SHIFT,
    };
} // namespace input

struct Player {
    Camera camera;
    input::Keybinds keybinds;
    float speed = 30.0f, mouse_sens = 0.1f;
    float sprint_speed = 8.0f;

    struct MoveFlags {
        uint8_t px : 1, py : 1, pz : 1, nx : 1, ny : 1, nz : 1, sprint : 1;
    } move{};

    void on_key(int key, int action) {
        if (key == keybinds.move_pz)
            move.pz = action != 0;
        if (key == keybinds.move_nz)
            move.nz = action != 0;
        if (key == keybinds.move_px)
            move.px = action != 0;
        if (key == keybinds.move_nx)
            move.nx = action != 0;
        if (key == keybinds.move_py)
            move.py = action != 0;
        if (key == keybinds.move_ny)
            move.ny = action != 0;
        if (key == keybinds.toggle_sprint)
            move.sprint = action != 0;
    }
};
