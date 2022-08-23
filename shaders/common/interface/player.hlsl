#pragma once

#include "common/input.hlsl"
#include "common/interface/camera.hlsl"

#define KEYBINDS_MOVE_R GAME_KEY_D
#define KEYBINDS_MOVE_L GAME_KEY_A
#define KEYBINDS_MOVE_D GAME_KEY_LEFT_CONTROL
#define KEYBINDS_MOVE_U GAME_KEY_SPACE
#define KEYBINDS_MOVE_B GAME_KEY_S
#define KEYBINDS_MOVE_F GAME_KEY_W
#define KEYBINDS_SPRINT GAME_KEY_LEFT_SHIFT
#define KEYBINDS_NOCLIP GAME_KEY_F

#define PLAYER_HEIGHT 1.80
#define PLAYER_EYE_OFFSET 0.112
#define PLAYER_EYE_HEIGHT (PLAYER_HEIGHT - PLAYER_EYE_OFFSET)

#define PLAYER_SPEED 1.5
#define PLAYER_ACCEL 10.0

struct Player {
    Camera camera, prev_camera;
    float3 pos, rot;
    float3 force_vel, move_vel;
    float mouse_sens, accel_rate, speed, sprint_speed;
    float bobbing_phase, max_speed;
    uint move_flags;
    uint prev_key_flags, key_flags;
    uint init_flags;
    uint on_ground;
    uint view_state;

    void default_init() {
        camera.default_init();
        prev_camera.default_init();

        mouse_sens = 0;
        speed = 0;
        sprint_speed = 0;
        bobbing_phase = 0;
        // move_flags = 0;
        on_ground = 0;
        accel_rate = 0;
    }

    void init();
    void calc_update(in out Input input);
    void apply_update(float3 v, in out Input input);

    void toggle_view();
    float3 view_vec();

    bool is_moving() { return move_flags != 0; }
    bool should_sprint() { return move_flags & (1u << 0); }
    bool should_move_l() { return move_flags & (1u << 1); }
    bool should_move_r() { return move_flags & (1u << 2); }
    bool should_move_f() { return move_flags & (1u << 3); }
    bool should_move_b() { return move_flags & (1u << 4); }
    bool should_move_u() { return move_flags & (1u << 5); }
    bool should_move_d() { return move_flags & (1u << 6); }
    bool should_noclip() { return move_flags & (1u << 7); }

    void set_flag_bit(in out uint flag, uint index, bool val) {
        uint mask = 0x1u << index;
        flag &= ~mask;
        flag |= mask * val;
    }

    void update_keys(in out Input input) {
        set_flag_bit(move_flags, 0, input.keyboard.keys[KEYBINDS_SPRINT] != 0);
        set_flag_bit(move_flags, 1, input.keyboard.keys[KEYBINDS_MOVE_L] != 0);
        set_flag_bit(move_flags, 2, input.keyboard.keys[KEYBINDS_MOVE_R] != 0);
        set_flag_bit(move_flags, 3, input.keyboard.keys[KEYBINDS_MOVE_F] != 0);
        set_flag_bit(move_flags, 4, input.keyboard.keys[KEYBINDS_MOVE_B] != 0);
        set_flag_bit(move_flags, 5, input.keyboard.keys[KEYBINDS_MOVE_U] != 0);
        set_flag_bit(move_flags, 6, input.keyboard.keys[KEYBINDS_MOVE_D] != 0);

        prev_key_flags = key_flags;
        set_flag_bit(key_flags, 0, input.keyboard.keys[KEYBINDS_SPRINT] != 0);
        set_flag_bit(key_flags, 1, input.keyboard.keys[KEYBINDS_MOVE_L] != 0);
        set_flag_bit(key_flags, 2, input.keyboard.keys[KEYBINDS_MOVE_R] != 0);
        set_flag_bit(key_flags, 3, input.keyboard.keys[KEYBINDS_MOVE_F] != 0);
        set_flag_bit(key_flags, 4, input.keyboard.keys[KEYBINDS_MOVE_B] != 0);
        set_flag_bit(key_flags, 5, input.keyboard.keys[KEYBINDS_MOVE_U] != 0);
        set_flag_bit(key_flags, 6, input.keyboard.keys[KEYBINDS_MOVE_D] != 0);
        set_flag_bit(key_flags, 7, input.keyboard.keys[KEYBINDS_NOCLIP] != 0);

        uint k0 = (prev_key_flags >> 7) & 0x1;
        uint k1 = (key_flags >> 7) & 0x1;
        uint m = (move_flags >> 7) & 0x1;
        if (k0 != k1 && k1) {
            set_flag_bit(move_flags, 7, !m);
        }
    }
};
