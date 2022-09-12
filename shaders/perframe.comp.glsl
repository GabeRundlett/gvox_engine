#version 450

#include <shared.inl>

DAXA_USE_PUSH_CONSTANT(PerframeCompPush)

void perframe_player(inout Player player) {
    const f32 mouse_sens = 1.0;

    player.rot.z += push_constant.gpu_input.mouse.pos_delta.x * mouse_sens * 0.001;
    player.rot.x -= push_constant.gpu_input.mouse.pos_delta.y * mouse_sens * 0.001;

    const float MAX_ROT = 1.57;
    if (player.rot.x > MAX_ROT)
        player.rot.x = MAX_ROT;
    if (player.rot.x < -MAX_ROT)
        player.rot.x = -MAX_ROT;
    float sin_rot_x = sin(player.rot.x), cos_rot_x = cos(player.rot.x);
    float sin_rot_y = sin(player.rot.y), cos_rot_y = cos(player.rot.y);
    float sin_rot_z = sin(player.rot.z), cos_rot_z = cos(player.rot.z);

    // clang-format off
    player.cam.rot_mat = 
        f32mat3x3(
            cos_rot_z, -sin_rot_z, 0,
            sin_rot_z,  cos_rot_z, 0,
            0,          0,         1
        ) *
        f32mat3x3(
            1,          0,          0,
            0,  cos_rot_x,  sin_rot_x,
            0, -sin_rot_x,  cos_rot_x
        );
    // clang-format on

    f32vec3 move_vec = f32vec3(0, 0, 0);
    f32vec3 forward = f32vec3(+sin_rot_z, +cos_rot_z, 0);
    f32vec3 lateral = f32vec3(+cos_rot_z, -sin_rot_z, 0);

    if (push_constant.gpu_input.keyboard.keys[GAME_KEY_W] != 0)
        move_vec += forward;
    if (push_constant.gpu_input.keyboard.keys[GAME_KEY_S] != 0)
        move_vec -= forward;
    if (push_constant.gpu_input.keyboard.keys[GAME_KEY_A] != 0)
        move_vec -= lateral;
    if (push_constant.gpu_input.keyboard.keys[GAME_KEY_D] != 0)
        move_vec += lateral;

    if (push_constant.gpu_input.keyboard.keys[GAME_KEY_SPACE] != 0)
        move_vec += f32vec3(0, 0, 1);
    if (push_constant.gpu_input.keyboard.keys[GAME_KEY_LEFT_CONTROL] != 0)
        move_vec -= f32vec3(0, 0, 1);

    player.pos += move_vec * push_constant.gpu_input.delta_time * 10;

    player.cam.pos = player.pos;
    player.cam.tan_half_fov = tan(push_constant.gpu_input.fov * 3.14159 / 360.0);
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    perframe_player(push_constant.gpu_globals.player);
}
