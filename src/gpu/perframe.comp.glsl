#include <utils/player.glsl>
#include <utils/voxel_world.glsl>
#include <utils/trace.glsl>

DAXA_USE_PUSH_CONSTANT(PerframeComputePush)

#define SETTINGS deref(daxa_push_constant.gpu_settings)
#define INPUT deref(daxa_push_constant.gpu_input)
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    player_perframe(
        daxa_push_constant.gpu_settings,
        daxa_push_constant.gpu_input,
        daxa_push_constant.gpu_globals);
    voxel_world_perframe(
        daxa_push_constant.gpu_settings,
        daxa_push_constant.gpu_input,
        daxa_push_constant.gpu_globals);

    {
        f32vec2 frame_dim = INPUT.frame_dim;
        f32vec2 inv_frame_dim = f32vec2(1.0) / frame_dim;
        f32 aspect = frame_dim.x * inv_frame_dim.y;
        f32vec2 uv = (deref(daxa_push_constant.gpu_input).mouse.pos * inv_frame_dim - 0.5) * f32vec2(2 * aspect, 2);
        f32vec3 ray_pos = create_view_pos(deref(daxa_push_constant.gpu_globals).player);
        f32vec3 ray_dir = create_view_dir(deref(daxa_push_constant.gpu_globals).player, uv);
        u32vec3 chunk_n = u32vec3(1u << SETTINGS.log2_chunks_per_axis);
        trace(daxa_BufferPtr(daxa_u32)(daxa_push_constant.gpu_allocator.heap), daxa_push_constant.voxel_chunks, chunk_n, ray_pos, ray_dir);
        if (dot(ray_pos, ray_pos) < MAX_SD * MAX_SD / 2) {
            deref(daxa_push_constant.gpu_globals).pick_pos = ray_pos;
        }
    }

    deref(daxa_push_constant.gpu_output).heap_size = deref(daxa_push_constant.gpu_allocator.state).offset;
    deref(daxa_push_constant.gpu_output).player_pos = deref(daxa_push_constant.gpu_globals).player.pos;

    if (INPUT.actions[GAME_ACTION_INTERACT0] != 0) {
        deref(daxa_push_constant.gpu_allocator.state).offset = 0;
    }
}
#undef INPUT
#undef SETTINGS
