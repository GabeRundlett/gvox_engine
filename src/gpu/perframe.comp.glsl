#include <utils/player.glsl>
#include <utils/voxel_world.glsl>
#include <utils/voxel_malloc.glsl>
#include <utils/trace.glsl>

DAXA_USE_PUSH_CONSTANT(PerframeComputePush)

#define SETTINGS deref(daxa_push_constant.gpu_settings)
#define INPUT deref(daxa_push_constant.gpu_input)
#define BRUSH_STATE deref(daxa_push_constant.gpu_globals).brush_state
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
        f32vec3 cam_pos = ray_pos;
        f32vec3 ray_dir = create_view_dir(deref(daxa_push_constant.gpu_globals).player, uv);
        u32vec3 chunk_n = u32vec3(1u << SETTINGS.log2_chunks_per_axis);
        trace(daxa_push_constant.voxel_malloc_global_allocator, daxa_push_constant.voxel_chunks, chunk_n, ray_pos, ray_dir);

        if (BRUSH_STATE.is_editing == 0) {
            BRUSH_STATE.initial_ray = ray_pos - cam_pos;
        }

        BRUSH_STATE.prev_pos = BRUSH_STATE.pos;
        BRUSH_STATE.pos = length(BRUSH_STATE.initial_ray) * ray_dir + cam_pos;

        if (INPUT.actions[GAME_ACTION_BRUSH_A] != 0 || INPUT.actions[GAME_ACTION_BRUSH_B] != 0) {
            BRUSH_STATE.is_editing = 1;
        } else {
            BRUSH_STATE.is_editing = 0;
        }
    }

    deref(daxa_push_constant.gpu_output).player_pos = deref(daxa_push_constant.gpu_globals).player.pos;

#if USE_OLD_ALLOC
    deref(daxa_push_constant.gpu_output).heap_size = deref(daxa_push_constant.voxel_malloc_global_allocator).offset;
#else
    deref(daxa_push_constant.gpu_output).heap_size =
        (deref(daxa_push_constant.voxel_malloc_global_allocator).page_count -
         deref(daxa_push_constant.voxel_malloc_global_allocator).available_pages_stack_size) *
        VOXEL_MALLOC_PAGE_SIZE_U32S;

    // Debug - reset the allocator
    // deref(daxa_push_constant.voxel_malloc_global_allocator).page_count = 0;
    // deref(daxa_push_constant.voxel_malloc_global_allocator).available_pages_stack_size = 0;
    // deref(daxa_push_constant.voxel_malloc_global_allocator).released_pages_stack_size = 0;
#endif

    voxel_malloc_perframe(
        daxa_push_constant.gpu_input,
        daxa_push_constant.voxel_malloc_global_allocator);
}
#undef INPUT
#undef SETTINGS
