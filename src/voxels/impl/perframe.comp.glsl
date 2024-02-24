#include "voxel_world.inl"

DAXA_DECL_PUSH_CONSTANT(VoxelWorldPerframeComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuOutput) gpu_output = push.uses.gpu_output;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_RWBufferPtr)

#include <renderer/kajiya/inc/camera.glsl>
#include <voxels/voxels.glsl>
#include <voxels/voxel_particle.glsl>

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    VoxelRWBufferPtrs ptrs = VOXELS_RW_BUFFER_PTRS;

    for (uint i = 0; i < MAX_CHUNK_UPDATES_PER_FRAME; ++i) {
        deref(ptrs.globals).chunk_update_infos[i].brush_flags = 0;
        deref(ptrs.globals).chunk_update_infos[i].i = INVALID_CHUNK_I;
    }

    deref(ptrs.globals).chunk_update_n = 0;

    deref(ptrs.globals).prev_offset = deref(ptrs.globals).offset;
    deref(ptrs.globals).offset = deref(gpu_input).player.player_unit_offset;

    deref(globals).indirect_dispatch.chunk_edit_dispatch = uvec3(CHUNK_SIZE / 8, CHUNK_SIZE / 8, 0);
    deref(globals).indirect_dispatch.subchunk_x2x4_dispatch = uvec3(1, 64, 0);
    deref(globals).indirect_dispatch.subchunk_x8up_dispatch = uvec3(1, 1, 0);

    VoxelMallocPageAllocator_perframe(ptrs.allocator);
    // VoxelLeafChunkAllocator_perframe(ptrs.voxel_leaf_chunk_allocator);
    // VoxelParentChunkAllocator_perframe(ptrs.voxel_parent_chunk_allocator);

    deref(advance(gpu_output, deref(gpu_input).fif_index)).voxel_world.voxel_malloc_output.current_element_count =
        VoxelMallocPageAllocator_get_consumed_element_count(daxa_BufferPtr(VoxelMallocPageAllocator)(as_address(ptrs.allocator)));
    // deref(advance(gpu_output, deref(gpu_input).fif_index)).voxel_world.voxel_leaf_chunk_output.current_element_count =
    //     VoxelLeafChunkAllocator_get_consumed_element_count(voxel_leaf_chunk_allocator);
    // deref(advance(gpu_output, deref(gpu_input).fif_index)).voxel_world.voxel_parent_chunk_output.current_element_count =
    //     VoxelParentChunkAllocator_get_consumed_element_count(voxel_parent_chunk_allocator);

    // Brush stuff
    vec2 frame_dim = deref(gpu_input).frame_dim;
    vec2 inv_frame_dim = vec2(1.0) / frame_dim;
    vec2 uv = get_uv(deref(gpu_input).mouse.pos, vec4(frame_dim, inv_frame_dim));
    ViewRayContext vrc = unjittered_vrc_from_uv(gpu_input, uv);
    vec3 ray_dir = ray_dir_ws(vrc);
    vec3 cam_pos = ray_origin_ws(vrc);
    vec3 ray_pos = cam_pos;
    voxel_trace(VoxelTraceInfo(VOXELS_BUFFER_PTRS, ray_dir, MAX_STEPS, MAX_DIST, 0.0, true), ray_pos);

    if (deref(globals).brush_state.is_editing == 0) {
        deref(globals).brush_state.initial_ray = ray_pos - cam_pos;
    }

    deref(globals).brush_input.prev_pos = deref(globals).brush_input.pos;
    deref(globals).brush_input.prev_pos_offset = deref(globals).brush_input.pos_offset;
    deref(globals).brush_input.pos = length(deref(globals).brush_state.initial_ray) * ray_dir + cam_pos;
    deref(globals).brush_input.pos_offset = deref(gpu_input).player.player_unit_offset;

    if (deref(gpu_input).actions[GAME_ACTION_BRUSH_A] != 0) {
        deref(globals).brush_state.is_editing = 1;
    } else if (deref(gpu_input).actions[GAME_ACTION_BRUSH_B] != 0) {
        if (deref(globals).brush_state.is_editing == 0) {
            deref(globals).brush_state.initial_frame = deref(gpu_input).frame_index;
        }
        deref(globals).brush_state.is_editing = 1;
    } else {
        deref(globals).brush_state.is_editing = 0;
    }
}
