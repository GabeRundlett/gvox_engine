#include <shared/shared.inl>

DAXA_USE_PUSH_CONSTANT(ChunkEditCompPush)

#define CHUNKGEN 1

#include <brush_input.glsl>
#include <utils/impl_brush_header.glsl>
#include <brush_kernel.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    u32 chunk_index = VOXEL_WORLD.chunkgen_index;
    if (VOXEL_WORLD.chunks_genstate[chunk_index].edit_stage != 1)
        return;
    u32vec3 voxel_i = gl_GlobalInvocationID.xyz;
    u32 voxel_index = voxel_i.x + voxel_i.y * CHUNK_SIZE + voxel_i.z * CHUNK_SIZE * CHUNK_SIZE;
    f32vec3 voxel_p = f32vec3(voxel_i) / VOXEL_SCL + VOXEL_WORLD.voxel_chunks[chunk_index].box.bound_min;

    Voxel air_voxel;
    air_voxel.block_id = BlockID_Air;
    air_voxel.col = block_color(BlockID_Air);

    BrushInput brush;
    brush.origin = floor((GLOBALS.brush_origin - GLOBALS.brush_offset) * VOXEL_SCL) / VOXEL_SCL;
    brush.p = voxel_p;
    brush.begin_p = GLOBALS.edit_origin - brush.origin;
    brush.prev_voxel = air_voxel;

    Voxel result;
    result.block_id = BlockID_Debug;
    custom_brush_kernel(brush, result);

    if (result.block_id == BlockID_Debug) {
        result = air_voxel;
    }

    VOXEL_WORLD.voxel_chunks[chunk_index].packed_voxels[voxel_index] = pack_voxel(result);
}
