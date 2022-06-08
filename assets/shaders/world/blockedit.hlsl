struct Push {
    float4 pos;
    uint globals_sb;
    uint output_image_i;
    uint edit_mode;
};

[[vk::push_constant]] const Push p;

#include "core.hlsl"
#include "drawing/defines.hlsl"
#include "utils/shape_dist.hlsl"

[numthreads(8, 8, 8)] void main(uint3 global_i
                                : SV_DispatchThreadID) {
    StructuredBuffer<Globals> globals = daxa::getBuffer<Globals>(p.globals_sb);
    float3 pick_pos;
    if (p.edit_mode == 0)
        pick_pos = globals[0].pick_pos[0].xyz;
    else
        pick_pos = globals[0].pick_pos[1].xyz;
    if (globals[0].inventory_index == 7)
        pick_pos += float3(0, -27, 0);
    int3 pick_block_i = int3(pick_pos);
    int3 pick_chunk_i = pick_block_i / CHUNK_SIZE;
    int xi = pick_chunk_i.x + p.pos.x;
    int yi = pick_chunk_i.y + p.pos.y;
    int zi = pick_chunk_i.z + p.pos.z;
    if (xi >= 0 && xi < CHUNK_NX && yi >= 0 && yi < CHUNK_NY && zi >= 0 && zi < CHUNK_NZ) {
        uint chunk_id = globals[0].chunk_images[zi][yi][xi];
        float3 p_pos = float3(1.0f * xi * CHUNK_SIZE, 1.0f * yi * CHUNK_SIZE, 1.0f * zi * CHUNK_SIZE);
        float3 block_pos = float3(global_i) + p_pos;
        if (chunk_id == globals[0].empty_chunk_index)
            return;
        RWTexture3D<uint> chunk = daxa::getRWTexture3D<uint>(chunk_id);

        if (globals[0].inventory_index == 6) {
            TreeSDF tree_sdf = sd_tree_custom(block_pos, 0, pick_block_i);
            if (tree_sdf.trunk_dist < 0)
                chunk[int3(global_i)] = p.edit_mode == 0 ? (uint)BlockID::Air : (uint)BlockID::Log;
            if (tree_sdf.leaves_dist < 0)
                chunk[int3(global_i)] = p.edit_mode == 0 ? (uint)BlockID::Air : (uint)BlockID::Leaves;
        } else if (globals[0].inventory_index == 7) {
            StructuredBuffer<ModelLoadBuffer> model = daxa::getBuffer<ModelLoadBuffer>(globals[0].model_load_index);
            float3 m_dim = model[0].dim.xyz;
            float3 m_pos = pick_block_i - m_dim / 2;
            if (block_pos.x >= m_pos.x && block_pos.x < m_pos.x + m_dim.x &&
                block_pos.y >= m_pos.y && block_pos.y < m_pos.y + m_dim.y &&
                block_pos.z >= m_pos.z && block_pos.z < m_pos.z + m_dim.z) {
                int3 model_tile_i = int3(block_pos - m_pos);
                uint model_tile_index = model_tile_i.x + model_tile_i.z * 128 + (63 - model_tile_i.y) * 128 * 128;
                BlockID model_tile = (BlockID)model[0].data[model_tile_index];
                if (model_tile != BlockID::Air)
                    chunk[int3(global_i)] = p.edit_mode == 0 ? (uint)BlockID::Air : (uint)BlockID::Cobblestone;
            }
        } else {
            if (length(int3(block_pos) - pick_block_i) <= 0.5 + BLOCKEDIT_RADIUS) {
                chunk[int3(global_i)] = p.edit_mode == 0 ? 1 : (uint)inventory_palette(globals[0].inventory_index);
            }
        }
    }
}
