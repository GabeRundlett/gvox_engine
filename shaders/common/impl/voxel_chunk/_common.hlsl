#pragma once

float3 Voxel::get_col() {
    return uint_to_float4(col_id).rgb;
}
float3 Voxel::get_nrm() {
    return uint_to_float4(nrm).rgb;
}

int3 VoxelChunk::calc_tile_offset(float3 p) {
    return int3((p - box.bound_min) * VOXEL_SCL);
}
uint calc_index(int3 tile_offset) {
    tile_offset = clamp(tile_offset, int3(0, 0, 0), int3(CHUNK_SIZE - 1, CHUNK_SIZE - 1, CHUNK_SIZE - 1));
    return tile_offset.x + tile_offset.y * CHUNK_SIZE + tile_offset.z * CHUNK_SIZE * CHUNK_SIZE;
}
uint VoxelChunk::sample_tile(float3 p) {
    return sample_tile(calc_tile_offset(p));
}
uint VoxelChunk::sample_tile(int3 tile_offset) {
    return sample_tile(calc_index(tile_offset));
}
uint VoxelChunk::sample_tile(uint index) {
    return sample_voxel(index).col_id;
}
Voxel VoxelChunk::sample_voxel(uint index) {
    return data[index];
}
float3 VoxelChunk::sample_color(float3 p) {
    // return sample_tile(p) * 0.01;
    return uint_to_float4(sample_tile(p)).rgb;
}
