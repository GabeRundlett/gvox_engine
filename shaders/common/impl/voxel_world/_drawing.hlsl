#pragma once

#include "common/interface/voxel_world.hlsl"

#include "common/impl/voxel_world/_common.hlsl"
#include "common/impl/voxel_chunk/_drawing.hlsl"

float vertex_ao(float2 side, float corner) {
    // if (side.x == 1.0 && side.y == 1.0) return 1.0;
    return (side.x + side.y + max(corner, side.x * side.y)) / 3.0;
}

float4 voxel_ao(float4 side, float4 corner) {
    float4 ao;
    ao.x = vertex_ao(side.xy, corner.x);
    ao.y = vertex_ao(side.yz, corner.y);
    ao.z = vertex_ao(side.zw, corner.z);
    ao.w = vertex_ao(side.wx, corner.w);
    return 1.0 - ao;
}

void VoxelWorld::eval_color(in out GameTraceState trace_state) {
    int temp_chunk_index;

    float3 sample_pos = trace_state.draw_sample.pos - trace_state.draw_sample.nrm * 0.01;
    uint index = calc_index(voxel_chunks[trace_state.shape_i].calc_tile_offset(sample_pos));
    Voxel voxel = voxel_chunks[trace_state.shape_i].sample_voxel(index);
    trace_state.draw_sample.col = voxel.get_col();
    // trace_state.draw_sample.col = sample_lod(sample_pos, temp_chunk_index) * 0.1;
    // trace_state.draw_sample.col = trace_state.shape_i * 0.01;
    // trace_state.draw_sample.col = trace_state.draw_sample.nrm;

    // FractalNoiseConfig noise_conf = {
    //     /* .amplitude   = */ 1.0,
    //     /* .persistance = */ 0.5,
    //     /* .scale       = */ 0.1,
    //     /* .lacunarity  = */ 2.0,
    //     /* .octaves     = */ 6,
    // };
    // float val = fractal_voronoi_noise(trace_state.draw_sample.pos, noise_conf);

    // float val = voronoi_noise(trace_state.draw_sample.pos);
    // val = pow(val, 2);

    // trace_state.draw_sample.col = val;

    if (box.inside(trace_state.draw_sample.pos + trace_state.draw_sample.nrm * 0.01)) {
        float3 mask = abs(trace_state.draw_sample.nrm);
        float3 v_pos = trace_state.draw_sample.pos * VOXEL_SCL - trace_state.draw_sample.nrm * 0.01;
        float3 b_pos = floor(v_pos + trace_state.draw_sample.nrm * 0.1) / VOXEL_SCL;
        float3 d1 = mask.zxy / VOXEL_SCL;
        float3 d2 = mask.yzx / VOXEL_SCL;
        float4 side = float4(
            sample_lod(b_pos + d1, temp_chunk_index) == 0,
            sample_lod(b_pos + d2, temp_chunk_index) == 0,
            sample_lod(b_pos - d1, temp_chunk_index) == 0,
            sample_lod(b_pos - d2, temp_chunk_index) == 0);
        float4 corner = float4(
            sample_lod(b_pos + d1 + d2, temp_chunk_index) == 0,
            sample_lod(b_pos - d1 + d2, temp_chunk_index) == 0,
            sample_lod(b_pos - d1 - d2, temp_chunk_index) == 0,
            sample_lod(b_pos + d1 - d2, temp_chunk_index) == 0);
        float2 uv = fmod(float2(dot(mask * v_pos.yzx, float3(1, 1, 1)), dot(mask * v_pos.zxy, float3(1, 1, 1))), float2(1, 1));
        float4 ao = voxel_ao(side, corner);
        float interp_ao = lerp(lerp(ao.z, ao.w, uv.x), lerp(ao.y, ao.x, uv.x), uv.y);
        trace_state.draw_sample.col *= 0.5 + interp_ao * 0.5;
    }

    // int temp_chunk_index;
    // trace_state.draw_sample.col = 0.1 * sample_lod(trace_state.draw_sample.pos, temp_chunk_index);

    trace_state.draw_sample.nrm = voxel.get_nrm();
    // trace_state.draw_sample.nrm = normalize(voxel.get_nrm() + trace_state.draw_sample.nrm * 0.1);
}
