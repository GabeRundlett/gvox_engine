#pragma once

#include "common/interface/voxel_world.hlsl"
#include "common/impl/voxel_chunk.hlsl"

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

uint VoxelWorld::get_chunk_index(int3 chunk_i) {
    return chunk_i.x + chunk_i.y * CHUNK_NX + chunk_i.z * CHUNK_NX * CHUNK_NY;
}

void VoxelWorld::init() {
    for (int zi = 0; zi < CHUNK_NZ; ++zi) {
        for (int yi = 0; yi < CHUNK_NY; ++yi) {
            for (int xi = 0; xi < CHUNK_NX; ++xi) {
                int index = get_chunk_index(int3(xi, yi, zi));
                voxel_chunks[index].box.bound_min = float3(xi, yi, zi) * (CHUNK_SIZE / VOXEL_SCL);
                voxel_chunks[index].box.bound_max = voxel_chunks[index].box.bound_min + (CHUNK_SIZE / VOXEL_SCL);
            }
        }
    }
    box.bound_min = voxel_chunks[0].box.bound_min;
    box.bound_max = voxel_chunks[CHUNK_N - 1].box.bound_max;

    float min_dist = 1e38;
    for (uint zi = 0; zi < CHUNK_NZ / 1; ++zi) {
        for (uint yi = 0; yi < CHUNK_NY / 1; ++yi) {
            for (uint xi = 0; xi < CHUNK_NX / 1; ++xi) {
                uint i = get_chunk_index(int3(xi, yi, zi));
                float3 chunk_p = (float3(xi, yi, zi) * 1 + 2) * CHUNK_SIZE / VOXEL_SCL + box.bound_min;
                float chunk_dist = length(chunk_p - center_pt);
                if (chunk_dist < min_dist) {
                    min_dist = chunk_dist;
                    chunkgen_i = int3(xi, yi, zi);
                }
            }
        }
    }
}

void VoxelWorld::chunkgen(int3 block_offset, in out Input input) {
    int3 chunk_i = clamp(chunkgen_i, int3(0, 0, 0), int3(CHUNK_NX - 1, CHUNK_NY - 1, CHUNK_NZ - 1));
    int index = clamp(get_chunk_index(chunk_i), 0, CHUNK_N - 1);
    switch (chunks_genstate[index].edit_stage) {
    case EditStage::ProceduralGen: voxel_chunks[index].gen(block_offset); break;
    case EditStage::BlockEdit: voxel_chunks[index].do_edit(block_offset, edit_info); break;
    default: break;
    }
}

void VoxelWorld::queue_edit() {
    int3 chunk_i = clamp(int3(edit_info.pos * VOXEL_SCL / CHUNK_SIZE), int3(0, 0, 0), int3(CHUNK_NX - 1, CHUNK_NY - 1, CHUNK_NZ - 1));
    int index = clamp(get_chunk_index(chunk_i), 0, CHUNK_N - 1);
    chunks_genstate[index].edit_stage = EditStage::BlockEdit;
}

void VoxelWorld::update(in out Input input) {
    uint prev_i = get_chunk_index(chunkgen_i);
    if (chunkgen_i.x != -1000)
        chunks_genstate[prev_i].edit_stage = EditStage::Finished;

    bool finished = true;
    float min_dist = 1e38;
    for (uint zi = 0; zi < CHUNK_NZ / 1; ++zi) {
        for (uint yi = 0; yi < CHUNK_NY / 1; ++yi) {
            for (uint xi = 0; xi < CHUNK_NX / 1; ++xi) {
                uint i = get_chunk_index(int3(xi, yi, zi));
                Box chunk_box = voxel_chunks[i].box;
                if (chunks_genstate[i].edit_stage != EditStage::Finished) {
                    finished = false;
                    float3 chunk_p = (chunk_box.bound_min + chunk_box.bound_max) * 0.5;
                    float chunk_dist = length(chunk_p - center_pt);
                    if (chunk_dist < min_dist) {
                        min_dist = chunk_dist;
                        chunkgen_i = int3(xi, yi, zi);
                    }
                    switch (chunks_genstate[i].edit_stage) {
                    case EditStage::None: chunks_genstate[i].edit_stage = EditStage::ProceduralGen; break;
                    default: break;
                    }
                }
            }
        }
    }
}

void VoxelWorld::trace(in out GameTraceState trace_state, Ray ray) {
    TraceRecord bound_trace;
    bound_trace.default_init();
    if (box.inside(ray.o)) {
        bound_trace.dist = 0;
        bound_trace.hit = true;
    } else {
        bound_trace = trace_box(ray, box);
        bound_trace.dist += 0.0001;
        // int chunk_index = 0;
        // uint lod = sample_lod(ray.o + ray.nrm * bound_trace.dist, chunk_index);
        // trace_state.trace_record = bound_trace;
        // trace_state.shape_i = chunk_index;
        // trace_state.shape_type = 0;
        // return;
    }
    int chunk_index = 0;
    if (bound_trace.hit && bound_trace.dist < trace_state.trace_record.dist) {
        Ray dda_ray = ray;
        dda_ray.o = ray.o + ray.nrm * bound_trace.dist;
        if (!box.inside(dda_ray.o))
            return;
        uint lod = sample_lod(dda_ray.o, chunk_index);
        if (lod != 0) {
            float prev_dist = bound_trace.dist;
            bound_trace = dda(dda_ray, chunk_index, trace_state.max_steps);
            bound_trace.dist += prev_dist;
        }
    }
    if (bound_trace.hit && bound_trace.dist < trace_state.trace_record.dist) {
        trace_state.trace_record = bound_trace;
        trace_state.shape_i = chunk_index;
        trace_state.shape_type = 0;
    }
}

void VoxelWorld::eval_color(in out GameTraceState trace_state) {
    int temp_chunk_index;

    trace_state.draw_sample.col = voxel_chunks[trace_state.shape_i].sample_color(trace_state.draw_sample.pos - trace_state.draw_sample.nrm * 0.01);
    // trace_state.draw_sample.col = sample_lod(trace_state.draw_sample.pos - trace_state.draw_sample.nrm * 0.01, temp_chunk_index) * 0.1;
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
}

uint VoxelWorld::sample_lod(float3 p, in out int index) {
    int3 block_i = clamp(int3((p - box.bound_min) * VOXEL_SCL), int3(0, 0, 0), int3(BLOCK_NX, BLOCK_NY, BLOCK_NZ));
    int3 chunk_i = block_i / CHUNK_SIZE;
    int3 inchunk_block_i = block_i - chunk_i * CHUNK_SIZE;

    index = get_chunk_index(chunk_i);

    uint lod_index_x2 = uniformity_lod_index<2>(inchunk_block_i / 2);
    uint lod_mask_x2 = uniformity_lod_mask(inchunk_block_i / 2);
    uint lod_index_x4 = uniformity_lod_index<4>(inchunk_block_i / 4);
    uint lod_mask_x4 = uniformity_lod_mask(inchunk_block_i / 4);
    uint lod_index_x8 = uniformity_lod_index<8>(inchunk_block_i / 8);
    uint lod_mask_x8 = uniformity_lod_mask(inchunk_block_i / 8);
    uint lod_index_x16 = uniformity_lod_index<16>(inchunk_block_i / 16);
    uint lod_mask_x16 = uniformity_lod_mask(inchunk_block_i / 16);
    uint lod_index_x32 = uniformity_lod_index<32>(inchunk_block_i / 32);
    uint lod_mask_x32 = uniformity_lod_mask(inchunk_block_i / 32);
    uint lod_index_x64 = uniformity_lod_index<64>(inchunk_block_i / 64);
    uint lod_mask_x64 = uniformity_lod_mask(inchunk_block_i / 64);

    EditStage chunk_edit_stage = chunks_genstate[index].edit_stage;
    if (chunk_edit_stage == EditStage::None || chunk_edit_stage == EditStage::ProceduralGen)
        return 7;
    if ((voxel_chunks[index].sample_tile(inchunk_block_i) >> 0x18) != (uint)BlockID::Air)
        return 0;
    if (uniformity_chunks[index].lod_nonuniform<2>(lod_index_x2, lod_mask_x2))
        return 1;
    if (uniformity_chunks[index].lod_nonuniform<4>(lod_index_x4, lod_mask_x4))
        return 2;
    if (uniformity_chunks[index].lod_nonuniform<8>(lod_index_x8, lod_mask_x8))
        return 3;
    if (uniformity_chunks[index].lod_nonuniform<16>(lod_index_x16, lod_mask_x16))
        return 4;
    if (uniformity_chunks[index].lod_nonuniform<32>(lod_index_x32, lod_mask_x32))
        return 5;
    if (uniformity_chunks[index].lod_nonuniform<64>(lod_index_x64, lod_mask_x64))
        return 6;

    return 7;
}

TraceRecord VoxelWorld::dda(Ray ray, in out int index, in uint /*max_steps*/) {
    TraceRecord result;
    result.default_init();
    result.dist = 0;

    const uint max_steps = (CHUNK_NX + CHUNK_NY + CHUNK_NZ) * CHUNK_SIZE * 3;
    float3 delta = float3(
        ray.nrm.x == 0.0 ? 3.0 * max_steps : abs(ray.inv_nrm.x),
        ray.nrm.y == 0.0 ? 3.0 * max_steps : abs(ray.inv_nrm.y),
        ray.nrm.z == 0.0 ? 3.0 * max_steps : abs(ray.inv_nrm.z));
    uint lod = sample_lod(ray.o, index);
    if (lod == 0) {
        result.hit = true;
        return result;
    }
    float cell_size = float(1l << (lod - 1)) / VOXEL_SCL;
    float3 t_start;
    if (ray.nrm.x < 0) {
        t_start.x = (ray.o.x / cell_size - floor(ray.o.x / cell_size)) * cell_size * delta.x;
    } else {
        t_start.x = (ceil(ray.o.x / cell_size) - ray.o.x / cell_size) * cell_size * delta.x;
    }
    if (ray.nrm.y < 0) {
        t_start.y = (ray.o.y / cell_size - floor(ray.o.y / cell_size)) * cell_size * delta.y;
    } else {
        t_start.y = (ceil(ray.o.y / cell_size) - ray.o.y / cell_size) * cell_size * delta.y;
    }
    if (ray.nrm.z < 0) {
        t_start.z = (ray.o.z / cell_size - floor(ray.o.z / cell_size)) * cell_size * delta.z;
    } else {
        t_start.z = (ceil(ray.o.z / cell_size) - ray.o.z / cell_size) * cell_size * delta.z;
    }
    float t_curr = min(min(t_start.x, t_start.y), t_start.z);
    float3 current_pos;
    float3 t_next = t_start;
    bool outside_bounds = false;
    uint side = 0;
    int x1_steps;
    for (x1_steps = 0; x1_steps < max_steps; ++x1_steps) {
        current_pos = ray.o + ray.nrm * t_curr;
        if (!box.inside(current_pos + ray.nrm * 0.001)) {
            outside_bounds = true;
            result.hit = false;
            break;
        }
        lod = sample_lod(current_pos, index);
        if (lod == 0) {
            result.hit = true;
            if (t_next.x < t_next.y) {
                if (t_next.x < t_next.z) {
                    side = 0;
                } else {
                    side = 2;
                }
            } else {
                if (t_next.y < t_next.z) {
                    side = 1;
                } else {
                    side = 2;
                }
            }
            break;
        }
        cell_size = float(1l << (lod - 1)) / VOXEL_SCL;

        t_next = (0.5 + sign(ray.nrm) * (0.5 - frac(current_pos / cell_size))) * cell_size * delta;
        t_curr += (min(min(t_next.x, t_next.y), t_next.z) + 0.0001 / VOXEL_SCL);
    }
    result.dist = t_curr;
    switch (side) {
    case 0: result.nrm = float3(ray.nrm.x < 0 ? 1 : -1, 0, 0); break;
    case 1: result.nrm = float3(0, ray.nrm.y < 0 ? 1 : -1, 0); break;
    case 2: result.nrm = float3(0, 0, ray.nrm.z < 0 ? 1 : -1); break;
    }
    return result;
}
