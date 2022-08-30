#pragma once

uint VoxelWorld::get_chunk_index(int3 chunk_i) {
    return chunk_i.x + chunk_i.y * CHUNK_COUNT_X + chunk_i.z * CHUNK_COUNT_X * CHUNK_COUNT_Y;
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
            bound_trace = dda(dda_ray, chunk_index);
            bound_trace.dist += prev_dist;
        }
    }
    if (bound_trace.hit && bound_trace.dist < trace_state.trace_record.dist) {
        trace_state.trace_record = bound_trace;
        trace_state.shape_i = chunk_index;
        trace_state.shape_type = 0;
    }
}

TraceRecord VoxelWorld::dda(Ray ray, in out int index) {
    TraceRecord result;
    result.default_init();
    result.dist = 0;

    const uint max_steps = (CHUNK_COUNT_X + CHUNK_COUNT_Y + CHUNK_COUNT_Z) * CHUNK_SIZE * 3;
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
