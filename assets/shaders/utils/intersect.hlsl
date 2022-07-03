#pragma once

#include "drawing/defines.hlsl"
#include "utils/dda.hlsl"
#include "utils/noise.hlsl"

RayIntersection ray_box_intersect(in Ray ray, float3 b_min, float3 b_max) {
    RayIntersection result;
    float tx1 = (b_min.x - ray.o.x) * ray.inv_nrm.x;
    float tx2 = (b_max.x - ray.o.x) * ray.inv_nrm.x;
    float tmin = min(tx1, tx2);
    float tmax = max(tx1, tx2);
    float ty1 = (b_min.y - ray.o.y) * ray.inv_nrm.y;
    float ty2 = (b_max.y - ray.o.y) * ray.inv_nrm.y;
    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));
    float tz1 = (b_min.z - ray.o.z) * ray.inv_nrm.z;
    float tz2 = (b_max.z - ray.o.z) * ray.inv_nrm.z;
    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));

    result.hit = (tmax >= tmin && tmin > 0);
    result.dist = tmin;
    result.steps = 0;

    bool is_x = tmin == tx1 || tmin == tx2;
    bool is_y = tmin == ty1 || tmin == ty2;
    bool is_z = tmin == tz1 || tmin == tz2;

    if (is_z) {
        if (ray.nrm.z < 0) {
            result.nrm = float3(0, 0, 1);
        } else {
            result.nrm = float3(0, 0, -1);
        }
    } else if (is_y) {
        if (ray.nrm.y < 0) {
            result.nrm = float3(0, 1, 0);
        } else {
            result.nrm = float3(0, -1, 0);
        }
    } else {
        if (ray.nrm.x < 0) {
            result.nrm = float3(1, 0, 0);
        } else {
            result.nrm = float3(-1, 0, 0);
        }
    }

    if (!result.hit) result.nrm = ray.nrm;

    return result;
}

RayIntersection ray_wirebox_intersect(in Ray ray, float3 b_min, float3 b_max, float t) {
    RayIntersection result;
    result.hit = false;
    bool hit = false;
    float dist = 100000;
    float3 nrm;

    result = ray_box_intersect(ray, b_min, float3(b_min.x + t, b_min.y + t, b_max.z));
    if (result.hit && result.dist < dist)
        hit = true, dist = result.dist, nrm = result.nrm;
    result = ray_box_intersect(ray, b_min, float3(b_min.x + t, b_max.y, b_min.z + t));
    if (result.hit && result.dist < dist)
        hit = true, dist = result.dist, nrm = result.nrm;
    result = ray_box_intersect(ray, b_min, float3(b_max.x, b_min.y + t, b_min.z + t));
    if (result.hit && result.dist < dist)
        hit = true, dist = result.dist, nrm = result.nrm;

    result = ray_box_intersect(ray, float3(b_max.x - t, b_min.y, b_min.z), float3(b_max.x, b_min.y + t, b_max.z));
    if (result.hit && result.dist < dist)
        hit = true, dist = result.dist, nrm = result.nrm;
    result = ray_box_intersect(ray, float3(b_min.x, b_min.y, b_max.z - t), float3(b_min.x + t, b_max.y, b_max.z));
    if (result.hit && result.dist < dist)
        hit = true, dist = result.dist, nrm = result.nrm;
    result = ray_box_intersect(ray, float3(b_min.x, b_max.y - t, b_min.z), float3(b_max.x, b_max.y, b_min.z + t));
    if (result.hit && result.dist < dist)
        hit = true, dist = result.dist, nrm = result.nrm;

    result = ray_box_intersect(ray, b_max, float3(b_max.x - t, b_max.y - t, b_min.z));
    if (result.hit && result.dist < dist)
        hit = true, dist = result.dist, nrm = result.nrm;
    result = ray_box_intersect(ray, b_max, float3(b_max.x - t, b_min.y, b_max.z - t));
    if (result.hit && result.dist < dist)
        hit = true, dist = result.dist, nrm = result.nrm;
    result = ray_box_intersect(ray, b_max, float3(b_min.x, b_max.y - t, b_max.z - t));
    if (result.hit && result.dist < dist)
        hit = true, dist = result.dist, nrm = result.nrm;

    result = ray_box_intersect(ray, float3(b_min.x + t, b_max.y, b_max.z), float3(b_min.x, b_max.y - t, b_min.z));
    if (result.hit && result.dist < dist)
        hit = true, dist = result.dist, nrm = result.nrm;
    result = ray_box_intersect(ray, float3(b_max.x, b_max.y, b_min.z + t), float3(b_max.x - t, b_min.y, b_min.z));
    if (result.hit && result.dist < dist)
        hit = true, dist = result.dist, nrm = result.nrm;
    result = ray_box_intersect(ray, float3(b_max.x, b_min.y + t, b_max.z), float3(b_min.x, b_min.y, b_max.z - t));
    if (result.hit && result.dist < dist)
        hit = true, dist = result.dist, nrm = result.nrm;

    result.hit = hit;
    result.dist = dist;
    result.nrm = nrm;

    return result;
}

RayIntersection ray_sphere_intersect(in Ray ray, float3 s0, float sr) {
    RayIntersection result;
    result.hit = false;

    float a = dot(ray.nrm, ray.nrm);
    float3 s0_r0 = ray.o - s0;
    float b = 2.0 * dot(ray.nrm, s0_r0);
    float c = dot(s0_r0, s0_r0) - (sr * sr);
    if (b * b - 4.0 * a * c < 0.0)
        return result;
    result.dist = (-b - sqrt((b * b) - 4.0 * a * c)) / (2.0 * a);
    result.hit = result.dist > 0;
    result.nrm = normalize(get_intersection_pos(ray, result) - s0);
    return result;
}

RayIntersection ray_cylinder_intersect(in Ray ray, in float3 pa, in float3 pb, float ra) {
    RayIntersection result;
    result.hit = false;

    float3 ba = pb - pa;

    float3 oc = ray.o - pa;

    float baba = dot(ba, ba);
    float bard = dot(ba, ray.nrm);
    float baoc = dot(ba, oc);

    float k2 = baba - bard * bard;
    float k1 = baba * dot(oc, ray.nrm) - baoc * bard;
    float k0 = baba * dot(oc, oc) - baoc * baoc - ra * ra * baba;

    float h = k1 * k1 - k2 * k0;
    if (h < 0.0)
        return result;
    h = sqrt(h);
    float t = (-k1 - h) / k2;

    // body
    float y = baoc + t * bard;
    if (y > 0.0 && y < baba) {

        result.dist = t;
        result.hit = t > 0;
        result.nrm = (oc + t * ray.nrm - ba * y / baba) / ra;
        return result;
    }

    // caps
    t = (((y < 0.0) ? 0.0 : baba) - baoc) / bard;
    if (abs(k1 + k2 * t) < h) {
        result.dist = t;
        result.hit = t > 0;
        result.nrm = ba * sign(y) / baba;
    }

    return result;
}

RayIntersection ray_capsule_intersect(in Ray ray, in float3 pa, in float3 pb, float ra) {
    RayIntersection result;
    result.hit = false;
    float3 ba = pb - pa;
    float3 oa = ray.o - pa;
    float baba = dot(ba, ba);
    float bard = dot(ba, ray.nrm);
    float baoa = dot(ba, oa);
    float rdoa = dot(ray.nrm, oa);
    float oaoa = dot(oa, oa);
    float a = baba - bard * bard;
    float b = baba * rdoa - baoa * bard;
    float c = baba * oaoa - baoa * baoa - ra * ra * baba;
    float h = b * b - a * c;
    if (h >= 0.) {
        float t = (-b - sqrt(h)) / a;
        float d = 1e38;
        float y = baoa + t * bard;
        if (y > 0. && y < baba) {
            d = t;
        } else {
            float3 oc = (y <= 0.) ? oa : ray.o - pb;
            b = dot(ray.nrm, oc);
            c = dot(oc, oc) - ra * ra;
            h = b * b - c;
            if (h > 0.0) {
                d = -b - sqrt(h);
            }
        }
        pa = ray.o + ray.nrm * d - pa;
        float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        result.nrm = (pa - h * ba) / ra;
        result.dist = d;
        result.hit = d != 1e38 && d >= 0;
    }
    return result;
}

uint tile_texture_index(StructuredBuffer<Globals> globals, BlockID block_id, BlockFace face) {
    // clang-format off
    switch (block_id) {
    case BlockID::Debug:           return 0;
    case BlockID::Air:             return 1;
    case BlockID::Bedrock:         return 2;
    case BlockID::Brick:           return 3;
    case BlockID::Cactus:          return 4;
    case BlockID::Cobblestone:     return 5;
    case BlockID::CompressedStone: return 6;
    case BlockID::DiamondOre:      return 7;
    case BlockID::Dirt:            return 8;
    case BlockID::DriedShrub:      return 9;
    case BlockID::Grass:
        switch (face) {
        case BlockFace::Back:
        case BlockFace::Front:
        case BlockFace::Left:
        case BlockFace::Right:     return 10;
        case BlockFace::Bottom:    return 8;
        case BlockFace::Top:       return 11;
        default:                   return 0;
        }
    case BlockID::Gravel:          return 12;
    case BlockID::Lava:            return 13 + int(globals[0].time * 6) % 8;
    case BlockID::Leaves:          return 21;
    case BlockID::Log:
        switch (face) {
        case BlockFace::Back:
        case BlockFace::Front:
        case BlockFace::Left:
        case BlockFace::Right:     return 22;
        case BlockFace::Bottom:
        case BlockFace::Top:       return 22;
        default:                   return 0;
        }
    case BlockID::MoltenRock:      return 24;
    case BlockID::Planks:          return 25;
    case BlockID::Rose:            return 26;
    case BlockID::Sand:            return 27;
    case BlockID::Sandstone:       return 28;
    case BlockID::Stone:           return 29;
    case BlockID::TallGrass:       return 11; // 30;
    case BlockID::Water:           return 31;
    default:                       return 0;
    }
    // clang-format on
}

float tile_ior(BlockID block_id) {
    // clang-format off
    switch (block_id) {
    case BlockID::Water: return 1.1;
    default:             return 1.00;
    }
    // clang-format on
}

void get_texture_info(in StructuredBuffer<Globals> globals, in RayIntersection ray_chunk_intersection, in float3 intersection_pos, in BlockID block_id, in out float2 tex_uv, in out BlockFace face_id, in out uint tex_id) {
    if (ray_chunk_intersection.nrm.x > 0.5) {
        face_id = BlockFace::Left;
        tex_uv = frac(intersection_pos.zy);
        tex_uv.y = 1 - tex_uv.y;
    } else if (ray_chunk_intersection.nrm.x < -0.5) {
        face_id = BlockFace::Right;
        tex_uv = frac(intersection_pos.zy);
        tex_uv = 1 - tex_uv;
    }
    if (ray_chunk_intersection.nrm.y > 0.5) {
        face_id = BlockFace::Bottom;
        tex_uv = frac(intersection_pos.xz);
        tex_uv.x = 1 - tex_uv.x;
    } else if (ray_chunk_intersection.nrm.y < -0.5) {
        face_id = BlockFace::Top;
        tex_uv = frac(intersection_pos.xz);
    }
    if (ray_chunk_intersection.nrm.z > 0.5) {
        face_id = BlockFace::Front;
        tex_uv = frac(intersection_pos.xy);
        tex_uv = 1 - tex_uv;
    } else if (ray_chunk_intersection.nrm.z < -0.5) {
        face_id = BlockFace::Back;
        tex_uv = frac(intersection_pos.xy);
        tex_uv.y = 1 - tex_uv.y;
    }

    tex_id = tile_texture_index(globals, block_id, face_id);
}

DAXA_DEFINE_BA_TEXTURE2DARRAY(float4)

RayIntersection ray_step_voxels(StructuredBuffer<Globals> globals, in Ray ray, in float3 b_min, in float3 b_max, in uint max_steps) {
    RayIntersection result;
    result.hit = false;
    result.dist = 0;
    result.nrm = 0;
    result.steps = 0;
    result.col = 1;
    result.pos = 0;

    DDA_RunState dda_run_state;
    dda_run_state.block_id = load_block_id(globals, ray.o);
    BlockID prev_block_id = dda_run_state.block_id;

    if (prev_block_id == BlockID::Water) {
        ray.nrm = normalize(ray.nrm + (rand_vec3(ray.o + ray.nrm) - 0.5) * 0.01);
        ray.inv_nrm = 1.0 / ray.nrm;
    }

    float total_dist = 0;

    for (uint i = 0; i < (ENABLE_REFLECTIONS * 6 + 1) && is_transparent(dda_run_state.block_id); ++i) {
        if (dda_run_state.block_id != prev_block_id) {
            float ripple_scl = 1;
            float x = (noise(globals[0].time * 10 + result.pos.x * 3.3 * ripple_scl + result.pos.z * 3.1 * ripple_scl) - 0.5) * 0.5 * 0.1;
            float y = (noise(globals[0].time * 11 + result.pos.y * 2.7 * ripple_scl + result.pos.x * 3.4 * ripple_scl) - 0.5) * 0.5 * 0.1;
            float z = (noise(globals[0].time * 11 + result.pos.z * 2.9 * ripple_scl + result.pos.y * 3.2 * ripple_scl) - 0.5) * 0.5 * 0.1;
            result.nrm = normalize(result.nrm + float3(x, y, z) * 0.1 + (rand_vec3(result.pos) - 0.5) * 0.01);
            if (rand(result.pos) < pow(dot(ray.nrm, -result.nrm), 0.2)) {
                ray.o = result.pos + ray.nrm * 0.01;
                ray.nrm = refract(ray.nrm, result.nrm, 1.0 / 1.4);
            } else {
                ray.o = result.pos + result.nrm * 0.01;
                ray.nrm = reflect(ray.nrm, result.nrm);
                dda_run_state.block_id = prev_block_id;
            }
            ray.inv_nrm = 1.0 / ray.nrm;
        }
        dda_run_state.outside_bounds = false;
        dda_run_state.side = 0;
        dda_run_state.dist = 0;
        prev_block_id = dda_run_state.block_id;
        run_dda_main(globals, ray, dda_run_state, b_min, b_max, max_steps);
        total_dist += dda_run_state.dist;
        result.steps += dda_run_state.total_steps;
        switch (dda_run_state.side) {
        case 0: result.nrm = float3(ray.nrm.x < 0 ? 1 : -1, 0, 0); break;
        case 1: result.nrm = float3(0, ray.nrm.y < 0 ? 1 : -1, 0); break;
        case 2: result.nrm = float3(0, 0, ray.nrm.z < 0 ? 1 : -1); break;
        }
        result.dist = dda_run_state.dist;
        result.pos = get_intersection_pos_corrected(ray, result);
        if (!dda_run_state.hit)
            break;
        if (prev_block_id == BlockID::Water) {
            result.col *= float3(0.42, 0.95, 1.0) * exp(-dda_run_state.dist * 0.05 + 0.0);
        }
    }

    float3 b_uv = float3(int3(result.pos) % 64) / 64;
    BlockFace face_id;
    float2 tex_uv = float2(0, 0);
    uint tex_id;
    get_texture_info(globals, result, result.pos, dda_run_state.block_id, tex_uv, face_id, tex_id);
    if (tex_id == 8 || tex_id == 11) {
        float r = rand(int3(result.pos));
        switch (int(r * 4) % 4) {
        case 0:
            tex_uv = 1 - tex_uv;
        case 1:
            tex_uv = float2(tex_uv.y, tex_uv.x);
            break;
        case 2:
            tex_uv = 1 - tex_uv;
        case 3:
            tex_uv = float2(tex_uv.x, tex_uv.y);
            break;
        default:
            break;
        }
    }
    result.col *= daxa::getTexture2DArray<float4>(globals[0].texture_index).Load(int4(tex_uv.x * 16, tex_uv.y * 16, tex_id, 0)).rgb;

    result.hit = dda_run_state.hit;
    if (!result.hit) {
        result.nrm = ray.nrm;
        return result;
    }
    switch (dda_run_state.side) {
    case 0: result.nrm = float3(ray.nrm.x < 0 ? 1 : -1, 0, 0); break;
    case 1: result.nrm = float3(0, ray.nrm.y < 0 ? 1 : -1, 0); break;
    case 2: result.nrm = float3(0, 0, ray.nrm.z < 0 ? 1 : -1); break;
    }

    float3 intersection_pos = get_intersection_pos_corrected(ray, result);
    BlockID block_id = load_block_id(globals, intersection_pos);
#if !SHOW_DEBUG_BLOCKS
    if (block_id == BlockID::Debug)
        result.hit = false;
#endif

    result.dist = total_dist;

    return result;
}

RayIntersection trace_chunks(StructuredBuffer<Globals> globals, in Ray ray) {
    float3 b_min = float3(0, 0, 0), b_max = float3(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    RayIntersection result;
    result.hit = false;
    result.dist = 0;
    result.nrm = 0;
    result.steps = 0;
    result.pos = 0;
    result.col = 0;

    float sdf_dist_total = 0;
    uint sdf_step_total = 0;

    if (point_box_contains(ray.o, b_min, b_max)) {
        BlockID block_id = load_block_id(globals, ray.o);
        if (is_block_occluding(block_id) && block_id != BlockID::Debug) {
#if SHOW_DEBUG_BLOCKS
            result.hit = true;
#else
            // result.hit = block_id != BlockID::Debug;
#endif
            result.pos = ray.o;
            return result;
        }
    } else {
        RayIntersection bounds_intersection = ray_box_intersect(ray, b_min, b_max);
        #if !VISUALIZE_SUBGRID
        if (!bounds_intersection.hit)
        #else
            bounds_intersection.pos = get_intersection_pos_corrected(ray, bounds_intersection);
        #endif
            return bounds_intersection;
        sdf_dist_total = bounds_intersection.dist;
        float3 sample_pos = ray.o + ray.nrm * sdf_dist_total;
        if (!point_box_contains(sample_pos, b_min, b_max)) {
            sdf_dist_total += 0.001;
            // sample_pos = ray.o + ray.nrm * sdf_dist_total;
            sample_pos = get_intersection_pos_corrected(ray, bounds_intersection);
        }
        BlockID block_id = load_block_id(globals, sample_pos);
        if (is_block_occluding(block_id) && block_id != BlockID::Debug) {
#if SHOW_DEBUG_BLOCKS
            result.hit = true;
#else
            result.hit = block_id != BlockID::Debug;
#endif
            result.nrm = bounds_intersection.nrm;
            result.dist = sdf_dist_total;
            result.pos = sample_pos;
            return result;
        }
    }

    float3 sample_pos = ray.o + ray.nrm * sdf_dist_total;
    Ray dda_ray = ray;
    dda_ray.o = sample_pos;
    RayIntersection dda_result = ray_step_voxels(globals, dda_ray, b_min, b_max, int(MAX_STEPS));
    sdf_dist_total += dda_result.dist;
    sdf_step_total += dda_result.steps;
    result.nrm = dda_result.nrm;
    if (dda_result.hit) {
        result = dda_result;
    }

    result.dist = sdf_dist_total;
    result.steps = sdf_step_total;
    return result;
}
