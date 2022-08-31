#pragma once

#include "common/interface/game.hlsl"
#include "utils/rand.hlsl"

#include "common/impl/game/_common.hlsl"
#include "common/impl/voxel_world/_drawing.hlsl"

void ShapeScene::eval_color(in out GameTraceState trace_state) {
    switch (shapes[trace_state.shape_i].material_id) {
    case 0: {
        Capsule c = capsules[shapes[trace_state.shape_i].storage_index];
        float3 local_pos = trace_state.draw_sample.pos - c.p1;
        float2x2 rot_mat = float2x2(float2(-c.forward.y, c.forward.x), c.forward);
        local_pos.xy = mul(rot_mat, local_pos.xy);

#define MINION 0
        float2 uv = local_pos.xz;
        float2 e1_uv = uv - float2(0.1, 0.0);
        float e1 = dot(e1_uv, e1_uv) > 0.001;
        float2 e2_uv = uv + float2(0.1, 0.0);
        float e2 = dot(e2_uv, e2_uv) > 0.001;

#if MINION
        float2 m_uv = uv + float2(0.0, 0.2);
#else
        float2 m_uv = uv + float2(0.0, 0.04);
#endif
        float m = clamp((dot(m_uv, m_uv) > 0.02) + (m_uv.y > -0.05), 0, 1);

        float face_fac = clamp(e1 * e2 * m + (local_pos.y < 0) * 10, 0, 1);
        float pants_fac = local_pos.z > -0.6;

        float3 result;
#if MINION
        result = float3(0.90, 0.78, 0.075);
#else
        result = float3(0.60, 0.27, 0.20);
#endif

        float radial = atan2(local_pos.y, local_pos.x) / 6.28 + 0.5;
        float2 b_pocket_pos = float2(abs(abs(radial - 0.5) - 0.25) - 0.075, (local_pos.z + 0.7) * 0.5);
        float2 f_pocket_pos = float2(abs(abs(radial - 0.5) - 0.25) - 0.200, (local_pos.z + 0.7) * 0.5);
        float b_pockets_fac = b_pocket_pos.y < 0.0 && b_pocket_pos.x < 0.04 && b_pocket_pos.x > -0.04 && dot(b_pocket_pos, b_pocket_pos) < 0.003 && local_pos.y < 0;
        float f_pockets_fac = f_pocket_pos.y < 0.0 && f_pocket_pos.x < 0.02 && f_pocket_pos.x > -0.06 && dot(f_pocket_pos, f_pocket_pos) < 0.003 && local_pos.y > 0;
        float belt_fac = (fmod(radial * 20, 1) < 0.8 && local_pos.z > -0.64 && local_pos.z < -0.62);

#if !MINION
        float2 shirt_uv = float2(local_pos.x, (local_pos.z + 0.40) * 4);
        float shirt_fac = shirt_uv.y > 0 || (dot(shirt_uv, shirt_uv) < 0.1 + local_pos.y * 0.1);
        result = lerp(float3(0.4, 0.05, 0.042), result, shirt_fac);
#else
        float goggles_fac = ((dot(e1_uv, e1_uv) > 0.013) && (dot(e2_uv, e2_uv) > 0.013)) || (local_pos.y < 0);
        result = lerp(float3(0.3, 0.3, 0.3), result, goggles_fac);
        float whites_eyes_fac = ((dot(e1_uv, e1_uv) > 0.008) && (dot(e2_uv, e2_uv) > 0.008)) || (local_pos.y < 0);
        result = lerp(float3(1.0, 1.0, 1.0), result, whites_eyes_fac);
#endif
        result = lerp(float3(0.04, 0.04, 0.12), result, pants_fac);
        result = lerp(result, float3(0.03, 0.03, 0.10), b_pockets_fac || f_pockets_fac || (f_pocket_pos.x > 0.045 && local_pos.z < -0.68 && local_pos.z > -1.5));
        result = lerp(float3(0.0, 0.0, 0.0), result, face_fac);
        result = lerp(result, float3(0.04, 0.02, 0.01), belt_fac);

        trace_state.draw_sample.col = result;
    } break;
    case 1: {
        float s = 50;
        float mx = smoothstep(0.45, 0.55, s * 0.5 - abs(s * 0.5 - frac(trace_state.draw_sample.pos.x) * s));
        float my = smoothstep(0.45, 0.55, s * 0.5 - abs(s * 0.5 - frac(trace_state.draw_sample.pos.y) * s));
        float mz = smoothstep(0.45, 0.55, s * 0.5 - abs(s * 0.5 - frac(trace_state.draw_sample.pos.z) * s));
        float3 m = lerp(float3(1, 0, 0), float3(0.01, 0.01, 0.01), mx) / 3 +
                   lerp(float3(0, 1, 0), float3(0.01, 0.01, 0.01), my) / 3 +
                   lerp(float3(0, 0, 1), float3(0.01, 0.01, 0.01), mz) / 3;
        trace_state.draw_sample.col = m;
    } break;
    case 2:
        trace_state.draw_sample.col = float3(0.04, 0.04, 0.12);
        break;
    }
}

static const float3 sun_col = float3(1, 0.75, 0.5) * 2;

float3 sample_sky(float3 nrm) {
    float sky_val = clamp(dot(nrm, float3(0, 0, 1)) * 0.5 + 0.5, 0, 1);
    return lerp(float3(0.11, 0.10, 0.09) * 0.1, float3(0.2, 0.21, 0.90) * 2, pow(sky_val, 1)) * 2;
}

float Game::draw_depth(uint2 pixel_i, float start_depth, in uint max_steps) {
    float2 view_uv = player.camera.create_view_uv(pixel_i);
    Ray ray = player.camera.create_view_ray(view_uv);
    ray.o += ray.nrm * start_depth;
    GameTraceRecord game_trace_record = trace(ray, max_steps);
    game_trace_record.trace_record.dist += start_depth;
    game_trace_record.draw_sample.depth += start_depth;
    return game_trace_record.trace_record.dist;
}

GameTraceRecord Game::trace(Ray ray, in uint max_steps) {
    GameTraceState trace_state;
    trace_state.default_init();
    trace_state.max_steps = max_steps;

    trace_collidables(trace_state, ray);
    debug_scene.trace(trace_state, ray, 2);

    trace_state.draw_sample.pos = ray.o + ray.nrm * trace_state.trace_record.dist;
    trace_state.draw_sample.nrm = trace_state.trace_record.nrm;
    trace_state.draw_sample.depth = trace_state.trace_record.dist;

    if (trace_state.trace_record.hit) {
        switch (trace_state.shape_type) {
        case 0:
            voxel_world.eval_color(trace_state);
            break;
        case 1:
            collidable_scene.eval_color(trace_state);
            break;
        case 2:
            debug_scene.eval_color(trace_state);
            break;
        }
    } else {
        trace_state.draw_sample.col = sample_sky(ray.nrm);
    }

    GameTraceRecord game_trace_record;
    game_trace_record.draw_sample = trace_state.draw_sample;
    game_trace_record.trace_record = trace_state.trace_record;
    return game_trace_record;
}

DrawSample Game::draw(in out Input input, uint2 pixel_i, float start_depth) {
    float uv_rand_offset = input.time;
    float2 uv_offset =
        float2(rand(float2(pixel_i) + uv_rand_offset + 10),
               rand(float2(pixel_i) + uv_rand_offset)) *
        1.0;

    float2 view_uv = player.camera.create_view_uv(pixel_i + uv_offset);
    Ray ray = player.camera.create_view_ray(view_uv);
    ray.o += ray.nrm * start_depth;

    GameTraceRecord game_trace_record = trace(ray, input.max_steps());
    game_trace_record.trace_record.dist += start_depth;
    game_trace_record.draw_sample.depth += start_depth;
    GameTraceRecord temp_trace_record;

    if (game_trace_record.trace_record.hit) {
        if (input.shadows_enabled()) {
            ray.o = game_trace_record.draw_sample.pos + game_trace_record.draw_sample.nrm * 0.001;
            ray.nrm = sun_nrm;
            ray.inv_nrm = 1 / ray.nrm;
            temp_trace_record = trace(ray, input.max_steps());
            float3 shade = max(dot(game_trace_record.draw_sample.nrm, sun_nrm), 0) * sun_col;
            if (temp_trace_record.trace_record.hit) {
                shade = 0;
            }
            shade += sample_sky(game_trace_record.draw_sample.nrm) * 0.3;

            // float val = dot(float(0, 0, -1), game_trace_record.draw_sample.nrm) * 0.5 + 0.5;
            // val = pow(val, 2);
            // shade *= val;

            game_trace_record.draw_sample.col *= shade;
        } else {
            float3 shade = max(dot(sun_nrm, game_trace_record.draw_sample.nrm), 0) * sun_col;
            shade += sample_sky(game_trace_record.draw_sample.nrm) * 0.3;

            // float val = dot(float(0, 0, -1), game_trace_record.draw_sample.nrm) * 0.5 + 0.5;
            // val = pow(val, 2);
            // shade *= val;
            
            game_trace_record.draw_sample.col *= shade;

        }
    }

    float3 p = floor(game_trace_record.draw_sample.pos * VOXEL_SCL) / VOXEL_SCL;
    float p_dist = length(p - pick_pos[0]);
    if (p_dist < EDIT_RADIUS && p_dist > EDIT_RADIUS - 2.0 / VOXEL_SCL) {
        game_trace_record.draw_sample.col += float3(0.07, 0.08, 0.1);
    }

    // float3 chunk_p = floor(game_trace_record.draw_sample.pos * VOXEL_SCL / CHUNK_SIZE + 0.5) / VOXEL_SCL * CHUNK_SIZE;
    // float chunk_p_dist = length(chunk_p - pick_pos[0]) - CHUNK_SIZE / VOXEL_SCL;
    // if (chunk_p_dist < EDIT_RADIUS) {
    //     game_trace_record.draw_sample.col += float3(0.3, 0.08, 0.1);
    // }

    return game_trace_record.draw_sample;
}
