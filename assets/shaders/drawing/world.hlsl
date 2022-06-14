#pragma once

#define sun_col float3(1.0, 0.84, 0.75) * 2.5

float3 sample_sky(in Ray sun_ray, float3 nrm) {
    float sun_val = clamp((dot(nrm, sun_ray.nrm) - 0.9999) * 10000, 0, 1);
    float sky_val = (dot(nrm, float3(0, 1, 0))) / 2 + clamp((dot(nrm, sun_ray.nrm) + 1) * 0.1, 0, 1);
    return sun_col * sun_val + lerp(float3(0.4, 0.6, 0.8), float3(0.3, 0.4, 0.1), sky_val);
}

float3 shaded(in Ray sun_ray, in float3 col, in float3 nrm) {
    float3 light = sun_col * max(dot(nrm, sun_ray.nrm), 0) + sample_sky(sun_ray, nrm) * 0.6;
    return col * light;
}

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

void draw_world(
    in StructuredBuffer<Globals> globals,
    in StructuredBuffer<PlayerBuffer> player_buffer,
    in float2 uv, in uint3 pixel_i,
    in float2 subsamples, in float2 inv_subsamples,
    in float2 inv_frame_dim, in float aspect,
    in out float3 color, in out float depth) {
    float3 front;
    float3 right;
    float3 up;
    Ray cam_ray;

    // if (pixel_i.x < globals[0].frame_dim.x / 2) {
    //     front = mul(globals[0].viewproj_mat, float4(0, 0, 1, 0)).xyz;
    //     right = mul(globals[0].viewproj_mat, float4(1, 0, 0, 0)).xyz;
    //     up = mul(globals[0].viewproj_mat, float4(0, 1, 0, 0)).xyz;
    //     cam_ray.o = globals[0].pos.xyz;
    // } else {
    float4x4 view_mat = player_buffer[0].player.camera.view_mat;
    front = mul(view_mat, float4(0, 0, 1, 0)).xyz;
    right = mul(view_mat, float4(1, 0, 0, 0)).xyz;
    up = mul(view_mat, float4(0, 1, 0, 0)).xyz;
    cam_ray.o = player_buffer[0].player.pos.xyz;
    // }

    float3 view_intersection_pos = globals[0].pick_pos[0].xyz;
    int3 view_intersection_block_pos = int3(view_intersection_pos);

    Ray sun_ray;
    float sun_angle = 0.3;
    float sun_yz = -abs(cos(sun_angle)) * 2;
    sun_ray.nrm = normalize(float3(-sin(sun_angle) * 3, sun_yz, -0.2 * sun_yz));
    sun_ray.inv_nrm = 1 / sun_ray.nrm;

    for (uint yi = 0; yi < subsamples.y; ++yi) {
        for (uint xi = 0; xi < subsamples.x; ++xi) {
            float2 view_uv = (uv + inv_frame_dim * float2(xi, yi) * inv_subsamples) * globals[0].fov * float2(aspect, 1);

            cam_ray.nrm = normalize(front + view_uv.x * right + view_uv.y * up);
            cam_ray.inv_nrm = 1 / cam_ray.nrm;

            RayIntersection ray_chunk_intersection = trace_chunks(globals, cam_ray);
#if VISUALIZE_STEP_COMPLEXITY
            // color.r += float(ray_chunk_intersection.steps) * 10.0 / MAX_STEPS;
            color.r += log(float(ray_chunk_intersection.steps) * 1 / MAX_STEPS + 1) * 10;
#elif 0
            float3 intersection_pos = get_intersection_pos_corrected(cam_ray, ray_chunk_intersection);
            BlockID block_id = load_block_id(globals, intersection_pos);
            int3 intersection_block_pos = int3(intersection_pos);
            if (ray_chunk_intersection.hit) {
                uint lod = get_lod(globals, intersection_block_pos);
                color.rgb += lod;
            }
#elif 0
            if (ray_chunk_intersection.hit) {
                float3 mask = abs(ray_chunk_intersection.nrm);
                float3 intersection_pos = get_intersection_pos_corrected(cam_ray, ray_chunk_intersection);
                float3 v_pos = intersection_pos - ray_chunk_intersection.nrm * 0.01;
                float3 b_pos = floor(v_pos + ray_chunk_intersection.nrm * 0.1);
                float3 d1 = mask.zxy;
                float3 d2 = mask.yzx;
                int temp_chunk_index;
                float4 side = float4(
                    load_block_id(globals, b_pos + d1) != BlockID::Air,
                    load_block_id(globals, b_pos + d2) != BlockID::Air,
                    load_block_id(globals, b_pos - d1) != BlockID::Air,
                    load_block_id(globals, b_pos - d2) != BlockID::Air);
                float4 corner = float4(
                    load_block_id(globals, b_pos + d1 + d2) != BlockID::Air,
                    load_block_id(globals, b_pos - d1 + d2) != BlockID::Air,
                    load_block_id(globals, b_pos - d1 - d2) != BlockID::Air,
                    load_block_id(globals, b_pos + d1 - d2) != BlockID::Air);
                float2 ao_uv = fmod(float2(dot(mask * v_pos.yzx, float3(1, 1, 1)), dot(mask * v_pos.zxy, float3(1, 1, 1))), float2(1, 1));
                float4 ao = voxel_ao(side, corner);
                float interp_ao = lerp(lerp(ao.z, ao.w, ao_uv.x), lerp(ao.y, ao.x, ao_uv.x), ao_uv.y);
                color.rgb = 0.5 + interp_ao * 0.5;
            }
#else
            float3 intersection_pos = get_intersection_pos_corrected(cam_ray, ray_chunk_intersection);
            BlockID block_id = load_block_id(globals, intersection_pos);
            int3 intersection_block_pos = int3(intersection_pos);

            float weather = 0; // sin(globals[0].time) * 0.5 + 0.5;

            if (ray_chunk_intersection.hit) {
#if ENABLE_SHADOWS
                sun_ray.o = intersection_pos + ray_chunk_intersection.nrm * 0.002;
                RayIntersection sun_ray_chunk_intersection = trace_chunks(globals, sun_ray);
                float val = float(!sun_ray_chunk_intersection.hit);
                val = max(val * dot(ray_chunk_intersection.nrm, sun_ray.nrm), 0.01);
                float3 light = val * (1 - weather) * sun_col;
#else
                // float3 light = shaded(sun_ray, float3(1, 1, 1), ray_chunk_intersection.nrm);
                float3 light = float3(0, 0, 0);
#endif

#if ENABLE_FAKE_SKY_LIGHTING
                float3 mask = abs(ray_chunk_intersection.nrm);
                float3 intersection_pos = get_intersection_pos_corrected(cam_ray, ray_chunk_intersection);
                float3 v_pos = intersection_pos - ray_chunk_intersection.nrm * 0.01;
                float3 b_pos = floor(v_pos + ray_chunk_intersection.nrm * 0.1);
                float3 d1 = mask.zxy;
                float3 d2 = mask.yzx;
                int temp_chunk_index;
                float4 side = float4(
                    load_block_id(globals, b_pos + d1) != BlockID::Air,
                    load_block_id(globals, b_pos + d2) != BlockID::Air,
                    load_block_id(globals, b_pos - d1) != BlockID::Air,
                    load_block_id(globals, b_pos - d2) != BlockID::Air);
                float4 corner = float4(
                    load_block_id(globals, b_pos + d1 + d2) != BlockID::Air,
                    load_block_id(globals, b_pos - d1 + d2) != BlockID::Air,
                    load_block_id(globals, b_pos - d1 - d2) != BlockID::Air,
                    load_block_id(globals, b_pos + d1 - d2) != BlockID::Air);
                float2 ao_uv = fmod(float2(dot(mask * v_pos.yzx, float3(1, 1, 1)), dot(mask * v_pos.zxy, float3(1, 1, 1))), float2(1, 1));
                float4 ao = voxel_ao(side, corner);
                float interp_ao = clamp(lerp(lerp(ao.z, ao.w, ao_uv.x), lerp(ao.y, ao.x, ao_uv.x), ao_uv.y), 0, 1);
                // interp_ao = pow(interp_ao, 0.25);
                light += sample_sky(sun_ray, ray_chunk_intersection.nrm) * 0.5 * (interp_ao * 0.9 + 0.1);
#elif !ENABLE_SHADOWS
                light = 1;
#endif

                float3 b_uv = float3(int3(intersection_pos) % 64) / 64;
                BlockFace face_id;
                float2 tex_uv = float2(0, 0);
                depth = ray_chunk_intersection.dist;
                uint tex_id;

                get_texture_info(globals, ray_chunk_intersection, intersection_pos, block_id, tex_uv, face_id, tex_id);

                if (tex_id == 8 || tex_id == 11) {
                    float r = rand(int3(intersection_pos));
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

#if ALBEDO == ALBEDO_TEXTURE
                // WTF!!!
                // float3 albedo = daxa::getTexture2DArray<float4>(globals[0].texture_index).Sample(
                //     daxa::getSampler<void>(globals[0].sampler_index),
                //     float3(tex_uv.x, tex_uv.y, float(tex_id)),
                //     int2(0, 0),
                //     1.0f).rgb;
                float3 albedo = daxa::getTexture2DArray<float4>(globals[0].texture_index).Load(int4(tex_uv.x * 16, tex_uv.y * 16, tex_id, 0)).rgb;
                // float3 albedo = 1;
#elif ALBEDO == ALBEDO_DEBUG_POS
                float3 albedo = b_uv;
#elif ALBEDO == ALBEDO_DEBUG_NRM
                float3 albedo = ray_chunk_intersection.nrm * 0.5 + 0.5;
#elif ALBEDO == ALBEDO_DEBUG_DIST
                float d = ray_chunk_intersection.dist;
                float3 albedo = float3(d, d, d) * 0.01;
#elif ALBEDO == ALBEDO_DEBUG_RANDOM
                // float3 albedo = float3(rand(int3(intersection_pos)), rand(int3(intersection_pos + 10)), rand(int3(intersection_pos + 20)));
                float3 albedo = float3(0.5, 0.5, 0.5);
#endif
#if SHOW_PICK_POS
                if (length(intersection_block_pos - view_intersection_block_pos) <= 0.5 + BLOCKEDIT_RADIUS) {
                    float luminance = (albedo.r * 0.2126 + albedo.g * 0.7152 + albedo.b * 0.0722);
                    const float block_outline = 1.0 / 16;
                    albedo = float3(0.2, 0.5, 0.9) * 0.2 + luminance;
                    if (tex_uv.x < block_outline || tex_uv.x > 1 - block_outline || tex_uv.y < block_outline || tex_uv.y > 1 - block_outline)
                        albedo = float3(0.1, 0.4, 1.0) * 0.2 + luminance * 2;
                }
#endif
                color += albedo * light;
                float dist_exp = 1 - exp(pow(ray_chunk_intersection.dist * lerp(0.005, 0.03, weather), 2) * -0.01);
                dist_exp = clamp(dist_exp, 0, 1);
                color = lerp(color, sample_sky(sun_ray, cam_ray.nrm), dist_exp);
            } else {
                color += sample_sky(sun_ray, cam_ray.nrm);
            }
#endif
        }
    }

    // {
    //     RayIntersection temp_inter = ray_box_intersect(cam_ray, view_intersection_block_pos, view_intersection_block_pos + 1.0);
    //     overlay(color, depth, shaded(sun_ray, float3(1, 1, 1) * 0.5, temp_inter.nrm), temp_inter.dist, temp_inter.hit, 0.2);
    // }

    color *= inv_subsamples.x * inv_subsamples.y;
}
