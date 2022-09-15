#version 450

#include <shared/shared.inl>
#include <utils/voxel.glsl>

DAXA_USE_PUSH_CONSTANT(DrawCompPush)

#define MAX_DIST 10000.0

struct Ray {
    f32vec3 o;
    f32vec3 nrm;
    f32vec3 inv_nrm;
};
struct IntersectionRecord {
    b32 hit;
    f32 internal_fac;
    f32 dist;
    f32vec3 nrm;
};
struct TraceRecord {
    IntersectionRecord intersection_record;
    f32vec3 color;
    u32 material;
};

void default_init(out IntersectionRecord result) {
    result.hit = false;
    result.internal_fac = 1.0;
    result.dist = MAX_DIST;
    result.nrm = f32vec3(0, 0, 0);
}

u32 sample_lod(f32vec3 p) {
    // i32vec3 block_i = clamp(i32vec3((p - CHUNK.box.bound_min) / VOXEL_SCL), i32vec3(0 + 0.001, 0 + 0.001, 0 + 0.001), i32vec3(CHUNK_SIZE - 0.001, CHUNK_SIZE - 0.001, CHUNK_SIZE - 0.001));
    // i32vec3 chunk_i = block_i / CHUNK_SIZE;
    // i32vec3 inchunk_block_i = block_i - chunk_i * CHUNK_SIZE;
    // u32 block_index = inchunk_block_i.x + inchunk_block_i.y * CHUNK_SIZE + inchunk_block_i.z * CHUNK_SIZE * CHUNK_SIZE;
    // Voxel voxel = unpack_voxel(CHUNK.packed_voxels[block_index]);
    // if (voxel.block_id == 0)
    //     return 1;
    return 0;
}
IntersectionRecord dda(Ray ray) {
    IntersectionRecord result;
    default_init(result);
    result.dist = 0;

    const u32 max_steps = CHUNK_SIZE * 3;
    f32vec3 delta = f32vec3(
        ray.nrm.x == 0.0 ? 3.0 * max_steps : abs(ray.inv_nrm.x),
        ray.nrm.y == 0.0 ? 3.0 * max_steps : abs(ray.inv_nrm.y),
        ray.nrm.z == 0.0 ? 3.0 * max_steps : abs(ray.inv_nrm.z));
    u32 lod = sample_lod(ray.o);
    if (lod == 0) {
        result.hit = true;
        return result;
    }
    f32 cell_size = f32(1l << (lod - 1)) / VOXEL_SCL;
    f32vec3 t_start;
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
    f32 t_curr = min(min(t_start.x, t_start.y), t_start.z);
    f32vec3 current_pos;
    f32vec3 t_next = t_start;
    b32 outside_bounds = false;
    u32 side = 0;
    i32 x1_steps;
    for (x1_steps = 0; x1_steps < max_steps; ++x1_steps) {
        current_pos = ray.o + ray.nrm * t_curr;
        // if (inside(current_pos + ray.nrm * 0.001, CHUNK.box) == false) {
        //     outside_bounds = true;
        //     result.hit = false;
        //     break;
        // }
        lod = sample_lod(current_pos);
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
        cell_size = f32(1l << (lod - 1)) / VOXEL_SCL;

        t_next = (0.5 + sign(ray.nrm) * (0.5 - fract(current_pos / cell_size))) * cell_size * delta;
        t_curr += (min(min(t_next.x, t_next.y), t_next.z) + 0.0001 / VOXEL_SCL);
    }
    result.dist = t_curr;
    switch (side) {
    case 0: result.nrm = f32vec3(ray.nrm.x < 0 ? 1 : -1, 0, 0); break;
    case 1: result.nrm = f32vec3(0, ray.nrm.y < 0 ? 1 : -1, 0); break;
    case 2: result.nrm = f32vec3(0, 0, ray.nrm.z < 0 ? 1 : -1); break;
    }
    return result;
}
IntersectionRecord intersect(Ray ray, Sphere s) {
    IntersectionRecord result;
    default_init(result);

    f32vec3 so_r = ray.o - s.o;
    f32 a = dot(ray.nrm, ray.nrm);
    f32 b = 2.0f * dot(ray.nrm, so_r);
    f32 c = dot(so_r, so_r) - (s.r * s.r);
    f32 f = b * b - 4.0f * a * c;
    if (f < 0.0f)
        return result;
    result.internal_fac = (c > 0) ? 1.0 : -1.0;
    result.dist = (-b - sqrt(f) * result.internal_fac) / (2.0f * a);
    result.hit = result.dist > 0.0f;
    result.nrm = normalize(ray.o + ray.nrm * result.dist - s.o) * result.internal_fac;
    return result;
}
IntersectionRecord intersect(Ray ray, Box b) {
    IntersectionRecord result;
    default_init(result);

    f32 tx1 = (b.bound_min.x - ray.o.x) * ray.inv_nrm.x;
    f32 tx2 = (b.bound_max.x - ray.o.x) * ray.inv_nrm.x;
    f32 tmin = min(tx1, tx2);
    f32 tmax = max(tx1, tx2);
    f32 ty1 = (b.bound_min.y - ray.o.y) * ray.inv_nrm.y;
    f32 ty2 = (b.bound_max.y - ray.o.y) * ray.inv_nrm.y;
    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));
    f32 tz1 = (b.bound_min.z - ray.o.z) * ray.inv_nrm.z;
    f32 tz2 = (b.bound_max.z - ray.o.z) * ray.inv_nrm.z;
    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));

    result.hit = false;
    if (tmax >= tmin) {
        if (tmin > 0) {
            result.dist = tmin;
            result.hit = true;
        } else if (tmax > 0) {
            result.dist = tmax;
            result.hit = true;
            result.internal_fac = -1.0;
            tmin = tmax;
        }
    }

    b32 is_x = tmin == tx1 || tmin == tx2;
    b32 is_y = tmin == ty1 || tmin == ty2;
    b32 is_z = tmin == tz1 || tmin == tz2;

    if (is_z) {
        if (ray.nrm.z < 0) {
            result.nrm = f32vec3(0, 0, 1);
        } else {
            result.nrm = f32vec3(0, 0, -1);
        }
    } else if (is_y) {
        if (ray.nrm.y < 0) {
            result.nrm = f32vec3(0, 1, 0);
        } else {
            result.nrm = f32vec3(0, -1, 0);
        }
    } else {
        if (ray.nrm.x < 0) {
            result.nrm = f32vec3(1, 0, 0);
        } else {
            result.nrm = f32vec3(-1, 0, 0);
        }
    }
    result.nrm *= result.internal_fac;

    return result;
}
IntersectionRecord intersect_chunk(Ray ray) {
    IntersectionRecord result;
    default_init(result);
    result = intersect(ray, VOXEL_WORLD.box);

    // if (inside(ray.o, CHUNK.box)) {
    //     result.dist = 0;
    //     result.hit = true;
    // } else {
    //     result = intersect(ray, VOXEL_WORLD.box);
    //     result.dist += 0.0001;
    // }

    // Ray dda_ray = ray;
    // dda_ray.o = ray.o + ray.nrm * result.dist;

    // if (result.hit && sample_lod(dda_ray.o) != 0) {
    //     f32 prev_dist = result.dist;
    //     result = dda(dda_ray);
    //     result.dist += prev_dist;
    // }

    return result;
}
Ray create_view_ray(f32vec2 uv) {
    Ray result;

    result.o = push_constant.gpu_globals.player.cam.pos;
    result.nrm = normalize(f32vec3(uv.x * push_constant.gpu_globals.player.cam.tan_half_fov, 1, -uv.y * push_constant.gpu_globals.player.cam.tan_half_fov));
    result.nrm = push_constant.gpu_globals.player.cam.rot_mat * result.nrm;

    return result;
}
u32 scene_id(f32vec3 p) {
    return 1;
}
TraceRecord trace_scene(in Ray ray) {
    TraceRecord trace;
    default_init(trace.intersection_record);
    trace.color = f32vec3(0.3, 0.4, 0.9);
    trace.material = 0;

    for (u32 i = 0; i < SCENE.sphere_n; ++i) {
        Sphere s = SCENE.spheres[i];
        IntersectionRecord s_hit = intersect(ray, s);
        if (s_hit.hit && s_hit.dist < trace.intersection_record.dist) {
            trace.intersection_record = s_hit;
            trace.color = f32vec3(0.5, 0.5, 1.0);
            trace.material = 1;
        }
    }

    for (u32 i = 0; i < SCENE.box_n; ++i) {
        Box b = SCENE.boxes[i];
        IntersectionRecord b_hit = intersect(ray, b);
        if (b_hit.hit && b_hit.dist < trace.intersection_record.dist) {
            trace.intersection_record = b_hit;
            trace.color = f32vec3(1.0, 0.5, 0.5);
            trace.material = 1;
        }
    }

    IntersectionRecord b0_hit = intersect_chunk(ray);
    if (b0_hit.hit && b0_hit.dist < trace.intersection_record.dist) {
        trace.intersection_record = b0_hit;
        trace.color = f32vec3(0.5, 1.0, 0.5);
        trace.material = 2;
    }

    return trace;
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec3 pixel_i = gl_GlobalInvocationID.xyz;
    if (pixel_i.x >= push_constant.gpu_input.frame_dim.x ||
        pixel_i.y >= push_constant.gpu_input.frame_dim.y)
        return;

    f32vec2 pixel_p = pixel_i.xy;
    f32vec2 frame_dim = push_constant.gpu_input.frame_dim;
    f32vec2 inv_frame_dim = f32vec2(1.0, 1.0) / frame_dim;
    f32 aspect = frame_dim.x * inv_frame_dim.y;

    f32vec2 uv = pixel_p * inv_frame_dim;
    uv = (uv - 0.5) * f32vec2(aspect, 1.0) * 2.0;

    f32vec3 col = f32vec3(0, 0, 0);

    Ray view_ray = create_view_ray(uv);
    view_ray.inv_nrm = 1.0 / view_ray.nrm;

    TraceRecord view_trace_record = trace_scene(view_ray);
    col = view_trace_record.color;

    if (view_trace_record.intersection_record.hit) {
        f32vec3 hit_pos = view_ray.o + view_ray.nrm * view_trace_record.intersection_record.dist;
        f32vec3 hit_nrm = view_trace_record.intersection_record.nrm;
        f32 hit_dist = view_trace_record.intersection_record.dist;
        Ray bounce_ray;
        bounce_ray.o = hit_pos;

        switch (view_trace_record.material) {
        case 0: {
            // bounce_ray.nrm = reflect(view_ray.nrm, hit_nrm);
            bounce_ray.nrm = refract(view_ray.nrm, hit_nrm, 1.0 / 1.4);
            bounce_ray.o -= view_trace_record.intersection_record.nrm * 0.001;
        } break;
        case 1: {
            bounce_ray.nrm = normalize(f32vec3(1, -2, 3));
            bounce_ray.o += view_trace_record.intersection_record.nrm * 0.001;
        } break;
        case 2: {
            bounce_ray.nrm = normalize(f32vec3(1, -2, 3));
            bounce_ray.o += view_trace_record.intersection_record.nrm * 0.001;
        } break;
        }
        bounce_ray.inv_nrm = 1.0 / bounce_ray.nrm;
        f32 shade = max(dot(bounce_ray.nrm, view_trace_record.intersection_record.nrm) * 0.5 + 0.5, 0.0);
        // TraceRecord bounce_trace_record = trace_scene(bounce_ray);
        switch (view_trace_record.material) {
        case 0: {
            // col = bounce_trace_record.color;
        } break;
        case 1: {
            // col = f32vec3(1, 0, 1);
            // f32 shade = max(dot(bounce_ray.nrm, view_trace_record.intersection_record.nrm), 0.0);
            // shade *= f32(!bounce_trace_record.intersection_record.hit);
            col *= shade;
        } break;
        case 2: {
            f32vec3 voxel_p = clamp((hit_pos - f32vec3(-2, -2, -2)) * 0.25, f32vec3(0.0001), f32vec3(0.9999));
            u32vec3 voxel_i = u32vec3(voxel_p * CHUNK_SIZE);
            u32 voxel_index = voxel_i.x + voxel_i.y * CHUNK_SIZE + voxel_i.z * CHUNK_SIZE * CHUNK_SIZE;
            // Voxel v = unpack_voxel(CHUNK.packed_voxels[voxel_index]);
            // col = f32vec3(v.block_id);
            // col = v.col;
            // col = v.nrm;
            // col = voxel_p;
            // col = hit_nrm;
            // shade *= max(f32(!bounce_trace_record.intersection_record.hit), 0.1);
            col *= shade;
        } break;
        }

        // col = hit_pos;
        // col = hit_nrm;
        // col = reflect(view_ray.nrm, hit_nrm);
        // col = refract(view_ray.nrm, hit_nrm, 1.0 / 1.4);
        // col = f32vec3(hit_dist);
        // col = view_ray.nrm;
        // col = bounce_ray.nrm;

        // Ray sun_ray;
        // sun_ray.o = hit_pos + view_trace_record.intersection_record.nrm * 0.001;
        // sun_ray.nrm = normalize(f32vec3(1, -2, 3));
        // sun_ray.inv_nrm = 1.0 / sun_ray.nrm;
        // TraceRecord sun_trace_record = trace_scene(sun_ray);
    }

    imageStore(
        daxa_GetRWImage(image2D, rgba32f, push_constant.image_id),
        i32vec2(pixel_i.xy),
        f32vec4(col, 1));
}
