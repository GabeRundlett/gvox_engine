#include <shared/shared.inl>

DAXA_USE_PUSH_CONSTANT(RaytraceCompPush)

#include <utils/rand.glsl>
#include <utils/raytrace.glsl>

struct RayState {
    u32vec2 result_i;
    b32 currently_tracing;
    b32 has_ray_result;

    Ray ray;
    f32vec3 delta;
    f32vec3 t_start;
    f32vec3 t_next;
    f32vec3 current_pos;
    f32 t_curr;
    f32 dist;
    u32 lod;
    u32 side;
};

struct RaySetupInfo {
    u32 result_index;
    u32vec2 result_i;

    b32 hit;
    f32vec3 ray_dir;
    f32vec3 delta;
    f32vec3 t_start;
};

#define WARP_SIZE (32)
#define BACKLOG_SIZE (WARP_SIZE * 2)
#define STEPS_UNTIL_REORDER (8)

// We keep a backlog of rays.
// This backlog contains a list of prepared rays that can immediately start tracing.
// Running threads in the warp can just grab a new ray from this list when they are dont tracing their current ray.
// Having the ray prepreation work done non-divergently is important as the rays can potentially diverge heavily.
// The backlog gets emptied one by one until there are no rays left.
// When its empty, 64 new rays are acquired and ALL threads in a warp prepare the 64 new rays.
struct RayBacklog {
    i32 size;
    i32 start_index;
    b32 need_regeneration;
    RaySetupInfo ray_setup_infos[BACKLOG_SIZE];
};

shared RayBacklog ray_backlog;

u32vec2 get_result_i(u32 result_index) {
    u32vec2 result;
    result.y = result_index / INPUT.frame_dim.x;
    result.x = result_index - result.y * INPUT.frame_dim.x;
    return result;
}

Ray create_view_ray(u32vec2 pixel_i) {
    f32vec2 pixel_p = pixel_i;
    f32vec2 frame_dim = INPUT.frame_dim;
    f32vec2 inv_frame_dim = f32vec2(1.0, 1.0) / frame_dim;
    f32 aspect = frame_dim.x * inv_frame_dim.y;
    f32 uv_rand_offset = INPUT.time;
    f32vec2 uv_offset = f32vec2(rand(pixel_p + uv_rand_offset + 10), rand(pixel_p + uv_rand_offset)) * 1.0 - 0.5;
    pixel_p += uv_offset * INPUT.settings.jitter_scl;
    f32vec2 uv = pixel_p * inv_frame_dim;
    uv = (uv - 0.5) * f32vec2(aspect, 1.0) * 2.0;
    return create_view_ray(uv);
}

RaySetupInfo raytrace_setup(u32 ray_backlog_index) {
    RaySetupInfo result;
    result.result_index = ray_backlog.start_index - ray_backlog_index;
    result.result_i = get_result_i(result.result_index);
    Ray ray = create_view_ray(result.result_i);
    ray.inv_nrm = 1.0 / ray.nrm;
    const u32 max_steps = WORLD_BLOCK_NX + WORLD_BLOCK_NY + WORLD_BLOCK_NZ;
    f32vec3 delta = f32vec3(
        ray.nrm.x == 0.0 ? 3.0 * max_steps : abs(ray.inv_nrm.x),
        ray.nrm.y == 0.0 ? 3.0 * max_steps : abs(ray.inv_nrm.y),
        ray.nrm.z == 0.0 ? 3.0 * max_steps : abs(ray.inv_nrm.z));
    u32 chunk_index;
    u32 lod = sample_lod(ray.o, chunk_index);
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
    return result;
}

void raytrace_begin(in RaySetupInfo ray_begin_info, in out RayState ray_state) {
    ray_state.result_i = ray_begin_info.result_i;
    // ray_state.currently_tracing = true;
    ray_state.has_ray_result = false;
    ray_state.ray = create_view_ray(ray_state.result_i);

    ray_state.dist = 0.5;
}

void raytrace_end(in out RayState ray_state) {
    ray_state.dist += ray_state.t_curr;
    // switch (ray_state.side) {
    // case 0: ray_state.nrm = f32vec3(ray_state.ray.nrm.x < 0 ? 1 : -1, 0, 0); break;
    // case 1: ray_state.nrm = f32vec3(0, ray_state.ray.nrm.y < 0 ? 1 : -1, 0); break;
    // case 2: ray_state.nrm = f32vec3(0, 0, ray_state.ray.nrm.z < 0 ? 1 : -1); break;
    // }
}

void raytrace_body(in out RayState ray_state) {
    ray_state.currently_tracing = false;
    ray_state.has_ray_result = true;
    return;

    // ray_state.current_pos = ray_state.ray.o + ray_state.ray.nrm * ray_state.t_curr;
    // if (inside(ray_state.current_pos + ray_state.ray.nrm * 0.001, VOXEL_WORLD.box) == false) {
    //     ray_state.currently_tracing = false;
    //     ray_state.has_ray_result = true;
    //     return;
    // }
    // u32 chunk_index;
    // ray_state.lod = sample_lod(ray_state.current_pos, chunk_index);
    // if (ray_state.lod == 0) {
    //     ray_state.currently_tracing = false;
    //     ray_state.has_ray_result = true;
    //     if (ray_state.t_next.x < ray_state.t_next.y) {
    //         if (ray_state.t_next.x < ray_state.t_next.z) {
    //             ray_state.side = 0;
    //         } else {
    //             ray_state.side = 2;
    //         }
    //     } else {
    //         if (ray_state.t_next.y < ray_state.t_next.z) {
    //             ray_state.side = 1;
    //         } else {
    //             ray_state.side = 2;
    //         }
    //     }
    //     return;
    // }
    // f32 cell_size = f32(1l << (ray_state.lod - 1)) / VOXEL_SCL;
    // ray_state.t_next = (0.5 + sign(ray_state.ray.nrm) * (0.5 - fract(ray_state.current_pos / cell_size))) * cell_size * ray_state.delta;
    // ray_state.t_curr += (min(min(ray_state.t_next.x, ray_state.t_next.y), ray_state.t_next.z) + 0.001 / 8);
}

void write_ray_result(RayState ray_state) {
    // raytrace_end(ray_state);
    imageStore(
        get_image(image2D, RT_IMAGE),
        i32vec2(ray_state.result_i),
        f32vec4(0, 0.5, 0.2, 1));
}

layout(local_size_x = WARP_SIZE, local_size_y = 1, local_size_z = 1) in;
void main() {
    // One thread from each group will take 64 "jobs" from the global job count
    if (subgroupElect()) {
        ray_backlog.start_index = atomicAdd(GLOBALS.ray_count, -64);
        ray_backlog.size = min(ray_backlog.start_index, 64);
        ray_backlog.need_regeneration = false;
    }

    RayState ray_state;
    ray_state.currently_tracing = false;
    ray_state.has_ray_result = false;
    ray_state.result_i = u32vec2(0, 0);

#if 1

    ray_state.result_i = get_result_i(gl_GlobalInvocationID.x);
    write_ray_result(ray_state);
    return;

#else

    while (true) {
        if (subgroupAny(!ray_state.currently_tracing)) {
            if (!ray_state.currently_tracing) {
                if (ray_state.has_ray_result) {
                    ray_state.has_ray_result = false;
                    write_ray_result(ray_state);
                }
                // local offset in the backlog.
                i32 ray_backlog_index = atomicAdd(ray_backlog.size, -1);
                if (ray_backlog_index > 0) {
                    // I got an item in the current backlog
                    raytrace_begin(ray_backlog.ray_setup_infos[ray_backlog_index], ray_state);
                    ray_state.currently_tracing = true;
                }
                if (ray_backlog_index <= 0) {
                    // I did NOT get an item in the current backlog
                    // We need to get a new backlog, and mark it to be reinitialized later.
                    if (subgroupElect()) {
                        // One thread aquires new ray index for the back log and marks it for reinitialization.
                        ray_backlog.need_regeneration = true;
                    }
                }
            }
            // Flush the shared memory (not used in most compilers)
            subgroupMemoryBarrier();
            if (ray_backlog.need_regeneration) {
                bool backlog_acquire_success = false;
                // reinitialize backlog.
                // We do it here so ALL threads in the warp work on generating the backlog with no divergence and full utilization.
                if (subgroupElect()) {
                    // We now set the backlog metadata, which can be done by one thread alone.
                    ray_backlog.need_regeneration = false;
                    ray_backlog.size = clamp(ray_backlog.start_index, 0, BACKLOG_SIZE);
                    ray_backlog.start_index = atomicAdd(GLOBALS.ray_count, -BACKLOG_SIZE);
                    // If therea re no rays left the backlog size will be empty.
                    backlog_acquire_success = ray_backlog.size > 0;
                }
                subgroupMemoryBarrier();
                if (backlog_acquire_success) {
                    // Reinit the ray setup infos.
                    for (u32 i = 0; i < ray_backlog.size; i += WARP_SIZE) {
                        u32 ray_backlog_index = i + gl_LocalInvocationID.x;
                        if (ray_backlog_index < ray_backlog.size) {
                            ray_backlog.ray_setup_infos[ray_backlog_index] = raytrace_setup(ray_backlog_index);
                        }
                    }
                    subgroupMemoryBarrier();
                    // All the threads that didnt get a new ray previously, now get their new ray out of the backlog.
                    if (!ray_state.currently_tracing) {
                        i32 ray_backlog_index = atomicAdd(ray_backlog.size, -1);
                        raytrace_begin(ray_backlog.ray_setup_infos[ray_backlog_index], ray_state);
                        ray_state.currently_tracing = true;
                    }
                }
            }
        }

        if (subgroupAll(!ray_state.currently_tracing)) {
            break;
        }

        for (u32 i = 0; (i < STEPS_UNTIL_REORDER) && subgroupAny(ray_state.currently_tracing); ++i) {
            if (ray_state.currently_tracing) {
                raytrace_body(ray_state);
            }
        }
    }

#endif
}
