#include <shared/shared.inl>

DAXA_USE_PUSH_CONSTANT(RaytraceCompPush)

struct RayState {
    u32 result_index;
    u32vec2 result_i;
};

struct RaySetupInfo {
    f32vec3 ray_dir;
    f32vec3 delta;
    f32vec3 t_start;
};

RaySetupInfo raytrace_setup() {
    RaySetupInfo result;
    f32vec2 uv = 
    Ray ray = create_view_ray(uv);

    const u32 max_steps = BRUSH_BLOCK_NX + BRUSH_BLOCK_NY + BRUSH_BLOCK_NZ;
    f32vec3 delta = f32vec3(
        ray.nrm.x == 0.0 ? 3.0 * max_steps : abs(ray.inv_nrm.x),
        ray.nrm.y == 0.0 ? 3.0 * max_steps : abs(ray.inv_nrm.y),
        ray.nrm.z == 0.0 ? 3.0 * max_steps : abs(ray.inv_nrm.z));
    u32 lod = sample_brush_lod(ray.o, chunk_index);
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

}

void raytrace_begin(in out RayState ray_state) {
}

void raytrace_end() {
    // DDA end
    result.dist = t_curr;
    switch (side) {
    case 0: result.nrm = f32vec3(ray.nrm.x < 0 ? 1 : -1, 0, 0); break;
    case 1: result.nrm = f32vec3(0, ray.nrm.y < 0 ? 1 : -1, 0); break;
    case 2: result.nrm = f32vec3(0, 0, ray.nrm.z < 0 ? 1 : -1); break;
    }
    return result;

    result.dist += prev_dist;
}

void raytrace_body(in RaySetupInfo ray_begin_info, in out RayState ray_state) {
    ray_state.current_pos = ray_state.ray.o + ray_state.ray.nrm * ray_state.t_curr;
    if (inside(ray_state.current_pos + ray_state.ray.nrm * 0.001, VOXEL_WORLD.box) == false) {
        ray_state.outside_bounds = true;
        ray_state.result.hit = false;
        break;
    }
    ray_state.lod = sample_lod(ray_state.current_pos, ray_state.chunk_index);
    if (ray_state.lod == 0) {
        ray_state.result.hit = true;
        if (ray_state.t_next.x < ray_state.t_next.y) {
            if (ray_state.t_next.x < ray_state.t_next.z) {
                ray_state.side = 0;
            } else {
                ray_state.side = 2;
            }
        } else {
            if (t_next.y < t_next.z) {
                ray_state.side = 1;
            } else {
                ray_state.side = 2;
            }
        }
        break;
    }
    ray_state.cell_size = f32(1l << (ray_state.lod - 1)) / VOXEL_SCL;
    ray_state.t_next = (0.5 + sign(ray_state.ray.nrm) * (0.5 - fract(ray_state.current_pos / ray_state.cell_size))) * ray_state.cell_size * ray_state.delta;
    ray_state.t_curr += (min(min(ray_state.t_next.x, ray_state.t_next.y), ray_state.t_next.z) + 0.001 / 8);
}

u32vec2 get_result_i(u32 result_index) {
    u32vec2 result;
    result.y = result_index / INPUT.frame_dim.x;
    result.x = result_index - result.y * INPUT.frame_dim.x;
    return result;
}

void write_ray_result(RayState ray_state) {
    // TODO: IMPLEMENT write out
}

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
    u32 need_regeneration;
    RaySetupInfo ray_setup_infos[BACKLOG_SIZE];
};

shared RayBacklog ray_backlog;

RayState set_ray_state(i32 ray_backlog_index) {
    RayState ray_state;
    ray_state.result_index = ray_backlog.start_index - ray_backlog_index;
    // Copy over ray info into thread registers.
    ray_state.result_i = get_result_i(ray_state.result_index);
    raytrace_begin(ray_state);
    return ray_state;
}

layout(local_size_x = WARP_SIZE, local_size_y = 1, local_size_z = 1) in;
void main() {
    b32 currently_tracing = false;
    b32 has_ray_result = false;

    if (subgroupElect()) {
        ray_backlog.start_index = atomicAdd(GLOBALS.ray_count, -64);
        ray_backlog.size = min(ray_backlog.start_index, 64);
        ray_backlog.need_regeneration = false;
    }

    RayState ray_state;

    while (true) {
        if (subgroupAny(!currently_tracing)) {
            if (!currently_tracing) {
                if (has_ray_result) {
                    has_ray_result = false;
                    write_ray_result(ray_state);
                }
                // local offset in the backlog.
                i32 ray_backlog_index = atomicAdd(ray_backlog.size, -1);
                if (ray_backlog_index > 0) {
                    // I got an item in the current backlog
                    ray_state = set_ray_state(ray_backlog_index);
                    currently_tracing = true;
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
            subgroupSharedMemoryBarrier();
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
                subgroupSharedMemoryBarrier();
                if (backlog_acquire_success) {
                    // Reinit the ray setup infos.
                    for (u32 i = 0; i < ray_backlog.size; i += WARP_SIZE) {
                        if (i + gl_LocalInvocationID.x < ray_backlog.size) {
                            ray_backlog.ray_setup_infos[i + gl_LocalInvocationID.x] = raytrace_begin();
                        }
                    }
                    subgroupSharedMemoryBarrier();
                    // All the threads that didnt get a new ray previously, now get their new ray out of the backlog.
                    if (!currently_tracing) {
                        i32 ray_backlog_index = atomicAdd(ray_backlog.size, -1);
                        ray_state = set_ray_state(ray_backlog_index);
                        currently_tracing = true;
                    }
                }
            }
        }

        if (subgroupAll(!currently_tracing)) {
            break;
        }

        for (u32 i = 0; (i < STEPS_UNTIL_REORDER) && subgroupAny(currently_tracing); ++i) {
            if (currently_tracing) {
                raytrace_body(ray_state, currently_tracing);
            }
        }
    }
}
