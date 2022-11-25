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
    f32vec3 t_next;
    f32vec3 current_pos;
    f32 t_curr;
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
#define CACHE_SIZE (WARP_SIZE * 2)
#define BATCH_SIZE (CACHE_SIZE * 2)
#define STEPS_UNTIL_REORDER (128)

// We keep a cache of rays.
// This cache contains a list of prepared rays that can immediately start tracing.
// Running threads in the warp can just grab a new ray from this list when they are dont tracing their current ray.
// Having the ray prepreation work done non-divergently is important as the rays can potentially diverge heavily.
// The cache gets emptied one by one until there are no rays left.
// When its empty, 64 new rays are acquired and ALL threads in a warp prepare the 64 new rays.
struct RayBatchSetupCache {
    i32 size;
    i32 start_index;
    b32 need_regeneration;
    RaySetupInfo ray_setup_infos[CACHE_SIZE];
};

struct RayBatch {
    i32 size;
    i32 start_index;
    RayBatchSetupCache setup_cache;
};

shared RayBatch ray_batch;

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

RaySetupInfo raytrace_setup(u32 ray_cache_index) {
    RaySetupInfo result;
    result.result_index = ray_batch.setup_cache.start_index + ray_cache_index;
    result.result_i = get_result_i(result.result_index);
    Ray ray = create_view_ray(result.result_i);
    ray.inv_nrm = 1.0 / ray.nrm;
    const u32 max_steps = WORLD_BLOCK_NX + WORLD_BLOCK_NY + WORLD_BLOCK_NZ;
    result.delta = f32vec3(
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
    if (ray.nrm.x < 0) {
        result.t_start.x = (ray.o.x / cell_size - floor(ray.o.x / cell_size)) * cell_size * result.delta.x;
    } else {
        result.t_start.x = (ceil(ray.o.x / cell_size) - ray.o.x / cell_size) * cell_size * result.delta.x;
    }
    if (ray.nrm.y < 0) {
        result.t_start.y = (ray.o.y / cell_size - floor(ray.o.y / cell_size)) * cell_size * result.delta.y;
    } else {
        result.t_start.y = (ceil(ray.o.y / cell_size) - ray.o.y / cell_size) * cell_size * result.delta.y;
    }
    if (ray.nrm.z < 0) {
        result.t_start.z = (ray.o.z / cell_size - floor(ray.o.z / cell_size)) * cell_size * result.delta.z;
    } else {
        result.t_start.z = (ceil(ray.o.z / cell_size) - ray.o.z / cell_size) * cell_size * result.delta.z;
    }
    return result;
}

void raytrace_begin(in RaySetupInfo ray_begin_info, in out RayState ray_state) {
    ray_state.result_i = ray_begin_info.result_i;
    // ray_state.has_ray_result = false;
    ray_state.ray = create_view_ray(ray_state.result_i);
    ray_state.t_curr = 0;
    ray_state.delta = ray_begin_info.delta;
}

void raytrace_end(in out RayState ray_state) {
    // switch (ray_state.side) {
    // case 0: ray_state.nrm = f32vec3(ray_state.ray.nrm.x < 0 ? 1 : -1, 0, 0); break;
    // case 1: ray_state.nrm = f32vec3(0, ray_state.ray.nrm.y < 0 ? 1 : -1, 0); break;
    // case 2: ray_state.nrm = f32vec3(0, 0, ray_state.ray.nrm.z < 0 ? 1 : -1); break;
    // }
}

void raytrace_body(in out RayState ray_state) {
    ray_state.current_pos = ray_state.ray.o + ray_state.ray.nrm * ray_state.t_curr;
    if (inside(ray_state.current_pos + ray_state.ray.nrm * 0.001, VOXEL_WORLD.box) == false) {
        ray_state.currently_tracing = false;
        ray_state.has_ray_result = true;
        return;
    }
    u32 chunk_index;
    u32 lod = sample_lod(ray_state.current_pos, chunk_index);
    if (lod == 0) {
        ray_state.currently_tracing = false;
        ray_state.has_ray_result = true;
        if (ray_state.t_next.x < ray_state.t_next.y) {
            if (ray_state.t_next.x < ray_state.t_next.z) {
                ray_state.side = 0;
            } else {
                ray_state.side = 2;
            }
        } else {
            if (ray_state.t_next.y < ray_state.t_next.z) {
                ray_state.side = 1;
            } else {
                ray_state.side = 2;
            }
        }
        return;
    }
    f32 cell_size = f32(1l << (lod - 1)) / VOXEL_SCL;
    ray_state.t_next = (0.5 + sign(ray_state.ray.nrm) * (0.5 - fract(ray_state.current_pos / cell_size))) * cell_size * ray_state.delta;
    ray_state.t_curr += (min(min(ray_state.t_next.x, ray_state.t_next.y), ray_state.t_next.z) + 0.001 / 8);
}

void write_ray_result(RayState ray_state) {
    // raytrace_end(ray_state);
    imageStore(
        get_image(image2D, RT_IMAGE),
        i32vec2(ray_state.result_i),
        f32vec4(f32vec3(fract(ray_state.current_pos)), 1));
}

i32 fast_atomic_decrement(inout i32 a) {
    u32vec4 exec = subgroupBallot(true);
    i32 active_thread_count_left_to_me = i32(subgroupBallotExclusiveBitCount(exec));
    i32 ret = a - active_thread_count_left_to_me;
    i32 active_thread_count = i32(subgroupBallotBitCount(exec));
    a = a - active_thread_count;
    return ret;
}

layout(local_size_x = WARP_SIZE, local_size_y = 1, local_size_z = 1) in;
void main() {
    // One thread from each group will take 64 "jobs" from the global job count
    if (subgroupElect()) {
        ray_batch.start_index = 0;
        ray_batch.size = 0;
        ray_batch.setup_cache.start_index = 0;
        ray_batch.setup_cache.size = 0;
        ray_batch.setup_cache.need_regeneration = true;
    }

    RayState ray_state;
    ray_state.currently_tracing = false;
    ray_state.has_ray_result = false;
    ray_state.result_i = u32vec2(0, 0);

    const i32 TOTAL_RAY_COUNT = i32(INPUT.frame_dim.x * INPUT.frame_dim.y);

    while (true) {
        if (subgroupAny(!ray_state.currently_tracing)) {
            if (!ray_state.currently_tracing) {
                if (ray_state.has_ray_result) {
                    write_ray_result(ray_state);
                    ray_state.has_ray_result = false;
                }
                // local offset in the cache.
                i32 ray_cache_index = CACHE_SIZE - fast_atomic_decrement(ray_batch.setup_cache.size);
                if (ray_cache_index < CACHE_SIZE) {
                    // I got an item in the current cache.
                    raytrace_begin(ray_batch.setup_cache.ray_setup_infos[ray_cache_index], ray_state);
                    ray_state.currently_tracing = true;
                }
                if (ray_cache_index >= CACHE_SIZE) {
                    // I did NOT get an item in the current cache.
                    // We need to get a new cache, and mark it to be reinitialized later.
                    if (subgroupElect()) {
                        // One thread aquires new ray index for the back log and marks it for reinitialization.
                        ray_batch.setup_cache.need_regeneration = true;
                    }
                }
            }
            // Flush the shared memory (not used in most compilers)
            subgroupMemoryBarrierShared();
            if (ray_batch.setup_cache.need_regeneration) {
                // reinitialize batch.
                // We do it here so ALL threads in the warp work on generating the batch with no divergence and full utilization.
                if (subgroupElect()) {
                    // We now set the batch metadata, which can be done by one thread alone.
                    ray_batch.setup_cache.need_regeneration = false;
                    // Get new start index into the ray batch for the start cache:
                    // First see if there are enough rays left in the batch:
                    if (ray_batch.size == 0) {
                        // batch is empty, need to fetch new batch of rays.
                        i32 global_remaining_rays = atomicAdd(GLOBALS.ray_count, -BATCH_SIZE);
                        ray_batch.size = clamp(global_remaining_rays, 0, BATCH_SIZE);
                        ray_batch.start_index = TOTAL_RAY_COUNT - global_remaining_rays;
                    }
                    // Remove rays from the batch and put them into the batch setup cache.
                    ray_batch.setup_cache.size = clamp(ray_batch.size, 0, CACHE_SIZE);
                    ray_batch.size -= ray_batch.setup_cache.size;
                    // Get start index for batch setup cache:
                    ray_batch.setup_cache.start_index = ray_batch.start_index + BATCH_SIZE - ray_batch.size;
                }
                subgroupMemoryBarrierShared();
                // If there are no rays left the batch setup cache size will be empty.
                if (ray_batch.setup_cache.size > 0) {
                    // Reinit the ray setup infos.
                    // The setup cache is generated here, where all threads in the warp are active.
                    // This reduces divergence and improves alu utalization.
                    for (u32 i = 0; i < ray_batch.setup_cache.size; i += WARP_SIZE) {
                        u32 ray_cache_index = i + gl_SubgroupInvocationID.x;
                        if (ray_cache_index < ray_batch.setup_cache.size) {
                            ray_batch.setup_cache.ray_setup_infos[ray_cache_index] = raytrace_setup(ray_cache_index);
                        }
                    }
                    subgroupMemoryBarrierShared();
                    // All the threads that didnt get a new ray previously, now get their new ray out of the setup cache.
                    if (!ray_state.currently_tracing) {
                        i32 ray_cache_index = CACHE_SIZE - fast_atomic_decrement(ray_batch.setup_cache.size);
                        raytrace_begin(ray_batch.setup_cache.ray_setup_infos[ray_cache_index], ray_state);
                        ray_state.currently_tracing = true;
                    }
                }
            }
        }

        if (subgroupAll(!ray_state.currently_tracing))
            break;

        [[unroll]]
        for (u32 i = 0; (i < STEPS_UNTIL_REORDER) && subgroupAny(ray_state.currently_tracing); ++i) {
            if (!ray_state.currently_tracing)
                break;
            raytrace_body(ray_state);
        }
    }
}
