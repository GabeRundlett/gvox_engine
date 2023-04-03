#include <shared/shared.inl>

#include <utils/math.glsl>
#include <utils/voxels.glsl>
#include <utils/noise.glsl>

DAXA_USE_PUSH_CONSTANT(ChunkEditComputePush)

#define INPUT deref(daxa_push_constant.gpu_input)
b32 mandelbulb(in f32vec3 c, in out f32vec3 color) {
    f32vec3 z = c;
    u32 i = 0;
    const f32 n = 8 + sin(INPUT.time) * 4;
    const u32 MAX_ITER = 4;
    f32 m = dot(z, z);
    f32vec4 trap = f32vec4(abs(z), m);
    for (; i < MAX_ITER; ++i) {
        f32 r = length(z);
        f32 p = atan(z.y / z.x);
        f32 t = acos(z.z / r);
        z = f32vec3(
            sin(n * t) * cos(n * p),
            sin(n * t) * sin(n * p),
            cos(n * t));
        z = z * pow(r, n) + c;
        trap = min(trap, f32vec4(abs(z), m));
        m = dot(z, z);
        if (m > 256.0)
            break;
    }
    color = f32vec3(m, trap.yz) * trap.w;
    return i == MAX_ITER;
}
#undef INPUT

f32 terrain_noise(f32vec3 p) {
    FractalNoiseConfig noise_conf = FractalNoiseConfig(
        /* .amplitude   = */ 1.0,
        /* .persistance = */ 0.2,
        /* .scale       = */ 0.002,
        /* .lacunarity  = */ 4.5,
        /* .octaves     = */ 6);
    f32 val = fractal_noise(p, noise_conf);
    val += p.z * 0.003 - 1;
    return val;
}

f32vec3 terrain_nrm(f32vec3 pos) {
    f32vec3 n = f32vec3(0.0);
    for (u32 i = 0; i < 4; ++i) {
        f32vec3 e = 0.5773 * (2.0 * f32vec3((((i + 3) >> 1) & 1), ((i >> 1) & 1), (i & 1)) - 1.0);
        n += e * terrain_noise(pos + 1.0 / VOXEL_SCL * e);
    }
    return normalize(n);
}

u32vec3 chunk_n;
u32 temp_chunk_index;
u32vec3 chunk_i;
u32 chunk_index;
daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr;
daxa_BufferPtr(VoxelChunk) voxel_chunk_ptr;
u32vec3 inchunk_voxel_i;
u32vec3 voxel_i;
f32vec3 voxel_pos;

void brushgen_world(in out f32vec3 col, in out u32 id) {
#if 1
    f32vec3 mandelbulb_color;
    if (mandelbulb((voxel_pos / (CHUNK_SIZE / VOXEL_SCL * 4) * 2 - 1) * 1, mandelbulb_color)) {
        col = f32vec3(0.98);
        id = 1;
    }
#else
    voxel_pos += f32vec3(1700, 1600, 150);

    f32 val = terrain_noise(voxel_pos);
    f32vec3 nrm = terrain_nrm(voxel_pos);
    f32 upwards = dot(nrm, f32vec3(0, 0, 1));

    if (val < 0) {
        id = 1;
        f32 r = good_rand(-val);
        if (val > -0.002 && upwards > 0.65) {
            // col = fract(f32vec3(voxel_i) / CHUNK_SIZE);
            col = f32vec3(0.054, 0.22, 0.028);
            if (r < 0.5) {
                col *= 0.7;
            }
        } else if (val > -0.05 && upwards > 0.5) {
            col = f32vec3(0.08, 0.05, 0.03);
            if (r < 0.5) {
                col.r *= 0.5;
                col.g *= 0.5;
                col.b *= 0.5;
            } else if (r < 0.52) {
                col.r *= 1.5;
                col.g *= 1.5;
                col.b *= 1.5;
            }
        } else if (val < -0.01 && val > -0.07 && upwards > 0.2) {
            col = f32vec3(0.09, 0.08, 0.07);
            if (r < 0.5) {
                col.r *= 0.75;
                col.g *= 0.75;
                col.b *= 0.75;
            }
        } else {
            col = f32vec3(0.08, 0.08, 0.07);
        }
    }
#endif
}

void brushgen_a(in out f32vec3 col, in out u32 id) {
    u32 voxel_data = sample_voxel_chunk(daxa_push_constant.voxel_malloc_global_allocator, voxel_chunk_ptr, inchunk_voxel_i, false);
    f32vec3 prev_col = uint_to_float4(voxel_data).rgb;
    u32 prev_id = voxel_data >> 0x18;

    col = prev_col;
    id = prev_id;

    if (sd_capsule(voxel_pos, deref(daxa_push_constant.gpu_globals).brush_state.pos, deref(daxa_push_constant.gpu_globals).brush_state.prev_pos, 2) < 0) {
        col = f32vec3(0, 0, 0);
        id = 0;
    }
}

void brushgen_b(in out f32vec3 col, in out u32 id) {
    u32 voxel_data = sample_voxel_chunk(daxa_push_constant.voxel_malloc_global_allocator, voxel_chunk_ptr, inchunk_voxel_i, false);
    f32vec3 prev_col = uint_to_float4(voxel_data).rgb;
    u32 prev_id = voxel_data >> 0x18;

    col = prev_col;
    id = prev_id;

    if (sd_capsule(voxel_pos, deref(daxa_push_constant.gpu_globals).brush_state.pos, deref(daxa_push_constant.gpu_globals).brush_state.prev_pos, 0.5) < 0) {
        // f32 val = noise(voxel_pos);
        // if (val > 0.2) {
        //     col = f32vec3(0.90, 0.01, 0.01);
        // } else if (val > -0.2) {
        //     col = f32vec3(0.01, 0.90, 0.01);
        // } else {
        //     col = f32vec3(0.01, 0.01, 0.90);
        // }
        col = f32vec3(0.01, 0.91, 0.91);
        id = 2;
    }
}

#define SETTINGS deref(daxa_push_constant.gpu_settings)
#define INPUT deref(daxa_push_constant.gpu_input)
#define VOXEL_WORLD deref(daxa_push_constant.gpu_globals).voxel_world
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    chunk_n = u32vec3(1u << SETTINGS.log2_chunks_per_axis);
    temp_chunk_index = gl_GlobalInvocationID.z / CHUNK_SIZE;
    chunk_i = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].i;
    chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;
    temp_voxel_chunk_ptr = daxa_push_constant.temp_voxel_chunks + temp_chunk_index;
    voxel_chunk_ptr = daxa_push_constant.voxel_chunks + chunk_index;
    inchunk_voxel_i = gl_GlobalInvocationID.xyz - u32vec3(0, 0, temp_chunk_index * CHUNK_SIZE);
    voxel_i = chunk_i * CHUNK_SIZE + inchunk_voxel_i;
    voxel_pos = f32vec3(voxel_i) / VOXEL_SCL;

    rand_seed(voxel_i.x + voxel_i.y * 1000 + voxel_i.z * 1000 * 1000);

    f32vec3 col = f32vec3(0.0);
    u32 id = 0;

    switch (deref(voxel_chunk_ptr).edit_stage) {
    case CHUNK_STAGE_WORLD_BRUSH: brushgen_world(col, id); break;
    case CHUNK_STAGE_USER_BRUSH_A: brushgen_a(col, id); break;
    case CHUNK_STAGE_USER_BRUSH_B: brushgen_b(col, id); break;
    }

    TempVoxel result;
    result.col_and_id = float4_to_uint(f32vec4(col, 0.0)) | (id << 0x18);
    deref(temp_voxel_chunk_ptr).voxels[inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE] = result;
}
#undef VOXEL_WORLD
#undef INPUT
#undef SETTINGS
