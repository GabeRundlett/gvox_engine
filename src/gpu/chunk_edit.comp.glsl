#include <shared/shared.inl>

#include <utils/math.glsl>
#include <utils/voxels.glsl>
#include <utils/noise.glsl>

DAXA_USE_PUSH_CONSTANT(ChunkEditComputePush)

#define SIMPLE 0
#define DEBUG 1
#define TERRAIN 2

#define SCENE TERRAIN

f32 terrain_noise(f32vec3 p) {
#if SCENE == SIMPLE
    return fract((p.x + p.y + p.z) * 0.05) - 0.5;
#elif SCENE == DEBUG
    p = f32vec3(2, 2, 180) - p;
    f32 dist = MAX_SD;
    dist = min(dist, -p.z);
    dist = min(dist, length(p + f32vec3(0, 0, 1)) - 1);
    return dist;
#elif SCENE == TERRAIN
    FractalNoiseConfig noise_conf = FractalNoiseConfig(
        /* .amplitude   = */ 1.0,
        /* .persistance = */ 0.2,
        /* .scale       = */ 0.002,
        /* .lacunarity  = */ 4.5,
        /* .octaves     = */ 6);
    f32 val = fractal_noise(p, noise_conf);
    val += p.z * 0.003 - 1;
    return val;
#endif
    return 0;
}

f32vec3 terrain_nrm(f32vec3 pos) {
    f32vec3 n = f32vec3(0.0);
    for (u32 i = 0; i < 4; ++i) {
        f32vec3 e = 0.5773 * (2.0 * f32vec3((((i + 3) >> 1) & 1), ((i >> 1) & 1), (i & 1)) - 1.0);
        n += e * terrain_noise(pos + 1.0 / VOXEL_SCL * e);
    }
    return normalize(n);
}

#define SETTINGS deref(daxa_push_constant.gpu_settings)
#define INPUT deref(daxa_push_constant.gpu_input)
#define VOXEL_WORLD deref(daxa_push_constant.gpu_globals).voxel_world
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    u32vec3 chunk_n;
    chunk_n.x = 1u << SETTINGS.log2_chunks_per_axis;
    chunk_n.y = chunk_n.x;
    chunk_n.z = chunk_n.x;
    u32 temp_chunk_index = gl_GlobalInvocationID.z / CHUNK_SIZE;
    u32vec3 chunk_i = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].i;
    daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr = daxa_push_constant.temp_voxel_chunks + temp_chunk_index;
    u32vec3 inchunk_voxel_i = gl_GlobalInvocationID.xyz - u32vec3(0, 0, temp_chunk_index * CHUNK_SIZE);
    u32vec3 voxel_i = chunk_i * CHUNK_SIZE + inchunk_voxel_i;

    f32vec3 col = f32vec3(0.0);
    u32 id = 0;

    // f32vec3 voxel_pos = f32vec3(voxel_i) / VOXEL_SCL + f32vec3(1700, 1600, 150);
    f32vec3 voxel_pos = f32vec3(voxel_i) / VOXEL_SCL + f32vec3(1700, 1600, 173.5);

    f32 val = terrain_noise(voxel_pos);
    f32vec3 nrm = terrain_nrm(voxel_pos);
    f32 upwards = dot(nrm, f32vec3(0, 0, 1));

    if (val < 0) {
        id = 1;
#if SCENE == SIMPLE
        col = f32vec3(1, 0, 1);
#elif SCENE == DEBUG
        col = f32vec3(0.1);
#elif SCENE == TERRAIN
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
#else
        col = f32vec3(1, 1, 0);
#endif
    }

    TempVoxel result;
    result.col_and_id = float4_to_uint(f32vec4(col, 0.0)) | (id << 0x18);
    deref(temp_voxel_chunk_ptr).voxels[inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE] = result;
}
#undef VOXEL_WORLD
#undef INPUT
#undef SETTINGS
