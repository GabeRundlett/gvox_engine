#include <shared/shared.inl>

#include <utils/math.glsl>
#include <utils/voxels.glsl>
#include <utils/noise.glsl>

DAXA_USE_PUSH_CONSTANT(ChunkEditComputePush)

#define SIMPLE 0

f32 terrain_noise(f32vec3 p) {
#if SIMPLE
    return fract((p.x + p.y + p.z) * 0.05) - 0.5;
#else
    FractalNoiseConfig noise_conf = FractalNoiseConfig(
        /* .amplitude   = */ 1.0,
        /* .persistance = */ 0.15,
        /* .scale       = */ 0.05,
        /* .lacunarity  = */ 6.8,
        /* .octaves     = */ 4);
    f32 val = fractal_noise(p, noise_conf);
    val -= p.z * 0.10 - 8;
    return val;
#endif
}

f32vec3 terrain_nrm(f32vec3 pos) {
    f32vec3 n = f32vec3(0.0);
    for (u32 i = 0; i < 4; ++i) {
        f32vec3 e = 0.5773 * (2.0 * f32vec3((((i + 3) >> 1) & 1), ((i >> 1) & 1), (i & 1)) - 1.0);
        n += e * terrain_noise(pos + 1.0 / VOXEL_SCL * e);
    }
    return -normalize(n);
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
    u32vec3 chunk_i = VOXEL_WORLD.chunk_update_is[temp_chunk_index];
    daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr = daxa_push_constant.temp_voxel_chunks + temp_chunk_index;
    u32vec3 inchunk_voxel_i = gl_GlobalInvocationID.xyz - u32vec3(0, 0, temp_chunk_index * CHUNK_SIZE);
    u32vec3 voxel_i = chunk_i * CHUNK_SIZE + inchunk_voxel_i;

    f32vec3 col = f32vec3(0.0);
    u32 id = 0;

    f32vec3 voxel_pos = f32vec3(voxel_i) / (CHUNK_SIZE / VOXEL_SCL);

    f32 val = terrain_noise(voxel_pos);

    f32 v0 = terrain_noise(voxel_pos + f32vec3(+1, +0, +0) / VOXEL_SCL);
    f32 v1 = terrain_noise(voxel_pos + f32vec3(+0, +1, +0) / VOXEL_SCL);
    f32 v2 = terrain_noise(voxel_pos + f32vec3(+0, +0, +1) / VOXEL_SCL);
    f32 v3 = terrain_noise(voxel_pos + f32vec3(-1, +0, +0) / VOXEL_SCL);
    f32 v4 = terrain_noise(voxel_pos + f32vec3(+0, -1, +0) / VOXEL_SCL);
    f32 v5 = terrain_noise(voxel_pos + f32vec3(+0, +0, -1) / VOXEL_SCL);

    bool is_exposed =
        !(v0 > 0 &&
          v1 > 0 &&
          v2 > 0 &&
          v3 > 0 &&
          v4 > 0 &&
          v5 > 0);

    f32vec3 nrm = terrain_nrm(voxel_pos);

    f32 upwards = dot(nrm, f32vec3(0, 0, 1));

    if (val > 0) {
#if SIMPLE
        id = 1;
        col = f32vec3(1, 0, 1);
#else
        if (is_exposed && val < 0.05 && upwards > 0.75) {
            id = 1;
            col = f32vec3(0.053, 0.2, 0.026);
            // if (good_rand(val) < 0.5) {
            //     col *= 0.9;
            // }
        } else if (val < 0.1 && upwards > 0.6) { //  + good_rand(val) * 0.2
            id = 2;
            col = f32vec3(0.08, 0.05, 0.03);
            // if (good_rand(val) < val) {
            //     col.r *= 0.8;
            //     col.g *= 0.9;
            //     col.b *= 0.8;
            // }
        } else {
            id = 3;
            col = f32vec3(0.03, 0.03, 0.03); // * (0.6 - floor((-sin(val * 2 + 3.5) * 0.5 + 0.5 + good_rand(val) * 0.2) * 4) / 50);
        }
#endif
    }

    TempVoxel result;
    result.col_and_id = float4_to_uint(f32vec4(col, 0.0)) | (id << 0x18);
    deref(temp_voxel_chunk_ptr).voxels[inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE] = result;
}
#undef VOXEL_WORLD
#undef INPUT
#undef SETTINGS
