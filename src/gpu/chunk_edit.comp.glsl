#include <shared/shared.inl>

#include <utils/math.glsl>
#include <utils/voxels.glsl>
#include <utils/noise.glsl>

u32vec3 chunk_n;
u32 temp_chunk_index;
u32vec3 chunk_i;
u32 chunk_index;
daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr;
daxa_BufferPtr(VoxelLeafChunk) voxel_chunk_ptr;
u32vec3 inchunk_voxel_i;
u32vec3 voxel_i;
f32vec3 voxel_pos;
BrushInput brush_input;

b32 mandelbulb(in f32vec3 c, in out f32vec3 color) {
    f32vec3 z = c;
    u32 i = 0;
    const f32 n = 8 + floor(good_rand(brush_input.pos) * 5);
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

void brushgen_world(in out f32vec3 col, in out u32 id) {
#if 0
    f32vec3 mandelbulb_color;
    if (mandelbulb((voxel_pos / (CHUNK_SIZE / VOXEL_SCL * 4) * 2 - 1) * 1, mandelbulb_color)) {
        col = f32vec3(0.02);
        id = 1;
    }
#elif 0

    id = 1;
    col = f32vec3(0.5, 0.1, 0.8);

#elif GEN_MODEL

    u32 packed_col_data = sample_gvox_palette_voxel(gvox_model, voxel_i, 0);
    id = sample_gvox_palette_voxel(gvox_model, voxel_i, 0);
    // id = packed_col_data >> 0x18;
    // u32 packed_emi_data = sample_gvox_palette_voxel(gvox_model, voxel_i, 2);
    col = uint_to_float4(packed_col_data).rgb;
    // if (id != 0) {
    //     id = 2;
    // }

#elif 1
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
#elif 0
    u32vec3 voxel_i = u32vec3(voxel_pos / (CHUNK_SIZE / VOXEL_SCL));
    if ((voxel_i.x + voxel_i.y + voxel_i.z) % 2 == 1) {
        id = 1;
        col = f32vec3(0.1);
    }
#else
    if (voxel_i.x > 511 || voxel_i.y > 511 || voxel_i.z > 511) {
        return;
    }
    u32 voxel_index = voxel_i.x + voxel_i.y * 512 + voxel_i.z * 512 * 512;
    f32vec3 vec = deref(test_data_buffer[voxel_index]);
    if (fract(length(vec) * 5) > 0.5) {
        id = 1;
        col.rgb = hsv2rgb(f32vec3(length(vec), 1, 1));
    }
    // const f32 scl = f32(1u << (2 + 3));
    // f32 val = 0;
    // val += length(voxel_pos - scl) - scl * 1.1;
    // if (val < 0) {
    //     id = 1;
    //     col = f32vec3(0.1);
    // }
#endif
}

void brushgen_a(in out f32vec3 col, in out u32 id) {
    u32 voxel_data = sample_voxel_chunk(voxel_malloc_global_allocator, voxel_chunk_ptr, inchunk_voxel_i, false);
    f32vec3 prev_col = uint_to_float4(voxel_data).rgb;
    u32 prev_id = voxel_data >> 0x18;

    col = prev_col;
    id = prev_id;

    if (sd_capsule(voxel_pos, brush_input.pos, brush_input.prev_pos, 32.0 / VOXEL_SCL) < 0) {
        col = f32vec3(0, 0, 0);
        id = 0;
    }
}

void brushgen_b(in out f32vec3 col, in out u32 id) {
    u32 voxel_data = sample_voxel_chunk(voxel_malloc_global_allocator, voxel_chunk_ptr, inchunk_voxel_i, false);
    f32vec3 prev_col = uint_to_float4(voxel_data).rgb;
    u32 prev_id = voxel_data >> 0x18;

    col = prev_col;
    id = prev_id;

    // f32vec3 mandelbulb_color;
    // if (mandelbulb(((voxel_pos-brush_input.pos + CHUNK_SIZE / VOXEL_SCL * 8 / 2) / (CHUNK_SIZE / VOXEL_SCL * 8) * 2 - 1) * 1, mandelbulb_color)) {
    //     col = floor(mandelbulb_color * 10) / 10;
    //     id = 1;
    // }

    if (sd_capsule(voxel_pos, brush_input.pos, brush_input.prev_pos, 32.0 / VOXEL_SCL) < 0) {
        // f32 val = noise(voxel_pos) + (rand() - 0.5) * 1.2;
        // if (val > 0.3) {
        //     col = f32vec3(0.99, 0.03, 0.01);
        // } else if (val > -0.3) {
        //     col = f32vec3(0.91, 0.05, 0.01);
        // } else {
        //     col = f32vec3(0.91, 0.15, 0.01);
        // }
        // col = f32vec3(rand(), rand(), rand());
        // col = f32vec3(floor(rand() * 4.0) / 4.0, floor(rand() * 4.0) / 4.0, floor(rand() * 4.0) / 4.0);
        col = f32vec3(0.1, 0.3, 0.8);
        id = 1;
    }
}

void brushgen_particles(in out f32vec3 col, in out u32 id) {
    for (u32 particle_i = 0; particle_i < deref(globals).voxel_particles_state.place_count; ++particle_i) {
        u32 sim_index = deref(placed_voxel_particles[particle_i]);
        SimulatedVoxelParticle self = deref(simulated_voxel_particles[sim_index]);
        if (u32vec3(floor(self.pos * VOXEL_SCL)) == voxel_i) {
            col = f32vec3(0.8, 0.8, 0.8);
            id = 1;
            return;
        }
    }

    u32 voxel_data = sample_voxel_chunk(voxel_malloc_global_allocator, voxel_chunk_ptr, inchunk_voxel_i, false);
    f32vec3 prev_col = uint_to_float4(voxel_data).rgb;
    u32 prev_id = voxel_data >> 0x18;

    col = prev_col;
    id = prev_id;
}

#define SETTINGS deref(settings)
#define VOXEL_WORLD deref(globals).voxel_world
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    chunk_n = u32vec3(1u << SETTINGS.log2_chunks_per_axis);
    temp_chunk_index = gl_GlobalInvocationID.z / CHUNK_SIZE;
    chunk_i = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].i;
    brush_input = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].brush_input;
    chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;
    temp_voxel_chunk_ptr = temp_voxel_chunks + temp_chunk_index;
    voxel_chunk_ptr = voxel_chunks + chunk_index;
    inchunk_voxel_i = gl_GlobalInvocationID.xyz - u32vec3(0, 0, temp_chunk_index * CHUNK_SIZE);
    voxel_i = chunk_i * CHUNK_SIZE + inchunk_voxel_i;
    voxel_pos = f32vec3(voxel_i) / VOXEL_SCL;

    rand_seed(voxel_i.x + voxel_i.y * 1000 + voxel_i.z * 1000 * 1000);

    f32vec3 col = f32vec3(0.0);
    u32 id = 0;

    u32 chunk_flags = deref(voxel_chunk_ptr).flags;

    if ((chunk_flags & CHUNK_FLAGS_WORLD_BRUSH) != 0) {
        brushgen_world(col, id);
    }
    if ((chunk_flags & CHUNK_FLAGS_USER_BRUSH_A) != 0) {
        brushgen_a(col, id);
    }
    if ((chunk_flags & CHUNK_FLAGS_USER_BRUSH_B) != 0) {
        brushgen_b(col, id);
    }
    if ((chunk_flags & CHUNK_FLAGS_PARTICLE_BRUSH) != 0) {
        brushgen_particles(col, id);
    }

    TempVoxel result;
    result.col_and_id = float4_to_uint(f32vec4(col, 0.0)) | (id << 0x18);
    deref(temp_voxel_chunk_ptr).voxels[inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE] = result;
}
#undef VOXEL_WORLD
#undef SETTINGS
