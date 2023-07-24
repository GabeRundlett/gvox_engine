#include <shared/shared.inl>

#include <utils/math.glsl>
#include <utils/voxels.glsl>
#include <utils/noise.glsl>

u32vec3 chunk_n;
u32 temp_chunk_index;
i32vec3 chunk_i;
u32 chunk_index;
daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr;
daxa_BufferPtr(VoxelLeafChunk) voxel_chunk_ptr;
u32vec3 inchunk_voxel_i;
i32vec3 voxel_i;
f32vec3 voxel_pos;
BrushInput brush_input;

DAXA_DECL_PUSH_CONSTANT(ChunkEditComputePush, push)

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

f32vec4 terrain_noise(f32vec3 p) {
    FractalNoiseConfig noise_conf = FractalNoiseConfig(
        /* .amplitude   = */ 1.0,
        /* .persistance = */ 0.2,
        /* .scale       = */ 0.005,
        /* .lacunarity  = */ 4.5,
        /* .octaves     = */ 6);
    f32vec4 val = fractal_noise(value_noise_texture, push.value_noise_sampler, p, noise_conf);
    val.x += p.z * 0.003 - 1.0;
    val.yzw = normalize(val.yzw + vec3(0, 0, 0.003));
    // val.x += -0.24;
    return val;
}

void brushgen_world(in out f32vec3 col, in out u32 id) {
#if 0   // Mandelbulb world
    f32vec3 mandelbulb_color;
    if (mandelbulb((voxel_pos / (CHUNK_WORLDSPACE_SIZE * 4) * 2 - 1) * 1, mandelbulb_color)) {
        col = f32vec3(0.02);
        id = 1;
    }
#elif 0 // Solid world
    id = 1;
    col = f32vec3(0.5, 0.1, 0.8);

#elif GEN_MODEL // Model world
    u32 packed_col_data = sample_gvox_palette_voxel(gvox_model, voxel_i, 0);
    id = sample_gvox_palette_voxel(gvox_model, voxel_i, 0);
    // id = packed_col_data >> 0x18;
    // u32 packed_emi_data = sample_gvox_palette_voxel(gvox_model, voxel_i, 2);
    col = uint_rgba8_to_float4(packed_col_data).rgb;
    // if (id != 0) {
    //     id = 2;
    // }

#elif 1 // Terrain world
    voxel_pos += f32vec3(0, 0, 200);

    f32vec4 val4 = terrain_noise(voxel_pos);
    f32 val = val4.x;
    f32vec3 nrm = normalize(val4.yzw); // terrain_nrm(voxel_pos);
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
#elif 1
    voxel_pos += f32vec3(0, 0, 150);
    if (voxel_i.x == 0 || voxel_i.y == 0 || voxel_i.z == 0) {
        f32 val = terrain_noise(voxel_pos);
        id = 1;
        col = f32vec3(val);
    }
#elif 1 // Ball world (each ball is centered on a chunk center)
    if (length(fract(voxel_pos / CHUNK_WORLDSPACE_SIZE) - 0.5) < 0.15) {
        id = 1;
        col = f32vec3(0.1);
    }
#elif 0 // Checker board world
    u32vec3 voxel_i = u32vec3(voxel_pos / CHUNK_WORLDSPACE_SIZE);
    if ((voxel_i.x + voxel_i.y + voxel_i.z) % 2 == 1) {
        id = 1;
        col = f32vec3(0.1);
    }
#endif
}

void brushgen_a(in out f32vec3 col, in out u32 id) {
    u32 voxel_data = sample_voxel_chunk(voxel_malloc_page_allocator, voxel_chunk_ptr, inchunk_voxel_i);
    f32vec3 prev_col = uint_rgba8_to_float4(voxel_data).rgb;
    u32 prev_id = voxel_data >> 0x18;

    col = prev_col;
    id = prev_id;

    if (sd_capsule(voxel_pos, brush_input.pos, brush_input.prev_pos, 32.0 / VOXEL_SCL) < 0) {
        col = f32vec3(0, 0, 0);
        id = 0;
    }
}

void brushgen_b(in out f32vec3 col, in out u32 id) {
    u32 voxel_data = sample_voxel_chunk(voxel_malloc_page_allocator, voxel_chunk_ptr, inchunk_voxel_i);
    f32vec3 prev_col = uint_rgba8_to_float4(voxel_data).rgb;
    u32 prev_id = voxel_data >> 0x18;

    col = prev_col;
    id = prev_id;

    // f32vec3 mandelbulb_color;
    // if (mandelbulb(((voxel_pos-brush_input.pos + CHUNK_WORLDSPACE_SIZE * 8 / 2) / (CHUNK_WORLDSPACE_SIZE * 8) * 2 - 1) * 1, mandelbulb_color)) {
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

    u32 voxel_data = sample_voxel_chunk(voxel_malloc_page_allocator, voxel_chunk_ptr, inchunk_voxel_i);
    f32vec3 prev_col = uint_rgba8_to_float4(voxel_data).rgb;
    u32 prev_id = voxel_data >> 0x18;

    col = prev_col;
    id = prev_id;
}

#define SETTINGS deref(settings)
#define VOXEL_WORLD deref(globals).voxel_world
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    // (const) number of chunks in each axis
    chunk_n = u32vec3(1u << SETTINGS.log2_chunks_per_axis);
    // Index in chunk_update_infos buffer
    temp_chunk_index = gl_GlobalInvocationID.z / CHUNK_SIZE;
    // Chunk 3D index in leaf chunk space (0^3 - 31^3)
    chunk_i = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].i;

    // Here we check whether the chunk update that we're handling is an update
    // for a chunk that has already been submitted. This is a bit inefficient,
    // since we'd hopefully like to queue a separate work item into the queue
    // instead, but this is tricky.
    if (chunk_i == INVALID_CHUNK_I) {
        return;
    }

    // Player chunk offset
    i32vec3 chunk_offset = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].chunk_offset;
    // Brush informations
    brush_input = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].brush_input;
    // Chunk u32 index in voxel_chunks buffer
    chunk_index = calc_chunk_index_from_worldspace(chunk_i, chunk_n);
    // Pointer to the previous chunk
    temp_voxel_chunk_ptr = temp_voxel_chunks + temp_chunk_index;
    // Pointer to the new chunk
    voxel_chunk_ptr = voxel_chunks + chunk_index;
    // Voxel offset in chunk
    inchunk_voxel_i = gl_GlobalInvocationID.xyz - u32vec3(0, 0, temp_chunk_index * CHUNK_SIZE);
    // Voxel 3D position (in voxel buffer)
    voxel_i = chunk_i * CHUNK_SIZE + i32vec3(inchunk_voxel_i);

    // Wrapped chunk index in leaf chunk space (0^3 - 31^3)
    i32vec3 wrapped_chunk_i = imod3(chunk_i - imod3(chunk_offset - i32vec3(chunk_n), i32vec3(chunk_n)), i32vec3(chunk_n));
    // Leaf chunk position in world space
    i32vec3 world_chunk = chunk_offset + wrapped_chunk_i;

    // Voxel position in world space (voxels)
    i32vec3 world_voxel = world_chunk * CHUNK_SIZE + i32vec3(inchunk_voxel_i);
    // Voxel position in world space (meters)
    voxel_pos = f32vec3(world_voxel) / VOXEL_SCL;

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
    // if ((chunk_flags & CHUNK_FLAGS_PARTICLE_BRUSH) != 0) {
    //     brushgen_particles(col, id);
    // }

    TempVoxel result;
    result.col_and_id = float4_to_uint_rgba8(f32vec4(col, 0.0)) | (id << 0x18);
    deref(temp_voxel_chunk_ptr).voxels[inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE] = result;
}
#undef VOXEL_WORLD
#undef SETTINGS
