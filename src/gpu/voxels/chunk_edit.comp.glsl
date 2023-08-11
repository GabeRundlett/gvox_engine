#include <shared/app.inl>

#if PER_CHUNK_COMPUTE

#include <utils/math.glsl>
#include <utils/voxels.glsl>

i32vec3 chunk_n;
i32vec3 chunk_i;
u32 chunk_index;

#define VOXEL_WORLD deref(globals).voxel_world
#define PLAYER deref(globals).player
#define CHUNKS(i) deref(voxel_chunks[i])
#define INDIRECT deref(globals).indirect_dispatch

void try_elect(in out ChunkWorkItem work_item) {
    u32 prev_update_n = atomicAdd(VOXEL_WORLD.chunk_update_n, 1);

    // Check if the work item can be added
    if (prev_update_n < MAX_CHUNK_UPDATES_PER_FRAME) {
        // Set the chunk edit dispatch z axis (64/8, 64/8, 64 x 8 x 8 / 8 = 64 x 8) = (8, 8, 512)
        atomicAdd(INDIRECT.chunk_edit_dispatch.z, CHUNK_SIZE / 8);
        atomicAdd(INDIRECT.subchunk_x2x4_dispatch.z, 1);
        atomicAdd(INDIRECT.subchunk_x8up_dispatch.z, 1);
        u32 prev_flags = atomicOr(CHUNKS(chunk_index).flags, work_item.brush_id);
        // Set the chunk update infos
        VOXEL_WORLD.chunk_update_infos[prev_update_n].i = chunk_i;
        VOXEL_WORLD.chunk_update_infos[prev_update_n].brush_input = work_item.brush_input;
        VOXEL_WORLD.chunk_update_infos[prev_update_n].chunk_offset = work_item.chunk_offset;
    }
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    // (const) number of chunks in each axis
    chunk_n = i32vec3(1 << deref(gpu_input).log2_chunks_per_axis);
    chunk_i = i32vec3(gl_GlobalInvocationID.xyz);
    chunk_index = calc_chunk_index_from_worldspace(chunk_i, chunk_n);

    // Temporary.
    ChunkWorkItem terrain_work_item;
    terrain_work_item.i = i32vec3(0);
    terrain_work_item.chunk_offset = PLAYER.chunk_offset;
    terrain_work_item.brush_id = CHUNK_FLAGS_WORLD_BRUSH;

    if ((CHUNKS(chunk_index).flags & CHUNK_FLAGS_ACCEL_GENERATED) == 0) {
        try_elect(terrain_work_item);
    } else if (PLAYER.chunk_offset != PLAYER.prev_chunk_offset) {
        // invalidate chunks outside the chunk_offset
        i32vec3 diff = clamp(i32vec3(PLAYER.chunk_offset - PLAYER.prev_chunk_offset), -chunk_n, chunk_n);

        i32vec3 start;
        i32vec3 end;

        start.x = diff.x < 0 ? 0 : chunk_n.x - diff.x;
        end.x = diff.x < 0 ? -diff.x : chunk_n.x;

        start.y = diff.y < 0 ? 0 : chunk_n.y - diff.y;
        end.y = diff.y < 0 ? -diff.y : chunk_n.y;

        start.z = diff.z < 0 ? 0 : chunk_n.z - diff.z;
        end.z = diff.z < 0 ? -diff.z : chunk_n.z;

        u32vec3 temp_chunk_i = u32vec3((i32vec3(chunk_i) - deref(globals).player.chunk_offset) % i32vec3(chunk_n));

        if ((temp_chunk_i.x >= start.x && temp_chunk_i.x < end.x) ||
            (temp_chunk_i.y >= start.y && temp_chunk_i.y < end.y) ||
            (temp_chunk_i.z >= start.z && temp_chunk_i.z < end.z)) {
            CHUNKS(chunk_index).flags &= ~CHUNK_FLAGS_ACCEL_GENERATED;
            try_elect(terrain_work_item);
        }
    }
}

#undef INDIRECT
#undef CHUNKS
#undef PLAYER
#undef VOXEL_WORLD

#endif

#if CHUNK_EDIT_COMPUTE

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
i32vec3 world_voxel;
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

f32vec4 terrain_noise(f32vec3 p) {
    FractalNoiseConfig noise_conf = FractalNoiseConfig(
        /* .amplitude   = */ 1.0,
        /* .persistance = */ 0.2,
        /* .scale       = */ 0.005,
        /* .lacunarity  = */ 4.5,
        /* .octaves     = */ 6);
    f32vec4 val = fractal_noise(value_noise_texture, deref(gpu_input).sampler_llr, p, noise_conf);
    val.x += p.z * 0.003 - 1.0;
    val.yzw = normalize(val.yzw + vec3(0, 0, 0.003));
    // val.x += -0.24;
    return val;
}

// Spruce Tree generation code (from main branch)
u32 rand_hash(u32 x) {
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}
u32 rand_hash(u32vec2 v) { return rand_hash(v.x ^ rand_hash(v.y)); }
u32 rand_hash(u32vec3 v) {
    return rand_hash(v.x ^ rand_hash(v.y) ^ rand_hash(v.z));
}
u32 rand_hash(u32vec4 v) {
    return rand_hash(v.x ^ rand_hash(v.y) ^ rand_hash(v.z) ^ rand_hash(v.w));
}
f32 rand_float_construct(u32 m) {
    const u32 ieee_mantissa = 0x007FFFFFu;
    const u32 ieee_one = 0x3F800000u;
    m &= ieee_mantissa;
    m |= ieee_one;
    f32 f = uintBitsToFloat(m);
    return f - 1.0;
}
f32 rand(f32 x) { return rand_float_construct(rand_hash(floatBitsToUint(x))); }
f32 rand(f32vec2 v) { return rand_float_construct(rand_hash(floatBitsToUint(v))); }
f32 rand(f32vec3 v) { return rand_float_construct(rand_hash(floatBitsToUint(v))); }
f32 rand(f32vec4 v) { return rand_float_construct(rand_hash(floatBitsToUint(v))); }

struct TreeSDF {
    f32 wood;
    f32 leaves;
};

void sd_branch(in out TreeSDF val, in f32vec3 p, in f32vec3 origin, in f32vec3 dir, in f32 scl) {
    f32vec3 bp0 = origin;
    f32vec3 bp1 = bp0 + dir;
    val.wood = min(val.wood, sd_capsule(p, bp0, bp1, 0.10));
    val.leaves = min(val.leaves, sd_sphere(p - bp1, 0.15 * scl));
    bp0 = bp1, bp1 = bp0 + dir * 0.5 + f32vec3(0, 0, 0.2);
    val.wood = min(val.wood, sd_capsule(p, bp0, bp1, 0.07));
    val.leaves = min(val.leaves, sd_sphere(p - bp1, 0.15 * scl));
}

TreeSDF sd_spruce_tree(in f32vec3 p, in f32vec3 seed) {
    TreeSDF val = TreeSDF(1e5, 1e5);
    val.wood = min(val.wood, sd_capsule(p, f32vec3(0, 0, 0), f32vec3(0, 0, 4.5), 0.15));
    val.leaves = min(val.leaves, sd_capsule(p, f32vec3(0, 0, 4.5), f32vec3(0, 0, 5.0), 0.15));
    for (u32 i = 0; i < 5; ++i) {
        f32 scl = 1.0 / (1.0 + i * 0.5);
        f32 scl2 = 1.0 / (1.0 + i * 0.1);
        u32 branch_n = 8 - i;
        for (u32 branch_i = 0; branch_i < branch_n; ++branch_i) {
            f32 angle = (1.0 / branch_n * branch_i) * 2.0 * PI + rand(seed + i + 1.0 * branch_i) * 0.5;
            sd_branch(val, p, f32vec3(0, 0, 1.0 + i * 0.8) * 1.0, normalize(f32vec3(cos(angle), sin(angle), +0.0)) * scl, scl2 * 1.5);
        }
    }
    return val;
}

// Random noise
f32vec3 hash33(f32vec3 p3) {
    p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

// Value noise + fbm noise
float hash(in ivec2 p) {
    int n = p.x * 3 + p.y * 113;
    n = (n << 13) ^ n;
    n = n * (n * n * 15731 + 789221) + 1376312589;
    return -1.0 + 2.0 * float(n & 0x0fffffff) / float(0x0fffffff);
}
float value_noise(in vec2 p) {
    ivec2 i = ivec2(floor(p));
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i + ivec2(0, 0)),
                   hash(i + ivec2(1, 0)), u.x),
               mix(hash(i + ivec2(0, 1)),
                   hash(i + ivec2(1, 1)), u.x),
               u.y);
}
f32 fbm2(vec2 uv) {
    f32 f = 0;
    mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
    f = 0.5000 * value_noise(uv);
    uv = m * uv;
    f += 0.2500 * value_noise(uv);
    uv = m * uv;
    return f * .5 + .5;
}

// Forest generation
#define TREE_MARCH_STEPS 4

f32vec3 get_closest_surface(f32vec3 center_cell_world, f32 current_noise, f32 rep, inout f32 scale) {
    f32vec3 offset = hash33(center_cell_world);
    scale = offset.z * .3 + .7;
    center_cell_world.xy += (offset.xy * 2 - 1) * max(0, rep / scale - 5);

    f32 step_size = rep / 2 / TREE_MARCH_STEPS;

    // Above terrain
    if (current_noise > 0) {
        for (u32 i = 0; i < TREE_MARCH_STEPS; i++) {
            center_cell_world.z -= step_size;
            if (terrain_noise(center_cell_world).x < 0)
                return center_cell_world;
        }
    }
    // Inside terrain
    else {
        for (u32 i = 0; i < TREE_MARCH_STEPS; i++) {
            center_cell_world.z += step_size;
            if (terrain_noise(center_cell_world).x > 0)
                return center_cell_world - f32vec3(0, 0, step_size);
        }
    }

    return f32vec3(0);
}

// Color palettes
f32vec3 palette(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d) {
    return a + b * cos(6.28318 * (c * t + d));
}
f32vec3 forest_biome_palette(f32 t) {
    return palette(t + .5, vec3(0.07, 0.22, 0.03), vec3(0.03, 0.05, 0.01), vec3(-1.212, -2.052, 0.058), vec3(1.598, 6.178, 0.380));
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
    u32 packed_col_data = sample_gvox_palette_voxel(gvox_model, world_voxel, 0);
    id = sample_gvox_palette_voxel(gvox_model, world_voxel, 0);
    // id = packed_col_data >> 0x18;
    // u32 packed_emi_data = sample_gvox_palette_voxel(gvox_model, world_voxel, 2);
    col = uint_rgba8_to_f32vec4(packed_col_data).rgb;
    // if (id != 0) {
    //     id = 2;
    // }

#elif 1 // Terrain world
    voxel_pos += f32vec3(0, 0, 280);

    f32vec4 val4 = terrain_noise(voxel_pos);
    f32 val = val4.x;
    f32vec3 nrm = normalize(val4.yzw); // terrain_nrm(voxel_pos);
    f32 upwards = dot(nrm, f32vec3(0, 0, 1));

    // Smooth noise depending on 2d position only
    f32 voxel_noise_xy = fbm2(voxel_pos.xy / 8 / 40);
    // Smooth biome color
    f32vec3 forest_biome_color = forest_biome_palette(voxel_noise_xy * 2 - 1);

    if (val < 0) {
        id = 1;
        f32 r = good_rand(-val);
        if (val > -0.002 && upwards > 0.65) {
            // col = fract(f32vec3(voxel_i) / CHUNK_SIZE);
            col = f32vec3(0.05, 0.26, 0.03);
            if (r < 0.5) {
                col *= 0.7;
            }
            // Mix with biome color
            col = mix(col, forest_biome_color * .75, .3);
        } else if (val > -0.05 && upwards > 0.5) {
            col = f32vec3(0.13, 0.09, 0.05);
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
            col = f32vec3(0.17, 0.15, 0.07);
            if (r < 0.5) {
                col.r *= 0.75;
                col.g *= 0.75;
                col.b *= 0.75;
            }
        } else {
            col = f32vec3(0.11, 0.10, 0.07);
        }
    }
#if ENABLE_TREE_GENERATION
    else {
        // Meters per cell
        f32 rep = 6;

        // Global cell ID
        f32vec3 qid = floor(voxel_pos / rep);
        // Local coordinates in current cell (centered at 0 [-rep/2, rep/2])
        f32vec3 q = mod(voxel_pos, rep) - rep / 2;
        // Current cell's center voxel (world space)
        f32vec3 cell_center_world = qid * rep + rep / 2.;

        // Query terrain noise at current cell's center
        f32vec4 center_noise = terrain_noise(cell_center_world);

        // Optimization: only run for chunks near enough the terrain surface
        bool can_spawn = center_noise.x >= -0.01 * rep / 4 && center_noise.x < 0.03 * rep / 4;

        // Forest density
        f32 forest_noise = fbm2(qid.xy / 10.);
        f32 forest_density = .45;

        if (forest_noise > forest_density)
            can_spawn = false;

        if (can_spawn) {
            // Tree scale
            f32 scale;
            // Try to get the nearest point on the surface below (in the starting cell)
            f32vec3 hitPoint = get_closest_surface(cell_center_world, center_noise.x, rep, scale);

            if (hitPoint == f32vec3(0) && center_noise.x > 0) {
                // If no terrain was found, try again for the bottom cell (upper tree case)
                scale = forest_noise;
                f32vec3 down_neighbor_cell_center_world = cell_center_world - f32vec3(0, 0, rep);
                hitPoint = get_closest_surface(down_neighbor_cell_center_world, terrain_noise(down_neighbor_cell_center_world).x, rep, scale);
            }

            // Debug space repetition boundaries
            // f32 tresh = 1. / 8.;
            // if (abs(abs(q.x)-rep/2.) <= tresh && abs(abs(q.y)-rep/2.) <= tresh ||
            //     abs(abs(q.x)-rep/2.) <= tresh && abs(abs(q.z)-rep/2.) <= tresh ||
            //     abs(abs(q.z)-rep/2.) <= tresh && abs(abs(q.y)-rep/2.) <= tresh) {
            //     id = 1;
            //     col = f32vec3(0,0,0);
            // }

            // Distance to tree
            TreeSDF tree = sd_spruce_tree((voxel_pos - hitPoint) / scale, qid);

            f32vec3 h_cell = vec3(0);  // hash33(qid);
            f32vec3 h_voxel = vec3(0); // hash33(voxel_pos);

            // Colorize tree
            if (tree.wood < 0) {
                id = 1;
                col = f32vec3(.68, .4, .15) * 0.16;
            } else if (tree.leaves < 0) {
                id = 1;
                col = forest_biome_color * 0.5;
            }
        }
    }
#endif
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
    f32vec3 prev_col = uint_rgba8_to_f32vec4(voxel_data).rgb;
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
    f32vec3 prev_col = uint_rgba8_to_f32vec4(voxel_data).rgb;
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
    // for (u32 particle_i = 0; particle_i < deref(globals).voxel_particles_state.place_count; ++particle_i) {
    //     u32 sim_index = deref(placed_voxel_particles[particle_i]);
    //     SimulatedVoxelParticle self = deref(simulated_voxel_particles[sim_index]);
    //     if (u32vec3(floor(self.pos * VOXEL_SCL)) == voxel_i) {
    //         col = f32vec3(0.8, 0.8, 0.8);
    //         id = 1;
    //         return;
    //     }
    // }

    u32 voxel_data = sample_voxel_chunk(voxel_malloc_page_allocator, voxel_chunk_ptr, inchunk_voxel_i);
    f32vec3 prev_col = uint_rgba8_to_f32vec4(voxel_data).rgb;
    u32 prev_id = voxel_data >> 0x18;

    col = prev_col;
    id = prev_id;
}

#define VOXEL_WORLD deref(globals).voxel_world
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    // (const) number of chunks in each axis
    chunk_n = u32vec3(1u << deref(gpu_input).log2_chunks_per_axis);
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
    world_voxel = world_chunk * CHUNK_SIZE + i32vec3(inchunk_voxel_i);
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
    result.col_and_id = f32vec4_to_uint_rgba8(f32vec4(col, 0.0)) | (id << 0x18);
    deref(temp_voxel_chunk_ptr).voxels[inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE] = result;
}
#undef VOXEL_WORLD

#endif

#if CHUNK_OPT_COMPUTE

#include <utils/voxels.glsl>

#define WAVE_SIZE gl_SubgroupSize
#define WAVE_SIZE_MUL (WAVE_SIZE / 32)

u32 sample_temp_voxel_id(daxa_BufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr, in u32vec3 in_chunk_i) {
    u32 in_chunk_index = in_chunk_i.x + in_chunk_i.y * CHUNK_SIZE + in_chunk_i.z * CHUNK_SIZE * CHUNK_SIZE;
    return deref(temp_voxel_chunk_ptr).voxels[in_chunk_index].col_and_id >> 24;
}

// For now, I'm testing with using non-zero as the accel structure, instead of uniformity.
u32 sample_base_voxel_id(daxa_BufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr, in u32vec3 in_chunk_i) {
#if VOXEL_ACCEL_UNIFORMITY
    return sample_temp_voxel_id(temp_voxel_chunk_ptr, in_chunk_i);
#else
    return 0;
#endif
}

#if CHUNK_OPT_STAGE == 0

shared u32 local_x2_copy[4][4];

#define VOXEL_CHUNK deref(voxel_chunk_ptr)
void chunk_opt_x2x4(daxa_BufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr, daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunk_ptr, in u32 chunk_local_workgroup) {
    u32vec2 x2_in_group_location = u32vec2(
        (gl_LocalInvocationID.x >> 5) & 0x3,
        (gl_LocalInvocationID.x >> 7) & 0x3);
    u32vec3 x2_i = u32vec3(
        (gl_LocalInvocationID.x >> 5) & 0x3,
        (gl_LocalInvocationID.x >> 7) & 0x3,
        (gl_LocalInvocationID.x & 0x1F));
    x2_i += 4 * u32vec3(chunk_local_workgroup & 0x7, (chunk_local_workgroup >> 3) & 0x7, 0);
    u32vec3 in_chunk_i = x2_i * 2;
    b32 at_least_one_occluding = false;
    u32 base_id_x1 = sample_base_voxel_id(temp_voxel_chunk_ptr, in_chunk_i);
    for (i32 x = 0; x < 2; ++x)
        for (i32 y = 0; y < 2; ++y)
            for (i32 z = 0; z < 2; ++z) {
                i32vec3 local_i = i32vec3(in_chunk_i) + i32vec3(x, y, z); // in x1 space
                at_least_one_occluding = at_least_one_occluding || (sample_temp_voxel_id(temp_voxel_chunk_ptr, local_i) != base_id_x1);
            }
    u32 result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(x2_i);
    }
    for (i32 i = 0; i < 1 * WAVE_SIZE_MUL; i++) {
        if ((gl_SubgroupInvocationID >> 5) == i) {
            result = subgroupOr(result);
        }
    }
    if ((gl_SubgroupInvocationID & 0x1F /* = %32 */) == 0) {
        u32 index = uniformity_lod_index(2)(x2_i);
        VOXEL_CHUNK.uniformity.lod_x2[index] = result;
        local_x2_copy[x2_in_group_location.x][x2_in_group_location.y] = result;
    }
    subgroupBarrier();
    if (gl_LocalInvocationID.x >= 64) {
        return;
    }
    u32vec3 x4_i = u32vec3(
        (gl_LocalInvocationID.x >> 4) & 0x1,
        (gl_LocalInvocationID.x >> 5) & 0x1,
        gl_LocalInvocationID.x & 0xF);
    x4_i += 2 * u32vec3(chunk_local_workgroup & 0x7, (chunk_local_workgroup >> 3) & 0x7, 0);
    x2_i = x4_i * 2;
    u32 base_id_x2 = sample_base_voxel_id(temp_voxel_chunk_ptr, x2_i * 2);
    at_least_one_occluding = false;
    for (i32 x = 0; x < 2; ++x)
        for (i32 y = 0; y < 2; ++y)
            for (i32 z = 0; z < 2; ++z) {
                i32vec3 local_i = i32vec3(x2_i) + i32vec3(x, y, z); // in x2 space
                u32 mask = uniformity_lod_mask(local_i);
                u32vec2 x2_in_group_index = u32vec2(
                    local_i.x & 0x3,
                    local_i.y & 0x3);
                b32 is_occluding = (local_x2_copy[x2_in_group_index.x][x2_in_group_index.y] & mask) != 0;
                at_least_one_occluding = at_least_one_occluding || is_occluding || (sample_temp_voxel_id(temp_voxel_chunk_ptr, local_i * 2) != base_id_x2);
            }
    result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(x4_i);
    }
    for (i32 i = 0; i < 2 * WAVE_SIZE_MUL; i++) {
        if ((gl_SubgroupInvocationID >> 4) == i) {
            result = subgroupOr(result);
        }
    }
    if ((gl_SubgroupInvocationID & 0xF /* = %16 */) == 0) {
        u32 index = uniformity_lod_index(4)(x4_i);
        VOXEL_CHUNK.uniformity.lod_x4[index] = result;
    }
}
#undef VOXEL_CHUNK

#define VOXEL_WORLD deref(globals).voxel_world
layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
void main() {
    i32vec3 chunk_i = VOXEL_WORLD.chunk_update_infos[gl_WorkGroupID.z].i;
    if (chunk_i == INVALID_CHUNK_I) {
        return;
    }
    u32vec3 chunk_n;
    chunk_n.x = 1u << deref(gpu_input).log2_chunks_per_axis;
    chunk_n.y = chunk_n.x;
    chunk_n.z = chunk_n.x;
    u32 chunk_index = calc_chunk_index_from_worldspace(chunk_i, chunk_n);
    daxa_BufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr = temp_voxel_chunks + gl_WorkGroupID.z;
    daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunk_ptr = voxel_chunks + chunk_index;
    chunk_opt_x2x4(temp_voxel_chunk_ptr, voxel_chunk_ptr, gl_WorkGroupID.y);
}
#undef VOXEL_WORLD

#endif

#if CHUNK_OPT_STAGE == 1

shared u32 local_x8_copy[64];
shared u32 local_x16_copy[16];

#define VOXEL_CHUNK deref(voxel_chunk_ptr)
void chunk_opt_x8up(daxa_BufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr, daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunk_ptr) {
    u32vec3 x8_i = u32vec3(
        (gl_LocalInvocationID.x >> 3) & 0x7,
        (gl_LocalInvocationID.x >> 6) & 0x7,
        gl_LocalInvocationID.x & 0x7);
    u32vec3 x4_i = x8_i * 2;

    u32 base_id_x4 = sample_base_voxel_id(temp_voxel_chunk_ptr, x4_i * 4);

    b32 at_least_one_occluding = false;
    for (i32 x = 0; x < 2; ++x)
        for (i32 y = 0; y < 2; ++y)
            for (i32 z = 0; z < 2; ++z) {
                i32vec3 local_i = i32vec3(x4_i) + i32vec3(x, y, z); // x4 space
                u32 index = uniformity_lod_index(4)(local_i);
                u32 mask = uniformity_lod_mask(local_i);
                b32 occluding = (VOXEL_CHUNK.uniformity.lod_x4[index] & mask) != 0;
                at_least_one_occluding = at_least_one_occluding || occluding || (sample_temp_voxel_id(temp_voxel_chunk_ptr, local_i * 4) != base_id_x4);
            }

    u32 result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(x8_i);
    }
    for (i32 i = 0; i < 4 * WAVE_SIZE_MUL; i++) {
        if ((gl_SubgroupInvocationID >> 3) == i) {
            result = subgroupOr(result);
        }
    }
    if ((gl_SubgroupInvocationID & 0x7 /* == % 8*/) == 0) {
        u32 index = uniformity_lod_index(8)(x8_i);
        VOXEL_CHUNK.uniformity.lod_x8[index] = result;
        local_x8_copy[index] = result;
    }

    subgroupBarrier();

    if (gl_LocalInvocationID.x >= 64) {
        return;
    }

    u32vec3 x16_i = u32vec3(
        (gl_LocalInvocationID.x >> 2) & 0x3,
        (gl_LocalInvocationID.x >> 4) & 0x3,
        gl_LocalInvocationID.x & 0x3);
    x8_i = x16_i * 2;
    u32 base_id_x8 = sample_base_voxel_id(temp_voxel_chunk_ptr, x8_i * 8);

    at_least_one_occluding = false;
    for (i32 x = 0; x < 2; ++x)
        for (i32 y = 0; y < 2; ++y)
            for (i32 z = 0; z < 2; ++z) {
                i32vec3 local_i = i32vec3(x8_i) + i32vec3(x, y, z); // x8 space
                u32 mask = uniformity_lod_mask(local_i);
                u32 index = uniformity_lod_index(8)(local_i);
                b32 is_occluding = (local_x8_copy[index] & mask) != 0;
                at_least_one_occluding = at_least_one_occluding || is_occluding || (sample_temp_voxel_id(temp_voxel_chunk_ptr, local_i * 8) != base_id_x8);
            }

    result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(x16_i);
    }
    for (i32 i = 0; i < 8 * WAVE_SIZE_MUL; i++) {
        if ((gl_SubgroupInvocationID >> 2) == i) {
            result = subgroupOr(result);
        }
    }
    if ((gl_SubgroupInvocationID & 0x3) == 0) {
        u32 index = uniformity_lod_index(16)(x16_i);
        VOXEL_CHUNK.uniformity.lod_x16[index] = result;
        local_x16_copy[index] = result;
    }

    subgroupBarrier();

    if (gl_LocalInvocationID.x >= 8) {
        return;
    }

    u32vec3 x32_i = u32vec3(
        (gl_LocalInvocationID.x >> 1) & 0x1,
        (gl_LocalInvocationID.x >> 2) & 0x1,
        gl_LocalInvocationID.x & 0x1);
    x16_i = x32_i * 2;
    u32 base_id_x16 = sample_base_voxel_id(temp_voxel_chunk_ptr, x16_i * 16);

    at_least_one_occluding = false;
    for (i32 x = 0; x < 2; ++x)
        for (i32 y = 0; y < 2; ++y)
            for (i32 z = 0; z < 2; ++z) {
                i32vec3 local_i = i32vec3(x16_i) + i32vec3(x, y, z); // x16 space
                u32 mask = uniformity_lod_mask(local_i);
                u32 index = uniformity_lod_index(16)(local_i);
                b32 is_occluding = (local_x16_copy[index] & mask) != 0;
                at_least_one_occluding = at_least_one_occluding || is_occluding || (sample_temp_voxel_id(temp_voxel_chunk_ptr, local_i * 16) != base_id_x16);
            }

    result = 0;
    if (at_least_one_occluding) {
        result = uniformity_lod_mask(x32_i);
    }
    for (i32 i = 0; i < 16 * WAVE_SIZE_MUL; i++) {
        if ((gl_SubgroupInvocationID >> 1) == i) {
            result = subgroupOr(result);
        }
    }
    if ((gl_SubgroupInvocationID & 0x1) == 0) {
        u32 index = uniformity_lod_index(32)(x32_i);
        VOXEL_CHUNK.uniformity.lod_x32[index] = result;
    }
}
#undef VOXEL_CHUNK

#define VOXEL_WORLD deref(globals).voxel_world
layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
void main() {
    i32vec3 chunk_i = VOXEL_WORLD.chunk_update_infos[gl_WorkGroupID.z].i;
    if (chunk_i == INVALID_CHUNK_I) {
        return;
    }
    u32vec3 chunk_n;
    chunk_n.x = 1u << deref(gpu_input).log2_chunks_per_axis;
    chunk_n.y = chunk_n.x;
    chunk_n.z = chunk_n.x;
    u32 chunk_index = calc_chunk_index_from_worldspace(chunk_i, chunk_n);
    daxa_BufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr = temp_voxel_chunks + gl_WorkGroupID.z;
    daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunk_ptr = voxel_chunks + chunk_index;
    chunk_opt_x8up(temp_voxel_chunk_ptr, voxel_chunk_ptr);

    // Finish the chunk
    if (gl_LocalInvocationIndex == 0) {
        deref(voxel_chunk_ptr).flags = CHUNK_FLAGS_ACCEL_GENERATED;
    }
}
#undef VOXEL_WORLD

#endif

#endif

#if CHUNK_ALLOC_COMPUTE

#extension GL_EXT_shader_atomic_int64 : require

#include <utils/math.glsl>
#include <utils/voxel_malloc.glsl>
#include <utils/voxels.glsl>

shared u32 compression_result[PALETTE_REGION_TOTAL_SIZE];
shared u64 voted_results[PALETTE_REGION_TOTAL_SIZE];
shared u32 palette_size;

void process_palette_region(u32 palette_region_voxel_index, u32 my_voxel, in out u32 my_palette_index) {
    if (palette_region_voxel_index == 0) {
        palette_size = 0;
    }
    voted_results[palette_region_voxel_index] = 0;
    barrier();
    for (u32 algo_i = 0; algo_i < PALETTE_MAX_COMPRESSED_VARIANT_N + 1; ++algo_i) {
        if (my_palette_index == 0) {
            u64 vote_result = atomicCompSwap(voted_results[algo_i], 0, u64(my_voxel) | (u64(1) << u64(32)));
            if (vote_result == 0) {
                my_palette_index = algo_i + 1;
                compression_result[palette_size] = my_voxel;
                palette_size++;
            } else if (my_voxel == u32(vote_result)) {
                my_palette_index = algo_i + 1;
            }
        }
        barrier();
        memoryBarrierShared();
        if (voted_results[algo_i] == 0) {
            break;
        }
    }
}

#define VOXEL_WORLD deref(globals).voxel_world
layout(local_size_x = PALETTE_REGION_SIZE, local_size_y = PALETTE_REGION_SIZE, local_size_z = PALETTE_REGION_SIZE) in;
void main() {
    u32vec3 chunk_n = u32vec3(1u << deref(gpu_input).log2_chunks_per_axis);
    u32 temp_chunk_index = gl_GlobalInvocationID.z / CHUNK_SIZE;
    i32vec3 chunk_i = VOXEL_WORLD.chunk_update_infos[temp_chunk_index].i;
    if (chunk_i == INVALID_CHUNK_I) {
        return;
    }
    u32 chunk_index = calc_chunk_index_from_worldspace(chunk_i, chunk_n);
    u32vec3 inchunk_voxel_i = gl_GlobalInvocationID.xyz - u32vec3(0, 0, temp_chunk_index * CHUNK_SIZE);
    u32 inchunk_voxel_index = inchunk_voxel_i.x + inchunk_voxel_i.y * CHUNK_SIZE + inchunk_voxel_i.z * CHUNK_SIZE * CHUNK_SIZE;
    u32 palette_region_voxel_index =
        gl_LocalInvocationID.x +
        gl_LocalInvocationID.y * PALETTE_REGION_SIZE +
        gl_LocalInvocationID.z * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE;
    u32vec3 palette_i = u32vec3(gl_WorkGroupID.xy, gl_WorkGroupID.z - temp_chunk_index * PALETTES_PER_CHUNK_AXIS);
    u32 palette_region_index =
        palette_i.x +
        palette_i.y * PALETTES_PER_CHUNK_AXIS +
        palette_i.z * PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS;

    daxa_BufferPtr(TempVoxelChunk) temp_voxel_chunk_ptr = temp_voxel_chunks + temp_chunk_index;
    daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunk_ptr = voxel_chunks + chunk_index;

    u32 my_voxel = deref(temp_voxel_chunk_ptr).voxels[inchunk_voxel_index].col_and_id;
    u32 my_palette_index = 0;

    process_palette_region(palette_region_voxel_index, my_voxel, my_palette_index);

    u32 prev_variant_n = deref(voxel_chunk_ptr).palette_headers[palette_region_index].variant_n;
    VoxelMalloc_Pointer prev_blob_ptr = deref(voxel_chunk_ptr).palette_headers[palette_region_index].blob_ptr;

    u32 bits_per_variant = ceil_log2(palette_size);

    u32 compressed_size = 0;
    VoxelMalloc_Pointer blob_ptr = my_voxel;

    if (palette_size > PALETTE_MAX_COMPRESSED_VARIANT_N) {
        compressed_size = PALETTE_REGION_TOTAL_SIZE;
        if (prev_variant_n > 1) {
            blob_ptr = prev_blob_ptr;
            VoxelMalloc_realloc(voxel_malloc_page_allocator, voxel_chunk_ptr, blob_ptr, compressed_size);
        } else {
            blob_ptr = VoxelMalloc_malloc(voxel_malloc_page_allocator, voxel_chunk_ptr, compressed_size);
        }
        if (palette_region_voxel_index == 0) {
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].variant_n = palette_size;
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].blob_ptr = blob_ptr;
        }

        compression_result[palette_region_voxel_index] = my_voxel;
    } else if (palette_size > 1) {
        compressed_size = palette_size + (bits_per_variant * PALETTE_REGION_TOTAL_SIZE + 31) / 32;
        if (prev_variant_n > 1) {
            blob_ptr = prev_blob_ptr;
            VoxelMalloc_realloc(voxel_malloc_page_allocator, voxel_chunk_ptr, blob_ptr, compressed_size);
        } else {
            blob_ptr = VoxelMalloc_malloc(voxel_malloc_page_allocator, voxel_chunk_ptr, compressed_size);
        }
        if (palette_region_voxel_index == 0) {
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].variant_n = palette_size;
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].blob_ptr = blob_ptr;
        }

        u32 mask = (~0u) >> (32 - bits_per_variant);
        u32 bit_index = palette_region_voxel_index * bits_per_variant;
        u32 data_index = bit_index / 32;
        u32 data_offset = bit_index - data_index * 32;
        u32 data = (my_palette_index - 1) & mask;
        u32 address = palette_size + data_index;
        // clang-format off
        atomicAnd(compression_result[address + 0], ~(mask << data_offset));
        atomicOr (compression_result[address + 0],   data << data_offset);
        if (data_offset + bits_per_variant > 32) {
            u32 shift = bits_per_variant - ((data_offset + bits_per_variant) & 0x1f);
            atomicAnd(compression_result[address + 1], ~(mask >> shift));
            atomicOr (compression_result[address + 1],   data >> shift);
        }
        // clang-format on
    } else {
        if (palette_region_voxel_index == 0) {
            if (prev_variant_n > 1) {
                VoxelMalloc_free(voxel_malloc_page_allocator, voxel_chunk_ptr, prev_blob_ptr);
            }
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].variant_n = palette_size;
            deref(voxel_chunk_ptr).palette_headers[palette_region_index].blob_ptr = my_voxel;
        }
    }

    barrier();
    memoryBarrierShared();

    if (palette_region_voxel_index < compressed_size) {
        daxa_RWBufferPtr(daxa_u32) blob_u32s;
        voxel_malloc_address_to_u32_ptr(voxel_malloc_page_allocator, blob_ptr, blob_u32s);
        deref(blob_u32s[palette_region_voxel_index]) = compression_result[palette_region_voxel_index];
    }
}
#undef VOXEL_WORLD

#endif
