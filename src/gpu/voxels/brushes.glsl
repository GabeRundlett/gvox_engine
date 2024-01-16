#pragma once

daxa_b32 mandelbulb(in daxa_f32vec3 c, in out daxa_f32vec3 color) {
    daxa_f32vec3 z = c;
    daxa_u32 i = 0;
    const daxa_f32 n = 8 + floor(good_rand(brush_input.pos) * 5);
    const daxa_u32 MAX_ITER = 4;
    daxa_f32 m = dot(z, z);
    daxa_f32vec4 trap = daxa_f32vec4(abs(z), m);
    for (; i < MAX_ITER; ++i) {
        daxa_f32 r = length(z);
        daxa_f32 p = atan(z.y / z.x);
        daxa_f32 t = acos(z.z / r);
        z = daxa_f32vec3(
            sin(n * t) * cos(n * p),
            sin(n * t) * sin(n * p),
            cos(n * t));
        z = z * pow(r, n) + c;
        trap = min(trap, daxa_f32vec4(abs(z), m));
        m = dot(z, z);
        if (m > 256.0)
            break;
    }
    color = daxa_f32vec3(m, trap.yz) * trap.w;
    return i == MAX_ITER;
}

daxa_f32vec4 terrain_noise(daxa_f32vec3 p) {
    FractalNoiseConfig noise_conf = FractalNoiseConfig(
        /* .amplitude   = */ 1.0,
        /* .persistance = */ 0.2,
        /* .scale       = */ 0.005,
        /* .lacunarity  = */ 4.5,
        /* .octaves     = */ 6);
    daxa_f32vec4 val = fractal_noise(value_noise_texture, deref(gpu_input).sampler_llr, p, noise_conf);
    val.x += p.z * 0.003 - 1.0;
    val.yzw = normalize(val.yzw + vec3(0, 0, 0.003));
    // val.x += -0.24;
    return val;
}

// Spruce Tree generation code (from main branch)
daxa_u32 rand_hash(daxa_u32 x) {
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}
daxa_u32 rand_hash(daxa_u32vec2 v) { return rand_hash(v.x ^ rand_hash(v.y)); }
daxa_u32 rand_hash(daxa_u32vec3 v) {
    return rand_hash(v.x ^ rand_hash(v.y) ^ rand_hash(v.z));
}
daxa_u32 rand_hash(daxa_u32vec4 v) {
    return rand_hash(v.x ^ rand_hash(v.y) ^ rand_hash(v.z) ^ rand_hash(v.w));
}
daxa_f32 rand_float_construct(daxa_u32 m) {
    const daxa_u32 ieee_mantissa = 0x007FFFFFu;
    const daxa_u32 ieee_one = 0x3F800000u;
    m &= ieee_mantissa;
    m |= ieee_one;
    daxa_f32 f = uintBitsToFloat(m);
    return f - 1.0;
}
daxa_f32 rand(daxa_f32 x) { return rand_float_construct(rand_hash(floatBitsToUint(x))); }
daxa_f32 rand(daxa_f32vec2 v) { return rand_float_construct(rand_hash(floatBitsToUint(v))); }
daxa_f32 rand(daxa_f32vec3 v) { return rand_float_construct(rand_hash(floatBitsToUint(v))); }
daxa_f32 rand(daxa_f32vec4 v) { return rand_float_construct(rand_hash(floatBitsToUint(v))); }

struct TreeSDF {
    daxa_f32 wood;
    daxa_f32 leaves;
};

void sd_branch(in out TreeSDF val, in daxa_f32vec3 p, in daxa_f32vec3 origin, in daxa_f32vec3 dir, in daxa_f32 scl) {
    daxa_f32vec3 bp0 = origin;
    daxa_f32vec3 bp1 = bp0 + dir;
    val.wood = min(val.wood, sd_capsule(p, bp0, bp1, 0.10));
    val.leaves = min(val.leaves, sd_sphere(p - bp1, 0.15 * scl));
    bp0 = bp1, bp1 = bp0 + dir * 0.5 + daxa_f32vec3(0, 0, 0.2);
    val.wood = min(val.wood, sd_capsule(p, bp0, bp1, 0.07));
    val.leaves = min(val.leaves, sd_sphere(p - bp1, 0.15 * scl));
}

TreeSDF sd_spruce_tree(in daxa_f32vec3 p, in daxa_f32vec3 seed) {
    TreeSDF val = TreeSDF(1e5, 1e5);
    val.wood = min(val.wood, sd_capsule(p, daxa_f32vec3(0, 0, 0), daxa_f32vec3(0, 0, 4.5), 0.15));
    val.leaves = min(val.leaves, sd_capsule(p, daxa_f32vec3(0, 0, 4.5), daxa_f32vec3(0, 0, 5.0), 0.15));
    for (daxa_u32 i = 0; i < 5; ++i) {
        daxa_f32 scl = 1.0 / (1.0 + i * 0.5);
        daxa_f32 scl2 = 1.0 / (1.0 + i * 0.1);
        daxa_u32 branch_n = 8 - i;
        for (daxa_u32 branch_i = 0; branch_i < branch_n; ++branch_i) {
            daxa_f32 angle = (1.0 / branch_n * branch_i) * 2.0 * PI + rand(seed + i + 1.0 * branch_i) * 0.5;
            sd_branch(val, p, daxa_f32vec3(0, 0, 1.0 + i * 0.8) * 1.0, normalize(daxa_f32vec3(cos(angle), sin(angle), +0.0)) * scl, scl2 * 1.5);
        }
    }
    return val;
}

// Random noise
daxa_f32vec3 hash33(daxa_f32vec3 p3) {
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
daxa_f32 fbm2(vec2 uv) {
    daxa_f32 f = 0;
    mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
    f = 0.5000 * value_noise(uv);
    uv = m * uv;
    f += 0.2500 * value_noise(uv);
    uv = m * uv;
    return f * .5 + .5;
}

// Forest generation
#define TREE_MARCH_STEPS 4

daxa_f32vec3 get_closest_surface(daxa_f32vec3 center_cell_world, daxa_f32 current_noise, daxa_f32 rep, inout daxa_f32 scale) {
    daxa_f32vec3 offset = hash33(center_cell_world);
    scale = offset.z * .3 + .7;
    center_cell_world.xy += (offset.xy * 2 - 1) * max(0, rep / scale - 5);

    daxa_f32 step_size = rep / 2 / TREE_MARCH_STEPS;

    // Above terrain
    if (current_noise > 0) {
        for (daxa_u32 i = 0; i < TREE_MARCH_STEPS; i++) {
            center_cell_world.z -= step_size;
            if (terrain_noise(center_cell_world).x < 0)
                return center_cell_world;
        }
    }
    // Inside terrain
    else {
        for (daxa_u32 i = 0; i < TREE_MARCH_STEPS; i++) {
            center_cell_world.z += step_size;
            if (terrain_noise(center_cell_world).x > 0)
                return center_cell_world - daxa_f32vec3(0, 0, step_size);
        }
    }

    return daxa_f32vec3(0);
}

// Color palettes
daxa_f32vec3 palette(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d) {
    return a + b * cos(6.28318 * (c * t + d));
}
daxa_f32vec3 forest_biome_palette(daxa_f32 t) {
    return pow(daxa_f32vec3(85, 114, 78) / 255.0, vec3(2.2)); // palette(t + .5, vec3(0.07, 0.22, 0.03), vec3(0.03, 0.05, 0.01), vec3(-1.212, -2.052, 0.058), vec3(1.598, 6.178, 0.380));
}

void brushgen_world_terrain(in out Voxel voxel) {
    voxel_pos += daxa_f32vec3(0, 0, 280);

    daxa_f32vec4 val4 = terrain_noise(voxel_pos);
    daxa_f32 val = val4.x;
    daxa_f32vec3 nrm = normalize(val4.yzw); // terrain_nrm(voxel_pos);
    daxa_f32 upwards = dot(nrm, daxa_f32vec3(0, 0, 1));

    // Smooth noise depending on 2d position only
    daxa_f32 voxel_noise_xy = fbm2(voxel_pos.xy / 8 / 40);
    // Smooth biome color
    daxa_f32vec3 forest_biome_color = forest_biome_palette(voxel_noise_xy * 2 - 1);

    if (val < 0) {
        voxel.material_type = 1;
        const bool SHOULD_COLOR_WORLD = true;
        if (SHOULD_COLOR_WORLD) {
            daxa_f32 r = good_rand(-val);
            if (val > -0.002) {
                // voxel.normal = nrm;
            }
            if (val > -0.002 && upwards > 0.65) {
                voxel.color = pow(daxa_f32vec3(105, 126, 78) / 255.0, vec3(2.2));
                if (r < 0.5) {
                    voxel.color *= 0.7;
                }
                // Mix with biome color
                voxel.color = mix(voxel.color, forest_biome_color * .75, .3);
            } else if (val > -0.05 && upwards > 0.5) {
                voxel.color = daxa_f32vec3(0.13, 0.09, 0.05);
                if (r < 0.5) {
                    voxel.color.r *= 0.5;
                    voxel.color.g *= 0.5;
                    voxel.color.b *= 0.5;
                } else if (r < 0.52) {
                    voxel.color.r *= 1.5;
                    voxel.color.g *= 1.5;
                    voxel.color.b *= 1.5;
                }
            } else if (val < -0.01 && val > -0.07 && upwards > 0.2) {
                voxel.color = daxa_f32vec3(0.17, 0.15, 0.07);
                if (r < 0.5) {
                    voxel.color.r *= 0.75;
                    voxel.color.g *= 0.75;
                    voxel.color.b *= 0.75;
                }
            } else {
                voxel.color = daxa_f32vec3(0.11, 0.10, 0.07);
            }
        } else {
            voxel.color = daxa_f32vec3(0.25);
        }
    } else if (ENABLE_TREE_GENERATION != 0) {
        // Meters per cell
        daxa_f32 rep = 6;

        // Global cell ID
        daxa_f32vec3 qid = floor(voxel_pos / rep);
        // Local coordinates in current cell (centered at 0 [-rep/2, rep/2])
        daxa_f32vec3 q = mod(voxel_pos, rep) - rep / 2;
        // Current cell's center voxel (world space)
        daxa_f32vec3 cell_center_world = qid * rep + rep / 2.;

        // Query terrain noise at current cell's center
        daxa_f32vec4 center_noise = terrain_noise(cell_center_world);

        // Optimization: only run for chunks near enough the terrain surface
        bool can_spawn = center_noise.x >= -0.01 * rep / 4 && center_noise.x < 0.03 * rep / 4;

        // Forest density
        daxa_f32 forest_noise = fbm2(qid.xy / 10.);
        daxa_f32 forest_density = .45;

        if (forest_noise > forest_density)
            can_spawn = false;

        if (can_spawn) {
            // Tree scale
            daxa_f32 scale;
            // Try to get the nearest point on the surface below (in the starting cell)
            daxa_f32vec3 hitPoint = get_closest_surface(cell_center_world, center_noise.x, rep, scale);

            if (hitPoint == daxa_f32vec3(0) && center_noise.x > 0) {
                // If no terrain was found, try again for the bottom cell (upper tree case)
                scale = forest_noise;
                daxa_f32vec3 down_neighbor_cell_center_world = cell_center_world - daxa_f32vec3(0, 0, rep);
                hitPoint = get_closest_surface(down_neighbor_cell_center_world, terrain_noise(down_neighbor_cell_center_world).x, rep, scale);
            }

            // Debug space repetition boundaries
            // daxa_f32 tresh = 1. / 8.;
            // if (abs(abs(q.x)-rep/2.) <= tresh && abs(abs(q.y)-rep/2.) <= tresh ||
            //     abs(abs(q.x)-rep/2.) <= tresh && abs(abs(q.z)-rep/2.) <= tresh ||
            //     abs(abs(q.z)-rep/2.) <= tresh && abs(abs(q.y)-rep/2.) <= tresh) {
            //     voxel.material_type = 1;
            //     voxel.color = daxa_f32vec3(0,0,0);
            // }

            // Distance to tree
            TreeSDF tree = sd_spruce_tree((voxel_pos - hitPoint) / scale, qid);

            daxa_f32vec3 h_cell = vec3(0);  // hash33(qid);
            daxa_f32vec3 h_voxel = vec3(0); // hash33(voxel_pos);

            // Colorize tree
            if (tree.wood < 0) {
                voxel.material_type = 1;
                voxel.color = daxa_f32vec3(.68, .4, .15) * 0.16;
            } else if (tree.leaves < 0) {
                voxel.material_type = 1;
                voxel.color = forest_biome_color * 0.5;
            }
        }
    }
}

void brushgen_world(in out Voxel voxel) {
    if (false) { // Mandelbulb world
        daxa_f32vec3 mandelbulb_color;
        if (mandelbulb((voxel_pos / 64 - 1) * 1, mandelbulb_color)) {
            voxel.color = daxa_f32vec3(0.02);
            voxel.material_type = 1;
        }
    } else if (false) { // Solid world
        voxel.material_type = 1;
        voxel.color = daxa_f32vec3(0.5, 0.1, 0.8);
    } else if (GEN_MODEL != 0) { // Model world
        daxa_u32 packed_col_data = sample_gvox_palette_voxel(gvox_model, world_voxel, 0);
        // voxel.material_type = sample_gvox_palette_voxel(gvox_model, world_voxel, 0);
        voxel.material_type = packed_col_data >> 0x18;
        // daxa_u32 packed_emi_data = sample_gvox_palette_voxel(gvox_model, world_voxel, 2);
        voxel.color = uint_rgba8_to_f32vec4(packed_col_data).rgb;
        // if (voxel.material_type != 0) {
        //     voxel.material_type = 2;
        // }
        if (voxel_pos.z == -1.0 / VOXEL_SCL) {
            voxel.color = vec3(0.1);
            voxel.material_type = 1;
        }
    } else if (true) { // Terrain world
        brushgen_world_terrain(voxel);
    } else if (true) { // Ball world (each ball is centered on a chunk center)
        if (length(fract(voxel_pos / 8) - 0.5) < 0.15) {
            voxel.material_type = 1;
            voxel.color = daxa_f32vec3(0.1);
        }
    } else if (false) { // Checker board world
        daxa_u32vec3 voxel_i = daxa_u32vec3(voxel_pos / 8);
        if ((voxel_i.x + voxel_i.y + voxel_i.z) % 2 == 1) {
            voxel.material_type = 1;
            voxel.color = daxa_f32vec3(0.1);
        }
    }
}

void brushgen_a(in out Voxel voxel) {
    PackedVoxel voxel_data = sample_voxel_chunk(voxel_malloc_page_allocator, voxel_chunk_ptr, inchunk_voxel_i);
    Voxel prev_voxel = unpack_voxel(voxel_data);

    voxel.color = prev_voxel.color;
    voxel.material_type = prev_voxel.material_type;

    if (sd_capsule(voxel_pos, brush_input.pos, brush_input.prev_pos, 32.0 / VOXEL_SCL) < 0) {
        voxel.color = daxa_f32vec3(0, 0, 0);
        voxel.material_type = 0;
    }
}

void brushgen_b(in out Voxel voxel) {
    PackedVoxel voxel_data = sample_voxel_chunk(voxel_malloc_page_allocator, voxel_chunk_ptr, inchunk_voxel_i);
    Voxel prev_voxel = unpack_voxel(voxel_data);

    voxel.color = prev_voxel.color;
    voxel.material_type = prev_voxel.material_type;

    if (sd_capsule(voxel_pos, brush_input.pos, brush_input.prev_pos, 32.0 / VOXEL_SCL) < 0) {
        // daxa_f32 val = noise(voxel_pos) + (rand() - 0.5) * 1.2;
        // if (val > 0.3) {
        //     voxel.color = daxa_f32vec3(0.99, 0.03, 0.01);
        // } else if (val > -0.3) {
        //     voxel.color = daxa_f32vec3(0.91, 0.05, 0.01);
        // } else {
        //     voxel.color = daxa_f32vec3(0.91, 0.15, 0.01);
        // }
        // voxel.color = daxa_f32vec3(rand(), rand(), rand());
        // voxel.color = daxa_f32vec3(floor(rand() * 4.0) / 4.0, floor(rand() * 4.0) / 4.0, floor(rand() * 4.0) / 4.0);
        voxel.color = daxa_f32vec3(0.1, 0.3, 0.8);
        voxel.material_type = 1;
    }
}

void brushgen_particles(in out Voxel voxel) {
    // for (daxa_u32 particle_i = 0; particle_i < deref(globals).voxel_particles_state.place_count; ++particle_i) {
    //     daxa_u32 sim_index = deref(placed_voxel_particles[particle_i]);
    //     SimulatedVoxelParticle self = deref(simulated_voxel_particles[sim_index]);
    //     if (daxa_u32vec3(floor(self.pos * VOXEL_SCL)) == voxel_i) {
    //         voxel.color = daxa_f32vec3(0.8, 0.8, 0.8);
    //         voxel.material_type = 1;
    //         return;
    //     }
    // }

    PackedVoxel voxel_data = sample_voxel_chunk(voxel_malloc_page_allocator, voxel_chunk_ptr, inchunk_voxel_i);
    Voxel prev_voxel = unpack_voxel(voxel_data);

    voxel.color = prev_voxel.color;
    voxel.material_type = prev_voxel.material_type;
}
