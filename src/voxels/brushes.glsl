#pragma once

#include <utilities/gpu/random.glsl>
#include <utilities/gpu/noise.glsl>

#include <g_samplers>

bool mandelbulb(in vec3 c, in out vec3 color) {
    vec3 z = c;
    uint i = 0;
    const float n = 8 + floor(good_rand(brush_input.pos) * 5);
    const uint MAX_ITER = 4;
    float m = dot(z, z);
    vec4 trap = vec4(abs(z), m);
    for (; i < MAX_ITER; ++i) {
        float r = length(z);
        float p = atan(z.y / z.x);
        float t = acos(z.z / r);
        z = vec3(
            sin(n * t) * cos(n * p),
            sin(n * t) * sin(n * p),
            cos(n * t));
        z = z * pow(r, n) + c;
        trap = min(trap, vec4(abs(z), m));
        m = dot(z, z);
        if (m > 256.0)
            break;
    }
    color = vec3(m, trap.yz) * trap.w;
    return i == MAX_ITER;
}

vec4 terrain_noise(vec3 p) {
    FractalNoiseConfig noise_conf = FractalNoiseConfig(
        /* .amplitude   = */ 1.0,
        /* .persistance = */ 0.2,
        /* .scale       = */ 0.005,
        /* .lacunarity  = */ 4.5,
        /* .octaves     = */ 6);
    vec4 val = fractal_noise(value_noise_texture, g_sampler_llr, p, noise_conf);
    val.x += p.z * 0.003 - 1.0;
    val.yzw = normalize(val.yzw + vec3(0, 0, 0.003));
    // val.x += -0.24;
    return val;
}

struct TreeSDF {
    float wood;
    float leaves;
};

void sd_branch(in out TreeSDF val, in vec3 p, in vec3 origin, in vec3 dir, in float scl) {
    vec3 bp0 = origin;
    vec3 bp1 = bp0 + dir;
    val.wood = min(val.wood, sd_capsule(p, bp0, bp1, 0.10));
    val.leaves = min(val.leaves, sd_sphere(p - bp1, 0.15 * scl));
    bp0 = bp1, bp1 = bp0 + dir * 0.5 + vec3(0, 0, 0.2);
    val.wood = min(val.wood, sd_capsule(p, bp0, bp1, 0.07));
    val.leaves = min(val.leaves, sd_sphere(p - bp1, 0.15 * scl));
}

TreeSDF sd_spruce_tree(in vec3 p, in vec3 seed) {
    TreeSDF val = TreeSDF(1e5, 1e5);
    val.wood = min(val.wood, sd_capsule(p, vec3(0, 0, 0), vec3(0, 0, 4.5), 0.15));
    val.leaves = min(val.leaves, sd_capsule(p, vec3(0, 0, 4.5), vec3(0, 0, 5.0), 0.15));
    for (uint i = 0; i < 5; ++i) {
        float scl = 1.0 / (1.0 + i * 0.5);
        float scl2 = 1.0 / (1.0 + i * 0.1);
        uint branch_n = 8 - i;
        for (uint branch_i = 0; branch_i < branch_n; ++branch_i) {
            float angle = (1.0 / branch_n * branch_i) * 2.0 * M_PI + good_rand(seed + i + 1.0 * branch_i) * 0.5;
            sd_branch(val, p, vec3(0, 0, 1.0 + i * 0.8) * 1.0, normalize(vec3(cos(angle), sin(angle), +0.0)) * scl, scl2 * 1.5);
        }
    }
    return val;
}

// Forest generation
#define TREE_MARCH_STEPS 4

vec3 get_closest_surface(vec3 center_cell_world, float current_noise, float rep, inout float scale) {
    vec3 offset = hash33(center_cell_world);
    scale = offset.z * .3 + .7;
    center_cell_world.xy += (offset.xy * 2 - 1) * max(0, rep / scale - 5);

    float step_size = rep / 2 / TREE_MARCH_STEPS;

    // Above terrain
    if (current_noise > 0) {
        for (uint i = 0; i < TREE_MARCH_STEPS; i++) {
            center_cell_world.z -= step_size;
            if (terrain_noise(center_cell_world).x < 0)
                return center_cell_world;
        }
    }
    // Inside terrain
    else {
        for (uint i = 0; i < TREE_MARCH_STEPS; i++) {
            center_cell_world.z += step_size;
            if (terrain_noise(center_cell_world).x > 0)
                return center_cell_world - vec3(0, 0, step_size);
        }
    }

    return vec3(0);
}

// Color palettes
vec3 palette(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d) {
    return a + b * cos(6.28318 * (c * t + d));
}
vec3 forest_biome_palette(float t) {
    return pow(vec3(85, 114, 78) / 255.0, vec3(2.2)); // palette(t + .5, vec3(0.07, 0.22, 0.03), vec3(0.03, 0.05, 0.01), vec3(-1.212, -2.052, 0.058), vec3(1.598, 6.178, 0.380));
}

void brushgen_world_terrain(in out Voxel voxel) {
    voxel_pos += vec3(0, 0, 280);

    vec4 val4 = terrain_noise(voxel_pos);
    float val = val4.x;
    vec3 nrm = normalize(val4.yzw); // terrain_nrm(voxel_pos);
    float upwards = dot(nrm, vec3(0, 0, 1));

    // Smooth noise depending on 2d position only
    float voxel_noise_xy = fbm2(voxel_pos.xy / 8 / 40);
    // Smooth biome color
    vec3 forest_biome_color = forest_biome_palette(voxel_noise_xy * 2 - 1);

    if (val < 0) {
        voxel.material_type = 1;
        const bool SHOULD_COLOR_WORLD = true;
        voxel.normal = nrm;
        voxel.roughness = 1.0;
        if (SHOULD_COLOR_WORLD) {
            float r = good_rand(-val);
            if (val > -0.002 && upwards > 0.65) {
                voxel.color = pow(vec3(105, 126, 78) / 255.0, vec3(2.2));
                if (r < 0.5) {
                    voxel.color *= 0.7;
                    voxel.roughness = 0.7;
                }
                // Mix with biome color
                voxel.color = mix(voxel.color, forest_biome_color * .75, .3);
            } else if (val > -0.05 && upwards > 0.5) {
                voxel.color = vec3(0.13, 0.09, 0.05);
                if (r < 0.5) {
                    voxel.color.r *= 0.5;
                    voxel.color.g *= 0.5;
                    voxel.color.b *= 0.5;
                    voxel.roughness = 0.99;
                } else if (r < 0.52) {
                    voxel.color.r *= 1.5;
                    voxel.color.g *= 1.5;
                    voxel.color.b *= 1.5;
                    voxel.roughness = 0.95;
                }
            } else if (val < -0.01 && val > -0.07 && upwards > 0.2) {
                voxel.color = vec3(0.17, 0.15, 0.07);
                if (r < 0.5) {
                    voxel.color.r *= 0.75;
                    voxel.color.g *= 0.75;
                    voxel.color.b *= 0.75;
                }
                voxel.roughness = 0.6;
            } else {
                voxel.color = vec3(0.11, 0.10, 0.07);
                voxel.roughness = 0.9;
            }
        } else {
            voxel.color = vec3(0.25);
        }
    } else if (ENABLE_TREE_GENERATION != 0) {
        // Meters per cell
        float rep = 6;

        // Global cell ID
        vec3 qid = floor(voxel_pos / rep);
        // Local coordinates in current cell (centered at 0 [-rep/2, rep/2])
        vec3 q = mod(voxel_pos, rep) - rep / 2;
        // Current cell's center voxel (world space)
        vec3 cell_center_world = qid * rep + rep / 2.;

        // Query terrain noise at current cell's center
        vec4 center_noise = terrain_noise(cell_center_world);

        // Optimization: only run for chunks near enough the terrain surface
        bool can_spawn = center_noise.x >= -0.01 * rep / 4 && center_noise.x < 0.03 * rep / 4;

        // Forest density
        float forest_noise = fbm2(qid.xy / 10.);
        float forest_density = .45;

        if (forest_noise > forest_density)
            can_spawn = false;

        if (can_spawn) {
            // Tree scale
            float scale;
            // Try to get the nearest point on the surface below (in the starting cell)
            vec3 hitPoint = get_closest_surface(cell_center_world, center_noise.x, rep, scale);

            if (hitPoint == vec3(0) && center_noise.x > 0) {
                // If no terrain was found, try again for the bottom cell (upper tree case)
                scale = forest_noise;
                vec3 down_neighbor_cell_center_world = cell_center_world - vec3(0, 0, rep);
                hitPoint = get_closest_surface(down_neighbor_cell_center_world, terrain_noise(down_neighbor_cell_center_world).x, rep, scale);
            }

            // Debug space repetition boundaries
            // float tresh = 1. / 8.;
            // if (abs(abs(q.x)-rep/2.) <= tresh && abs(abs(q.y)-rep/2.) <= tresh ||
            //     abs(abs(q.x)-rep/2.) <= tresh && abs(abs(q.z)-rep/2.) <= tresh ||
            //     abs(abs(q.z)-rep/2.) <= tresh && abs(abs(q.y)-rep/2.) <= tresh) {
            //     voxel.material_type = 1;
            //     voxel.color = vec3(0,0,0);
            // }

            // Distance to tree
            TreeSDF tree = sd_spruce_tree((voxel_pos - hitPoint) / scale, qid);

            vec3 h_cell = vec3(0);  // hash33(qid);
            vec3 h_voxel = vec3(0); // hash33(voxel_pos);

            // Colorize tree
            if (tree.wood < 0) {
                voxel.material_type = 1;
                voxel.color = vec3(.68, .4, .15) * 0.16;
                voxel.roughness = 0.99;
            } else if (tree.leaves < 0) {
                voxel.material_type = 1;
                voxel.color = forest_biome_color * 0.5;
                voxel.roughness = 0.5;
            }
        }
    }
}

#define GEN_MODEL 1

void brushgen_world(in out Voxel voxel) {
    if (false) { // Mandelbulb world
        vec3 mandelbulb_color;
        if (mandelbulb((voxel_pos / 64 - 1) * 1, mandelbulb_color)) {
            voxel.color = vec3(0.02);
            voxel.material_type = 1;
            voxel.roughness = 0.5;
        }
    } else if (false) { // Solid world
        voxel.material_type = 1;
        voxel.color = vec3(0.5, 0.1, 0.8);
        voxel.roughness = 0.5;
    } else if (false) { // test
        float map_scale = 2.0;
        vec2 map_uv = voxel_pos.xy / (4097.0 / VOXEL_SCL) / map_scale;

        const float offset = 1.0 / 512.0;
        vec4 heights = textureGather(daxa_sampler2D(test_texture, g_sampler_llc), map_uv);
        heights = heights * 4097.0 / VOXEL_SCL - 128.0;
        heights = heights * map_scale * 0.6;
        vec2 w = fract(map_uv * 4097.0 - 0.5 + offset);
        float map_height = mix(mix(heights.w, heights.z, w.x), mix(heights.x, heights.y, w.x), w.y);
        vec3 map_color = texture(daxa_sampler2D(test_texture2, g_sampler_llc), map_uv).rgb;
        bool solid = voxel_pos.z < map_height;
        if (solid) {
            voxel.color = pow(map_color, vec3(2.2));
            voxel.material_type = 1;
            voxel.roughness = 0.99;

            vec3 pos_origin = floor(voxel_pos);
            pos_origin.z = heights.w;
            vec3 pos_down = pos_origin + vec3(0, map_scale / VOXEL_SCL, 0);
            pos_down.z = heights.x;
            vec3 pos_right = pos_origin + vec3(map_scale / VOXEL_SCL, 0, 0);
            pos_right.z = heights.z;
            vec3 vertical_dir = normalize(pos_origin - pos_down);
            vec3 horizontal_dir = normalize(pos_origin - pos_right);
            voxel.normal = normalize(cross(horizontal_dir, vertical_dir));
        }
    } else if (GEN_MODEL != 0) { // Model world
        uint packed_col_data = sample_gvox_palette_voxel(gvox_model, world_voxel, 0);
        // voxel.material_type = sample_gvox_palette_voxel(gvox_model, world_voxel, 0);
        voxel.color = uint_rgba8_to_f32vec4(packed_col_data).rgb;
        voxel.material_type = ((packed_col_data >> 0x18) != 0 || voxel.color != vec3(0)) ? 1 : 0;
        voxel.roughness = 0.9;

        float test = length(vec3(1.0, 0.25, 0.0) - voxel.color);
        if (test <= 0.7) {
            // voxel.color = vec3(0.1);
            voxel.material_type = 3;
            voxel.roughness = test * 0.1;
        }
        // uint packed_emi_data = sample_gvox_palette_voxel(gvox_model, world_voxel, 2);
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
            voxel.color = vec3(0.1);
            voxel.roughness = 0.5;
        }
    } else if (false) { // Checker board world
        uvec3 voxel_i = uvec3(voxel_pos / 8);
        if ((voxel_i.x + voxel_i.y + voxel_i.z) % 2 == 1) {
            voxel.material_type = 1;
            voxel.color = vec3(0.1);
            voxel.roughness = 0.5;
        }
    }
}

void brushgen_a(in out Voxel voxel) {
    PackedVoxel voxel_data = sample_voxel_chunk(voxel_malloc_page_allocator, voxel_chunk_ptr, inchunk_voxel_i);
    Voxel prev_voxel = unpack_voxel(voxel_data);

    voxel.color = prev_voxel.color;
    voxel.material_type = prev_voxel.material_type;
    voxel.normal = prev_voxel.normal;
    voxel.roughness = prev_voxel.roughness;

    float sd = sd_capsule(voxel_pos, brush_input.pos + brush_input.pos_offset, brush_input.prev_pos + brush_input.prev_pos_offset, 32.0 / VOXEL_SCL);
    if (sd < 0) {
        voxel.color = vec3(0, 0, 0);
        voxel.material_type = 0;
    }
    if (sd < 2.5 / VOXEL_SCL) {
        voxel.normal = vec3(0, 0, 1);
    }
}

void brushgen_b(in out Voxel voxel) {
    PackedVoxel voxel_data = sample_voxel_chunk(voxel_malloc_page_allocator, voxel_chunk_ptr, inchunk_voxel_i);
    Voxel prev_voxel = unpack_voxel(voxel_data);

    voxel.color = prev_voxel.color;
    voxel.material_type = prev_voxel.material_type;
    voxel.normal = prev_voxel.normal;
    voxel.roughness = prev_voxel.roughness;

    float sd = sd_capsule(voxel_pos, brush_input.pos + brush_input.pos_offset, brush_input.prev_pos + brush_input.prev_pos_offset, 32.0 / VOXEL_SCL);
    if (sd < 0) {
        // float val = noise(voxel_pos) + (good_rand() - 0.5) * 1.2;
        // if (val > 0.3) {
        //     voxel.color = vec3(0.99, 0.03, 0.01);
        // } else if (val > -0.3) {
        //     voxel.color = vec3(0.91, 0.05, 0.01);
        // } else {
        //     voxel.color = vec3(0.91, 0.15, 0.01);
        // }
        // voxel.color = vec3(good_rand(), good_rand(), good_rand());
        // voxel.color = vec3(floor(good_rand() * 4.0) / 4.0, floor(good_rand() * 4.0) / 4.0, floor(good_rand() * 4.0) / 4.0);
        voxel.material_type = 3;
        voxel.color = vec3(0.95, 0.05, 0.05);
        voxel.roughness = 0.2;
        // voxel.normal = normalize(voxel_pos - (brush_input.pos + brush_input.pos_offset));
    }
    if (sd < 2.5 / VOXEL_SCL) {
        voxel.normal = vec3(0, 0, 1);
    }
}

void brushgen_particles(in out Voxel voxel) {
    // for (uint particle_i = 0; particle_i < deref(globals).voxel_particles_state.place_count; ++particle_i) {
    //     uint sim_index = deref(advance(placed_voxel_particles, particle_i));
    //     SimulatedVoxelParticle self = deref(advance(simulated_voxel_particles, sim_index));
    //     if (uvec3(floor(self.pos * VOXEL_SCL)) == voxel_i) {
    //         voxel.color = vec3(0.8, 0.8, 0.8);
    //         voxel.material_type = 1;
    //         return;
    //     }
    // }

    PackedVoxel voxel_data = sample_voxel_chunk(voxel_malloc_page_allocator, voxel_chunk_ptr, inchunk_voxel_i);
    Voxel prev_voxel = unpack_voxel(voxel_data);

    voxel.color = prev_voxel.color;
    voxel.material_type = prev_voxel.material_type;
}
