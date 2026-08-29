#pragma once

#include <utilities/gpu/random.glsl>
#include <utilities/gpu/noise.glsl>

#include <renderer/globals.glsl>

bool mandelbulb(in vec3 c, in out vec3 color) {
    vec3 z = c;
    uint i = 0;
    const float n = 8; // + floor(good_rand(brush_input.pos) * 5);
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
    p = p + vec3(-183, -110, -54);
    FractalNoiseConfig noise_conf = FractalNoiseConfig(
        /* .amplitude   = */ 1.0,
        /* .persistance = */ 0.2,
        /* .scale       = */ 0.005,
        /* .lacunarity  = */ 4.5,
        /* .octaves     = */ 6);
    vec4 val = fractal_noise(g_value_noise_tex, g_sampler_llr, p, noise_conf);
    // const float ground_level = 6362000.0;
    const float ground_level = 0.0;
    val.x += (p.z - ground_level + 100.0) * 0.003 - 0.4;
    val.yzw = normalize(val.yzw + vec3(0, 0, 0.003));
    // val.x += -0.24;
    return val;
}

struct TreeSDF {
    float wood;
    float leaves;
};

struct TreeSDFNrm {
    float wood;
    float leaves;
    vec3 wood_nrm;
    vec3 leaves_nrm;
};

vec3 sd_capsule_normal(in vec3 p, in vec3 a, in vec3 b) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return normalize(pa - ba * h);
}

void sd_spruce_branch(in out TreeSDFNrm val, in vec3 p, in vec3 origin, in vec3 dir, in float scl) {
    vec3 bp0 = origin;
    vec3 bp1 = bp0 + dir;
    val.wood = min(val.wood, sd_capsule(p, bp0, bp1, 0.10));
    float leaves_d0 = sd_sphere(p - bp1, 0.15 * scl);
    if (leaves_d0 < val.leaves) {
        val.leaves = leaves_d0;
        val.leaves_nrm = normalize(p - bp1);
    }
    bp0 = bp1, bp1 = bp0 + dir * 0.5 + vec3(0, 0, 0.2);
    val.wood = min(val.wood, sd_capsule(p, bp0, bp1, 0.07));
    float leaves_d1 = sd_sphere(p - bp1, 0.15 * scl);
    if (leaves_d1 < val.leaves) {
        val.leaves = leaves_d1;
        val.leaves_nrm = normalize(p - bp1);
    }
}

TreeSDFNrm sd_spruce_tree(in vec3 p, in vec3 seed) {
    TreeSDFNrm val = TreeSDFNrm(1e5, 1e5, GENERATE_NORMAL, GENERATE_NORMAL);
    val.wood = min(val.wood, sd_capsule(p, vec3(0, 0, 0), vec3(0, 0, 4.5), 0.15));
    val.leaves = min(val.leaves, sd_capsule(p, vec3(0, 0, 4.5), vec3(0, 0, 5.0), 0.15));
    for (uint i = 0; i < 5; ++i) {
        float scl = 1.0 / (1.0 + i * 0.5);
        float scl2 = 1.0 / (1.0 + i * 0.1);
        uint branch_n = 8 - i;
        for (uint branch_i = 0; branch_i < branch_n; ++branch_i) {
            float angle = (1.0 / branch_n * branch_i) * 2.0 * M_PI + good_rand(seed + i + 1.0 * branch_i) * 0.5;
            sd_spruce_branch(val, p, vec3(0, 0, 1.0 + i * 0.8) * 1.0, normalize(vec3(cos(angle), sin(angle), +0.0)) * scl, scl2 * 1.5);
        }
    }
    return val;
}

// Color palettes
vec3 palette(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d) {
    return a + b * cos(6.28318 * (c * t + d));
}
vec3 forest_biome_palette(float t) {
    return pow(vec3(85, 154, 78) / 255.0, vec3(2.2)); // palette(t + .5, vec3(0.07, 0.22, 0.03), vec3(0.03, 0.05, 0.01), vec3(-1.212, -2.052, 0.058), vec3(1.598, 6.178, 0.380));
}

// #define UserAllocatorType GrassStrandAllocator
// #define UserIndexType uint
// #define UserMaxElementCount MAX_GRASS_BLADES
// #include <utilities/allocator.glsl>

// #define UserAllocatorType FlowerAllocator
// #define UserIndexType uint
// #define UserMaxElementCount MAX_FLOWERS
// #include <utilities/allocator.glsl>

// #define UserAllocatorType TreeParticleAllocator
// #define UserIndexType uint
// #define UserMaxElementCount MAX_TREE_PARTICLES
// #include <utilities/allocator.glsl>

// #define UserAllocatorType FireParticleAllocator
// #define UserIndexType uint
// #define UserMaxElementCount MAX_FIRE_PARTICLES
// #include <utilities/allocator.glsl>

void spawn_grass(in out Voxel voxel) {
    // GrassStrand grass_strand;
    // grass_strand.origin = voxel_pos;
    // grass_strand.packed_voxel = pack_voxel(voxel);
    // grass_strand.flags = 1;

    // uint index = GrassStrandAllocator_malloc(grass_allocator);
    // daxa_RWBufferPtr(GrassStrand) grass_strands = deref(grass_allocator).heap;
    // if (index < MAX_GRASS_BLADES) {
    //     deref(advance(grass_strands, index)) = grass_strand;
    // }
}
void spawn_flower(in out Voxel voxel, uint flower_type) {
    // Flower flower;
    // flower.origin = voxel_pos;
    // flower.packed_voxel = pack_voxel(voxel);
    // flower.type = flower_type;
    // flower.flags = 1;

    // uint index = FlowerAllocator_malloc(flower_allocator);
    // daxa_RWBufferPtr(Flower) flowers = deref(flower_allocator).heap;
    // if (index < MAX_FLOWERS) {
    //     deref(advance(flowers, index)) = flower;
    // }
}
void spawn_tree_particle(in out Voxel voxel) {
    // TreeParticle tree_particle;
    // tree_particle.origin = voxel_pos;
    // tree_particle.packed_voxel = pack_voxel(voxel);
    // tree_particle.flags = 1;

    // uint index = TreeParticleAllocator_malloc(tree_particle_allocator);
    // daxa_RWBufferPtr(TreeParticle) tree_particles = deref(tree_particle_allocator).heap;
    // if (index < MAX_TREE_PARTICLES) {
    //     deref(advance(tree_particles, index)) = tree_particle;
    // }
}
void spawn_fire_particle(in out Voxel voxel) {
    // FireParticle fire_particle;
    // fire_particle.origin = voxel_pos;
    // fire_particle.packed_voxel = pack_voxel(voxel);
    // fire_particle.flags = 1;

    // uint index = FireParticleAllocator_malloc(fire_particle_allocator);
    // daxa_RWBufferPtr(FireParticle) fire_particles = deref(fire_particle_allocator).heap;
    // if (index < MAX_FIRE_PARTICLES) {
    //     deref(advance(fire_particles, index)) = fire_particle;
    // }
}

vec3 voxel_pos;

void try_spawn_grass(in out Voxel voxel, vec3 nrm) {
    // randomly spawn grass
    float r2 = good_rand(voxel_pos.xy);
    float upwards = dot(nrm, vec3(0, 0, 1));
    if (upwards > 0.35 && r2 < 0.75) {
        FractalNoiseConfig noise_conf = FractalNoiseConfig(
            /* .amplitude   = */ 1.0,
            /* .persistance = */ 0.5,
            /* .scale       = */ 0.1,
            /* .lacunarity  = */ 2,
            /* .octaves     = */ 3);
        vec4 flower_noise_val = fractal_noise(g_value_noise_tex, g_sampler_llr, vec3(voxel_pos.xy, 0), noise_conf);
        float v = flower_noise_val.x * (1.0 / 0.875);

        // voxel.albedo = pow(vec3(85, 166, 78) / 255.0 * 0.5, vec3(2.2));
        voxel.albedo = hsv2rgb(vec3(0.18 + v * 0.1 + fract(r2 * 426.7) * 0.01, 0.8, 0.4));
        voxel.material_type = 1;
        voxel.roughness = 0.9;
        voxel.normal = nrm;

        // spawn strand!!

        // if (r2 < 0.2) {
        //     if (r2 < 0.99 * 0.2) {
        //         spawn_grass(voxel);
        //     } else {
        //         uint flower_type = 0;
        //         if (v < 0.4) {
        //             flower_type = FLOWER_TYPE_DANDELION;
        //         } else if (v < 0.5) {
        //             flower_type = FLOWER_TYPE_DANDELION_WHITE;
        //         } else if (v < 0.65) {
        //             flower_type = FLOWER_TYPE_TULIP;
        //         } else {
        //             flower_type = FLOWER_TYPE_LAVENDER;
        //         }
        //         spawn_flower(voxel, flower_type);
        //     }
        // }
    }
}

#define ENABLE_TREE_GENERATION 0

void brushgen_world_terrain(in out Voxel voxel) {
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
            if (val > -0.05 && upwards > 0.25) {
                voxel.albedo = vec3(0.26, 0.18, 0.10);
                if (r < 0.5) {
                    voxel.albedo.r *= 0.5;
                    voxel.albedo.g *= 0.5;
                    voxel.albedo.b *= 0.5;
                    voxel.roughness = 0.99;
                } else if (r < 0.52) {
                    voxel.albedo.r *= 1.5;
                    voxel.albedo.g *= 1.5;
                    voxel.albedo.b *= 1.5;
                    voxel.roughness = 0.95;
                }
            } else if (val < -0.01 && val > -0.07 && upwards > 0.2) {
                voxel.albedo = vec3(0.34, 0.30, 0.14);
                if (r < 0.5) {
                    voxel.albedo.r *= 0.75;
                    voxel.albedo.g *= 0.75;
                    voxel.albedo.b *= 0.75;
                }
                voxel.roughness = 0.85;
            } else {
                voxel.albedo = vec3(0.33, 0.30, 0.21);
                voxel.roughness = 0.9;
            }
        } else {
            voxel.albedo = vec3(0.75);
        }
    } else if (true) {
        vec4 grass_val4 = terrain_noise(voxel_pos - vec3(0, 0, VOXEL_SIZE));
        float grass_val = grass_val4.x;
        if (grass_val < 0.0) {
            // try_spawn_grass(voxel, nrm);
        } else if (ENABLE_TREE_GENERATION != 0) {
            // try_spawn_tree(voxel, forest_biome_color, nrm);
        }
    }
}

#define GEN_MODEL 0

void brush_fire(in out Voxel voxel) {
    float sd_base = FLT_MAX;
    float sd_flame = FLT_MAX;

    vec3 lantern_c = vec3(0); // brush_input.pos + brush_input.pos_offset;

    sd_base = sd_union(sd_base, sd_box(voxel_pos - lantern_c, vec3((VOXEL_SIZE * 3).xx, VOXEL_SIZE)));

    sd_flame = sd_union(sd_flame, sd_round_cone(voxel_pos - (lantern_c + vec3((VOXEL_SIZE * -0.5).xx, 0.2)), VOXEL_SIZE * 4, VOXEL_SIZE * 2, 0.4));
    float flame_rand = good_rand(voxel_pos);

    if (sd_base < 0) {
        voxel.material_type = 1;
        voxel.albedo = vec3(0.05, 0.05, 0.05);
        voxel.roughness = 0.9;
        voxel.normal = GENERATE_NORMAL;
    } else if (sd_flame < 0) {
        voxel.material_type = 3;
        voxel.albedo = vec3(0.95, 0.2 + floor((flame_rand + voxel_pos.z - lantern_c.z) * 2.0) * 0.05, 0.05);
        voxel.roughness = 0.3 + flame_rand * 0.3;
        voxel.normal = GENERATE_NORMAL;
        if (sd_flame > -VOXEL_SIZE) {
            spawn_fire_particle(voxel);
        }
    }
}
void brush_torch(in out Voxel voxel) {
    float sd_base = FLT_MAX;
    float sd_flame = FLT_MAX;

    vec3 lantern_c = vec3(0); // brush_input.pos + brush_input.pos_offset;

    sd_base = sd_union(sd_base, sd_box(voxel_pos - (lantern_c + vec3(0, 0, 1.0)), vec3((VOXEL_SIZE * 2.5).xx, VOXEL_SIZE)));
    sd_base = sd_union(sd_base, sd_box(voxel_pos - (lantern_c + vec3(0, 0, 0.5)), vec3((VOXEL_SIZE * 1.5).xx, 0.5)));

    sd_flame = sd_union(sd_flame, sd_round_cone(voxel_pos - (lantern_c + vec3(0, 0, 1.0 + VOXEL_SIZE * 2)), VOXEL_SIZE * 2.0, VOXEL_SIZE * 0.5, 0.2));
    float flame_rand = good_rand(voxel_pos);

    if (sd_base < 0) {
        voxel.material_type = 1;
        voxel.albedo = vec3(.22, .13, .05);
        voxel.roughness = 0.9;
        voxel.normal = GENERATE_NORMAL;
    } else if (sd_flame < 0) {
        voxel.material_type = 3;
        voxel.albedo = vec3(0.95, 0.15, 0.05);
        voxel.roughness = 0.3 + flame_rand * 0.3;
        voxel.normal = GENERATE_NORMAL;

        if (sd_flame > -VOXEL_SIZE) {
            spawn_fire_particle(voxel);
        }
    }
}

void sd_maple_branch(in out TreeSDFNrm val, in vec3 p, in vec3 origin, in vec3 dir, in float scl, float time) {
    float upwards_curl_factor = 0.2;
    vec3 bp0 = origin;
    for (uint segment_i = 0; segment_i < 4; ++segment_i) {
        vec3 bp1 = bp0 + dir * scl + vec3(0, 0, upwards_curl_factor);
        upwards_curl_factor += 0.2 + sin(time * 0.1 + segment_i * 37) * 0.1;
        float branch_dist = sd_capsule(p, bp0, bp1, 0.10);
        if (branch_dist < val.wood) {
            val.wood = branch_dist;
            val.wood_nrm = sd_capsule_normal(p, bp0, bp1);
        }
        bp0 = bp1;
        if (segment_i < 2)
            continue;
        float leaves_dist = sd_sphere(p - bp1, scl * 0.4 + 0.6);
        if (leaves_dist < val.leaves) {
            val.leaves = leaves_dist;
            val.leaves_nrm = normalize(p - bp1);
        }
        // val.leaves = sd_smooth_union(val.leaves, leaves_dist, 1.0);
    }
}

TreeSDFNrm sd_maple_tree(in vec3 p, in vec3 seed, float time) {
    TreeSDFNrm val = TreeSDFNrm(1e5, 1e5, GENERATE_NORMAL, GENERATE_NORMAL);

    float sd_trunk_base = sd_round_cone(
        p,
        vec3(0, 0, 0),
        vec3(0, 0, 5),
        0.5, 0.4);
    float sd_trunk_mid = sd_round_cone(
        p,
        vec3(0, 0, 5),
        vec3(0, 0, 8),
        0.4, 0.41);
    float sd_trunk_top = sd_round_cone(
        p,
        vec3(0, 0, 5),
        vec3(0, 0, 16),
        0.41, 0.2);

    val.wood = sd_union(sd_trunk_base, sd_union(sd_trunk_mid, sd_trunk_top));
    val.wood_nrm = normalize(vec3(p.xy, 0));

    for (uint i = 0; i < 7; ++i) {
        float scl = (1 - 0.05 * pow(i, 2)) * 0.02 * pow(i, 2) + 1.6 - i * 0.13;
        uint branch_n = 8 - i / 2;
        for (uint branch_i = 0; branch_i < branch_n; ++branch_i) {
            float angle = sin(time * 0.1 + branch_i) + (1.0 / branch_n * branch_i) * 2.0 * M_PI + good_rand(seed + i + 1.0 * branch_i) * 0.5 + branch_i * 10;
            float branch_base = 4.0 + i * 1.8 + branch_i * 0.1;
            vec3 dir = normalize(vec3(cos(angle), sin(angle), +0.0));
            sd_maple_branch(val, p, vec3(0, 0, branch_base), dir, scl, time);
        }
    }
    return val;
}

void brush_maple_tree(in out Voxel voxel) {
    vec3 tree_pos = vec3(0); // brush_input.pos + brush_input.pos_offset;

    float tree_size = good_rand(tree_pos);
    float space_scl = 1.5 - tree_size * 0.5;
    TreeSDFNrm tree = sd_maple_tree((voxel_pos - tree_pos) * space_scl, tree_pos, 0);
    tree.wood /= space_scl;
    tree.leaves /= space_scl;

    float leaf_rand = good_rand(voxel_pos);

    uint prev_mat_type = voxel.material_type;

    if (tree.wood < 0) {
        voxel.material_type = 1;
        voxel.albedo = vec3(.22, .13, .05);
        voxel.roughness = 0.99;
        voxel.normal = tree.wood_nrm;
    } else if (tree.leaves * 5.0 + leaf_rand * 15.0 < 0) {
        voxel.material_type = 1;
        voxel.albedo = vec3(.28, .8, .15) * 0.5;
        // voxel.albedo = hsv2rgb(vec3(0.0 + good_rand(tree_pos) * 0.05, 0.9, 0.9));
        voxel.roughness = 0.95;
        voxel.normal = tree.leaves_nrm;
        if (tree.leaves - leaf_rand > -VOXEL_SIZE) {
            // should be a particle spawner
            // voxel.albedo = vec3(.9, .1, .9);
            if (prev_mat_type == 0) {
                spawn_tree_particle(voxel);
            }
        }
    }
}

void brush_spruce_tree(in out Voxel voxel) {
    vec3 tree_pos = vec3(0); // brush_input.pos + brush_input.pos_offset;

    TreeSDFNrm tree = sd_spruce_tree(voxel_pos - tree_pos, tree_pos);
    float leaf_rand = good_rand(voxel_pos);

    if (tree.wood < 0) {
        voxel.material_type = 1;
        voxel.albedo = vec3(.22, .13, .05);
        voxel.roughness = 0.99;
        voxel.normal = GENERATE_NORMAL;
    } else if (tree.leaves + leaf_rand * 0.05 < 0) {
        voxel.material_type = 1;
        voxel.albedo = vec3(.14, .2, .07);
        voxel.roughness = 0.95;
        voxel.normal = tree.leaves_nrm;
        if (dot(voxel.normal, vec3(0, 0, 1)) - leaf_rand * 1.0 > 0.0) {
            voxel.albedo = vec3(0.7, 0.7, 0.71);
        }
    }
}

void sd_spruce_tree_big_branch(in out TreeSDFNrm val, in vec3 p, in vec3 origin, in vec3 dir, in float scl) {
    float upwards_curl_factor = 0.1;
    vec3 bp0 = origin;
    for (uint segment_i = 0; segment_i < 4; ++segment_i) {
        vec3 bp1 = bp0 + dir * scl + vec3(0, 0, upwards_curl_factor);
        upwards_curl_factor += 0.1;
        val.wood = sd_union(val.wood, sd_capsule(p, bp0, bp1, 0.10));
        for (uint i = 0; i < 3; ++i) {
            vec3 needle_p = mix(bp0, bp1, i * 0.3);
            {
                float leaves_dist = sd_capsule(p, needle_p, needle_p + (dir.yxz * vec3(-1, 1, 1) + dir * 0.6) * (4 - segment_i) * 0.4, 0.1);
                val.leaves = sd_union(val.leaves, leaves_dist);
            }
            {
                float leaves_dist = sd_capsule(p, needle_p, needle_p + (-dir.yxz * vec3(-1, 1, 1) + dir * 0.6) * (4 - segment_i) * 0.4, 0.1);
                val.leaves = sd_union(val.leaves, leaves_dist);
            }
        }
        bp0 = bp1;
    }
}

TreeSDFNrm sd_spruce_tree_big(in vec3 p, in vec3 seed) {
    TreeSDFNrm val = TreeSDFNrm(1e5, 1e5, GENERATE_NORMAL, GENERATE_NORMAL);

    float sd_trunk_base = sd_round_cone(
        p,
        vec3(0, 0, 0),
        vec3(0, 0, 5),
        0.5, 0.4);
    float sd_trunk_mid = sd_round_cone(
        p,
        vec3(0, 0, 5),
        vec3(0, 0, 8),
        0.4, 0.41);
    float sd_trunk_top = sd_round_cone(
        p,
        vec3(0, 0, 5),
        vec3(0, 0, 16),
        0.41, 0.2);

    val.wood = sd_union(sd_trunk_base, sd_union(sd_trunk_mid, sd_trunk_top));

    for (uint i = 0; i < 9; ++i) {
        float scl = 1.5 - i * 0.15;
        uint branch_n = 11 - i / 2;
        for (uint branch_i = 0; branch_i < branch_n; ++branch_i) {
            float angle = (1.0 / branch_n * branch_i) * 2.0 * M_PI + good_rand(seed + i + 1.0 * branch_i) * 15.5 + branch_i * 10;
            float branch_base = 4.0 + i * 1.4 + branch_i * 0.1;
            vec3 dir = normalize(vec3(cos(angle), sin(angle), +0.0));
            sd_spruce_tree_big_branch(val, p, vec3(0, 0, branch_base), dir, scl);
        }
    }
    return val;
}

void brush_spruce_tree_big(in out Voxel voxel) {
    vec3 tree_pos = vec3(0); // brush_input.pos + brush_input.pos_offset;

    float tree_size = good_rand(tree_pos);
    float space_scl = 1.5 - tree_size * 0.5;
    TreeSDFNrm tree = sd_spruce_tree_big((voxel_pos - tree_pos) * space_scl, tree_pos);
    tree.wood /= space_scl;
    tree.leaves /= space_scl;

    float leaf_rand = good_rand(voxel_pos);

    uint prev_mat_type = voxel.material_type;

    if (tree.wood < 0) {
        voxel.material_type = 1;
        voxel.albedo = vec3(.22, .13, .05);
        voxel.roughness = 0.99;
        voxel.normal = vec3(0, 0, -1);
    } else if (tree.leaves * 5.0 < 0) {
        voxel.material_type = 1;
        // voxel.albedo = vec3(.28, .8, .15) * 0.5;
        voxel.albedo = hsv2rgb(vec3(0.35 + good_rand(tree_pos) * 0.03, 0.4, 0.2));
        voxel.roughness = 0.95;
        voxel.normal = tree.leaves_nrm;
        if (dot(voxel.normal, vec3(0, 0, 1)) > 0.6) {
            voxel.albedo = vec3(0.9);
        }
    }
}

// A sample of the closest fern surface found so far: distance, analytic
// normal (from the capsule that produced it), and material properties.
struct FernSample {
    float dist;
    vec3 nrm;
    vec3 albedo;
    float roughness;
};

// Analytic gradient of sd_capsule: the direction from the capsule's medial
// axis to the query point. Using this (rather than GENERATE_NORMAL, which is
// just a flat placeholder) gives every stem/rib/leaflet a proper rounded
// per-voxel normal.

void fern_add_capsule(in out FernSample val, in vec3 p, in vec3 a, in vec3 b, in float r, in vec3 albedo, in float roughness) {
    float d = sd_capsule(p, a, b, r);
    if (d < val.dist) {
        val.dist = d;
        val.nrm = sd_capsule_normal(p, a, b);
        val.albedo = albedo;
        val.roughness = roughness;
    }
}

// Procedural waving fern: a curved central stem with alternating fronds
// branching off of it, each frond itself a curved midrib carrying pairs of
// small leaflets (pinnae). Everything is built out of capsules so normals
// can be computed analytically. `time` drives a wind sway that increases
// with height/distance-from-stem, so tips move more than roots.
//
// `loop_t` is the animation phase as a fraction of one full loop (0..1, and
// loop_t=0 must look identical to loop_t=1 for the bake to cycle cleanly).
// To guarantee that, every sway term is built from sin/cos of
// 2*PI*N*loop_t + (a loop_t-independent phase offset) for some integer N
// (whole sway cycles per loop) -- never from a continuous, unbounded time
// value, which would only line back up at the seam by coincidence.
FernSample sd_fern(in vec3 p, in vec3 seed, in float loop_t) {
    FernSample val;
    val.dist = 1e5;
    val.nrm = GENERATE_NORMAL;
    val.albedo = vec3(0.10, 0.35, 0.08);
    val.roughness = 0.9;

    vec3 stem_col = vec3(0.04, 0.2, 0.02) * 3;

    const uint STEM_SEGMENTS = 20;
    const float stem_height = 7.0;
    const float seg_len = stem_height / float(STEM_SEGMENTS);

    float phase = 2.0 * M_PI * loop_t;

    vec3 bp0 = vec3(0);
    vec3 dir = vec3(0, 0, 1);

    const float thickness = 14;

    for (uint i = 0; i < STEM_SEGMENTS; ++i) {
        float t = float(i) / float(STEM_SEGMENTS - 1);

        // Wind sway: bigger amplitude and lower apparent "stiffness" toward the tip.
        // One full sway cycle per loop (integer N=1), so it seams cleanly.
        float sway = sin(phase + t * 2.2) * 0.038 * t * t;
        float sway_perp = cos(phase + t * 1.7) * 0.032 * t * t;
        vec3 seg_dir = normalize(dir + vec3(sway, sway_perp, 0.0));
        vec3 bp1 = bp0 + seg_dir * seg_len;

        float stem_r = mix(0.08, 0.03, t) * thickness;
        fern_add_capsule(val, p, bp0, bp1, stem_r, stem_col, 0.85);

        // Branch off a frond partway up the stem, alternating left/right.
        if (i >= 1 && i + 1 < STEM_SEGMENTS) {
            float side = ((i & 1u) == 0u) ? 1.0 : -1.0;
            float rand_a = good_rand(seed + float(i) * 3.1);
            float rand_b = i * M_PI * 2; // good_rand(seed + float(i) * 7.3 + 1.0);

            // Fronds are longest in the middle of the stem, short near the base and tip.
            float frond_len = mix(3.5, 8.5, sin(t * M_PI)) * (0.85 + 0.3 * rand_a);
            float base_angle = 0.85 + rand_b * 10.35;
            float droop = 0.35 + 0.35 * t;

            vec3 rib_dir = normalize(vec3(cos(base_angle) * side, sin(base_angle) * side, 0.35));
            vec3 rp0 = bp1;

            const uint RIB_SEGMENTS = 8;
            float rib_seg_len = frond_len / float(RIB_SEGMENTS);

            for (uint j = 0; j < RIB_SEGMENTS; ++j) {
                float jt = float(j) / float(RIB_SEGMENTS - 1);

                // The frond sways too, with extra phase offset per-branch and per-segment
                // (all loop_t-independent, so still N=1 per loop) so fronds don't all
                // move in lockstep.
                float frond_sway = sin(phase + t * 4.0 + jt * 1.6 + rand_a * 6.0) * (0.15 + 0.35 * jt) * side * 0.3;
                float frond_sway_z = cos(phase + t * 3.0 + jt * 1.2 + rand_b * 6.0) * 0.03 * jt;

                rib_dir = normalize(rib_dir + vec3(frond_sway * 0.25, frond_sway * 0.15, -droop / float(RIB_SEGMENTS) + frond_sway_z));
                vec3 rp1 = rp0 + rib_dir * rib_seg_len;

                float rib_r = mix(0.05, 0.008, jt) * thickness;
                fern_add_capsule(val, p, rp0, rp1, rib_r, stem_col * 0.9, 0.85);

                // Leaflets (pinnae): a symmetric pair sprouting off this rib segment,
                // shrinking toward the frond tip.
                vec3 up_ish = abs(rib_dir.z) < 0.95 ? vec3(0, 0, 1) : vec3(1, 0, 0);
                vec3 perp = normalize(cross(rib_dir, up_ish));
                float leaflet_len = mix(1.0, 0.12, jt) * (0.6 + 0.4 * (1.0 - abs(t - 0.5) * 2.0)) * 2.4;

                for (float sgn = -1.0; sgn <= 1.0; sgn += 2.0) {
                    vec3 lp0 = mix(rp0, rp1, 0.5);
                    vec3 leaflet_dir = normalize(perp * sgn + vec3(0, 0, -0.25) + rib_dir * 0.25);
                    vec3 lp1 = lp0 + leaflet_dir * leaflet_len;

                    float leaf_rand = good_rand(seed + float(i * 13u + j * 7u) + sgn * 1.7) * 0.5 + 0.5;
                    vec3 leaf_col = vec3(0.02, 0.1, 0.02) * leaf_rand * 3; // mix(vec3(0.02, 0.2, 0.02), vec3(0.10, 0.45, 0.10), leaf_rand);
                    fern_add_capsule(val, p, lp0, lp1, mix(0.045, 0.022, jt) * thickness, leaf_col, 0.9);
                }

                rp0 = rp1;
            }
        }

        bp0 = bp1;
        dir = seg_dir;
    }

    return val;
}

void brush_fern(in out Voxel voxel, in vec3 seed, in float loop_t) {
    FernSample fern = sd_fern(voxel_pos, seed, loop_t);
    if (fern.dist < 0.0) {
        voxel.material_type = 1;
        voxel.albedo = fern.albedo;
        voxel.roughness = fern.roughness;
        voxel.normal = normalize(fern.nrm + vec3(0, 0, 3));
    }
}
