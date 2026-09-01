#include "animation_playground.inl"
#include <voxels/pack_unpack.inl>

DAXA_DECL_PUSH_CONSTANT(AnimationPlaygroundGenPush, push)

#include "../brushes.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

shared bool shared_solid[512];

// Rotate around a coordinate axis (i.e. in a plane perpendicular to that axis) by angle <a>.
void rotate2d(inout vec2 p, float a) {
    p = cos(a) * p + sin(a) * vec2(p.y, -p.x);
}

// Repeat around the origin by a fixed angle. Num of repetitions is used to specify the angle.
// Returns the index of the cell that <p> ended up in.
float mod_polar(inout vec2 p, float repetitions) {
    float angle = 2.0 * M_PI / repetitions;
    float a = atan(p.y, p.x) + angle / 2.;
    float r = length(p);
    float c = floor(a / angle);
    a = mod(a, angle) - angle / 2.;
    p = vec2(cos(a), sin(a)) * r;
    // For an odd number of repetitions, fix cell index of the cell in -x direction
    // (cell index would be e.g. -5 and 5 in the two halves of the cell):
    if (abs(c) >= (repetitions / 2.0))
        c = abs(c);
    return c;
}

const float MAT_OAK_LEAF = 0.0;
const float MAT_OAK_BARK = 2.0;

float sd_oak_root(vec3 pos, float len, float rad, float rand, out vec4 material, out vec3 root_space) {
    float root_half_len = len * 0.5;
    float progress_along = pos.y / (2.0 * root_half_len);
    float root_rad = rad * (1.0 - progress_along * 0.8);

    float wave = sin((rand + pos.y) / len * 15.0) * 0.25 * rad;
    pos.xz += vec2(wave);
    pos.x -= progress_along * progress_along * 10;
    float min_dist = sd_capsule(pos, vec3(0.0), vec3(0.0, len, 0.0), root_rad);
    root_space = pos;

    float u = atan(pos.z, pos.x);
    material = vec4(MAT_OAK_BARK, 0.0 /* overridden for AO */, u, progress_along);

    return min_dist;
}

float sd_oak_root(vec3 pos_ws, float scale, float rand, out vec4 material, out vec3 root_space) {
    float root_len = 1.0 * scale;
    float root_rad = 0.08 * scale;

    return sd_oak_root(pos_ws, root_len, root_rad, rand, material, root_space);
}

float sd_oak_branch(vec3 pos, float len, float rad, float rand, out vec4 material, out vec3 branch_space) {
    float branch_half_len = len * 0.5;
    float progress_along = pos.y / (2.0 * branch_half_len);
    float branch_rad = rad * (1.0 - progress_along * 0.8);

    float wave = sin((rand + pos.y) / len * 10.0) * 0.25 * rad;
    pos.xz += vec2(wave);
    float min_dist = sd_capsule(pos, vec3(0.0), vec3(0.0, len, 0.0), branch_rad);
    branch_space = pos;

    float u = atan(pos.z, pos.x);
    material = vec4(MAT_OAK_BARK, 0.0 /* overridden for AO */, u, progress_along);

    return min_dist;
}

float sd_oak_branch(vec3 pos_ws, float scale, float rand, out vec4 material, out vec3 branch_space) {
    float branch_len = 1.0 * scale;
    float branch_rad = 0.07 * scale;

    return sd_oak_branch(pos_ws, branch_len, branch_rad, rand, material, branch_space);
}

// Returns the rotation angle `a` (matching rotate2d(p, a)'s convention) that turns `before` into
// `after`, given |before| == |after| (a pure rotation of a 2D vector, e.g. what mod_polar/rotate2d
// do to a point). Used to undo those domain rotations when carrying a normal back out of
// branch-local space.
float rotation_angle_between(vec2 before, vec2 after) {
    return atan(before.y * after.x - before.x * after.y, dot(before, after));
}

// One leaf bundle, as a single outward-facing sphere, placed at the tip of a branch.
//
// `branch_pos` is the point already carried into the branch's local space by sd_oak_tree:
// trunk_pos.xz is rotated (about Y) by mod_polar into one of several angular slices, then the whole
// point is translated down along Y and rotated (about Z) by `branch_tilt_angle` via
// rotate2d(branch_pos.xy, ...). `pre_xz`/`post_xz` are trunk_pos.xz before/after that mod_polar
// call, which we use to recover the (otherwise-undocumented) angle mod_polar picked, so the leaf
// normal can be rotated back out of branch-local space into tree space: undo the Y-axis tilt first
// (last transform applied), then undo the mod_polar rotation.
float sd_oak_leaves(vec3 branch_pos, float branch_len, vec2 pre_xz, vec2 post_xz, float branch_tilt_angle, out vec4 material, out vec3 leaf_normal) {
    const float bundle_radius = 1.6;

    vec3 leaves_pos = branch_pos - vec3(0, branch_len, 0);
    float leaves_dist = sd_sphere(leaves_pos, bundle_radius);

    leaves_dist = min(leaves_dist, sd_sphere(leaves_pos - vec3(0, 0, 1.2), bundle_radius * 0.9));
    leaves_dist = min(leaves_dist, sd_sphere(leaves_pos - vec3(0, 0, -1.2), bundle_radius * 0.8));
    leaves_dist = min(leaves_dist, sd_sphere(leaves_pos - vec3(0, 1, 0), bundle_radius * 0.7));
    leaves_dist = min(leaves_dist, sd_sphere(leaves_pos - vec3(1, 0, 0), bundle_radius * 0.7));

    vec3 n = normalize(leaves_pos);
    vec2 nxy = n.xy;
    rotate2d(nxy, -branch_tilt_angle);
    n.xy = nxy;

    float polar_angle = rotation_angle_between(pre_xz, post_xz);
    vec2 nxz = n.xz;
    rotate2d(nxz, -polar_angle);
    n.xz = nxz;

    leaf_normal = n;
    material = vec4(MAT_OAK_LEAF, 1.0, 0.0, 0.0);

    return leaves_dist;
}

float sd_oak_tree(vec3 pos_tree_space, float loop_t, out vec4 material, out vec3 leaf_normal, out vec3 min_tree_space) {
    pos_tree_space.y -= 1.5;
    float min_dist = MAX_DIST;
    float tree_bounding_sphere_dist = sd_sphere(pos_tree_space - vec3(0.0, 7.0, 0.0), 9.5);

    // If we're far from the tree, bail early
    if (tree_bounding_sphere_dist > 12.0) {
        return tree_bounding_sphere_dist - 10.0;
    }
    if (tree_bounding_sphere_dist > 1.0) {
        return min(min_dist, tree_bounding_sphere_dist);
    }

    vec4 trunk_material;
    vec3 trunk_pos = pos_tree_space;

    vec3 trunk_space = pos_tree_space;
    float trunk_dist = sd_oak_branch(trunk_pos, 10.0, 1, 0.0, trunk_material, trunk_space);

    FractalNoiseConfig wind_noise_conf = FractalNoiseConfig(
        /* .amplitude   = */ 1.0,
        /* .persistance = */ 0.3,
        /* .scale       = */ 1.,
        /* .lacunarity  = */ 2.5,
        /* .octaves     = */ 1);
    vec4 wind_noise_val = fractal_noise(g_value_noise_tex, g_sampler_llr, pos_tree_space + vec3(sin(loop_t * M_PI) * 0.5, cos(loop_t * M_TAU), 0), wind_noise_conf);
    float wind_noise = (wind_noise_val.x - 0.2) * 0.015 * max(trunk_dist-1.5,0) * max(trunk_dist-1.5,0);

    if (trunk_dist < min_dist) {
        min_dist = trunk_dist;
        material = trunk_material;
        min_tree_space = trunk_space;
    }

    float min_branch_dist = MAX_DIST;
    vec4 min_branch_material;

    vec4 branch_material;
    vec3 branch_pos;
    float branch_dist, id, rand;
    vec2 pre_xz, post_xz;
    float branch_tilt_angle;
    vec4 leaf_material;
    vec3 leaf_normal_local;
    float leaf_dist;

    const float wind_branch_tilt = 0.007;
    const float smooth_blend = 0.5;

    branch_pos = trunk_pos;
    branch_pos += vec3(wind_noise, 0, 0);
    pre_xz = branch_pos.xz;
    id = mod_polar(branch_pos.xz, 6.0);
    post_xz = branch_pos.xz;
    rand = good_rand(id * 736.884);
    branch_pos.y -= 4.0 + 1.0 * rand;
    branch_tilt_angle = -M_PI * (0.32 + rand * 0.1) + sin(loop_t * M_TAU + rand * 179) * wind_branch_tilt;
    rotate2d(branch_pos.xy, branch_tilt_angle);

    vec3 branch_space = pos_tree_space;
    branch_dist = sd_oak_branch(branch_pos, 5.75, rand, branch_material, branch_space);
    if (branch_dist < min_branch_dist) {
        min_branch_material = branch_material;
        if (branch_dist < min_dist)
            min_tree_space = branch_space;
    }
    min_branch_dist = sd_smooth_union(branch_dist, min_branch_dist, smooth_blend);

    FractalNoiseConfig leaf_noise_conf = FractalNoiseConfig(
        /* .amplitude   = */ 1.0,
        /* .persistance = */ 0.2,
        /* .scale       = */ 7.5,
        /* .lacunarity  = */ 4.5,
        /* .octaves     = */ 1);

    float leaf_rand = 0;
    const float leaf_density = 1;
    const float leaf_randomness = 3;

    const float leaf_canopy_normal_strenth = 0.4;
    const float leaf_ball_normal_strength = 1;

    leaf_rand = fractal_noise(g_value_noise_tex, g_sampler_llr, branch_pos, leaf_noise_conf).x;
    leaf_dist = sd_oak_leaves(branch_pos, 5.75, pre_xz, post_xz, branch_tilt_angle, leaf_material, leaf_normal_local);
    if (leaf_dist * leaf_density + leaf_rand * leaf_randomness < 0 && leaf_dist < min_dist) {
        min_dist = leaf_dist;
        material = leaf_material;
        leaf_normal = leaf_normal_local;
        leaf_normal = normalize(normalize(pos_tree_space) * leaf_canopy_normal_strenth + leaf_normal * leaf_ball_normal_strength);
    }

    branch_pos = trunk_pos;
    pre_xz = branch_pos.xz;
    id = mod_polar(branch_pos.xz, 5.0);
    post_xz = branch_pos.xz;
    rand = good_rand(id * 736.884);
    branch_pos.y -= 0.8 + 1.0 * rand;
    branch_tilt_angle = -M_PI * (0.32 + rand * 0.1) - 0.8;
    rotate2d(branch_pos.xy, branch_tilt_angle);

    branch_space = pos_tree_space;
    branch_dist = sd_oak_root(branch_pos, 8.75, rand, branch_material, branch_space);
    if (branch_dist < min_branch_dist) {
        min_branch_material = branch_material;
        if (branch_dist < min_dist)
            min_tree_space = branch_space;
    }
    min_branch_dist = sd_smooth_union(branch_dist, min_branch_dist, smooth_blend);

    branch_pos = trunk_pos;
    branch_pos += vec3(wind_noise, 0, 0);
    pre_xz = branch_pos.xz;
    rotate2d(branch_pos.xz, -M_PI * 0.35);
    id = mod_polar(branch_pos.xz, 5.0);
    post_xz = branch_pos.xz;
    rand = good_rand(id * 736.884);
    branch_pos.y -= 7.5 + 1.0 * rand;
    branch_tilt_angle = -M_PI * (0.35 - rand * 0.05) + sin(loop_t * M_TAU + rand * 179) * wind_branch_tilt * 1.5;
    rotate2d(branch_pos.xy, branch_tilt_angle);

    branch_dist = sd_oak_branch(branch_pos, 5.0, 0.0, branch_material, branch_space);
    if (branch_dist < min_branch_dist) {
        min_branch_material = branch_material;
        if (branch_dist < min_dist)
            min_tree_space = branch_space;
    }
    min_branch_dist = sd_smooth_union(branch_dist, min_branch_dist, smooth_blend);

    leaf_rand = fractal_noise(g_value_noise_tex, g_sampler_llr, branch_pos, leaf_noise_conf).x;
    leaf_dist = sd_oak_leaves(branch_pos, 5.0, pre_xz, post_xz, branch_tilt_angle, leaf_material, leaf_normal_local);
    if (leaf_dist * leaf_density + leaf_rand * leaf_randomness < 0 && leaf_dist < min_dist) {
        min_dist = leaf_dist;
        material = leaf_material;
        leaf_normal = leaf_normal_local;
        leaf_normal = normalize(normalize(pos_tree_space) * leaf_canopy_normal_strenth + leaf_normal * leaf_ball_normal_strength);
    }

    branch_pos = trunk_pos;
    branch_pos += vec3(wind_noise, 0, 0);
    pre_xz = branch_pos.xz;
    rotate2d(branch_pos.xz, -M_PI * 0.65);
    id = mod_polar(branch_pos.xz, 3.0);
    post_xz = branch_pos.xz;
    rand = good_rand(id * 736.884);
    branch_pos.y -= 9.5 + 0.5 * rand;
    branch_tilt_angle = -M_PI * (0.22 - 0.1 * rand) + sin(loop_t * M_TAU + rand * 179) * wind_branch_tilt * 2;
    rotate2d(branch_pos.xy, branch_tilt_angle);

    branch_dist = sd_oak_branch(branch_pos, 4.0, 0.0, branch_material, branch_space);
    if (branch_dist < min_branch_dist) {
        min_branch_material = branch_material;
        min_tree_space = branch_space;
    }
    min_branch_dist = sd_smooth_union(branch_dist, min_branch_dist, smooth_blend);

    leaf_rand = fractal_noise(g_value_noise_tex, g_sampler_llr, branch_pos, leaf_noise_conf).x;
    leaf_dist = sd_oak_leaves(branch_pos, 4.0, pre_xz, post_xz, branch_tilt_angle, leaf_material, leaf_normal_local);
    if (leaf_dist * leaf_density + leaf_rand * leaf_randomness < 0 && leaf_dist < min_dist) {
        min_dist = leaf_dist;
        material = leaf_material;
        leaf_normal = leaf_normal_local;
        leaf_normal = normalize(normalize(pos_tree_space) * leaf_canopy_normal_strenth + leaf_normal * leaf_ball_normal_strength);
    }

    if (min_branch_dist < min_dist) {
        material = min_branch_material;
    }
    min_dist = sd_smooth_union(min_dist, min_branch_dist, smooth_blend);

    // Ambient occlusion is stronger at the center of the tree
    vec3 pos_to_canopy_center = vec3(0.0, 10.0, 0.0) - pos_tree_space;
    material.y = min(1.0, dot(pos_to_canopy_center, pos_to_canopy_center) / 36.0);

    return min_dist;
}

vec3 sd_oak_tree_normal(vec3 p, float loop_t, float dt) {
    vec3 normal_ws = vec3(0.0);
    for (int i = 0; i < 4; i++) {
        vec3 e = 0.5773 * (2.0 * vec3((((i + 3) >> 1) & 1), ((i >> 1) & 1), (i & 1)) - 1.0);
        vec3 sample_pos_ws = p + e * dt;
        vec4 mat;
        vec3 leaf_nrm;
        vec3 tree_space;
        float dist = sd_oak_tree(sample_pos_ws, loop_t, mat, leaf_nrm, tree_space);
        normal_ws += e * dist;
    }
    return normalize(normal_ws);
}

void main() {
    ivec3 brick_i = ivec3(gl_WorkGroupID.xyz);
    int frame_index = brick_i.z / push.grid_dims_bricks.z;
    brick_i.z = brick_i.z % push.grid_dims_bricks.z;

    int bricks_per_frame = push.grid_dims_bricks.x * push.grid_dims_bricks.y * push.grid_dims_bricks.z;
    int brick_index_in_frame = brick_i.x + brick_i.y * push.grid_dims_bricks.x + brick_i.z * push.grid_dims_bricks.x * push.grid_dims_bricks.y;
    int brick_index = frame_index * bricks_per_frame + brick_index_in_frame;

    daxa_RWBufferPtr(BrickPrimitive) brick_ptr = advance(push.bricks, brick_index);
    daxa_RWBufferPtr(VoxelShadingAttribBrick) attribs_ptr = advance(push.brick_attribs, brick_index);

    ivec3 grid_voxel_dims = push.grid_dims_bricks * BRICK_SIZE;
    vec3 grid_center = vec3(grid_voxel_dims) * 0.5;

    uvec3 local_voxel = gl_LocalInvocationID.xyz;
    uint voxel_index = local_voxel.x + local_voxel.y * BRICK_SIZE + local_voxel.z * BRICK_SIZE * BRICK_SIZE;
    uvec3 voxel_i = brick_i * BRICK_SIZE + local_voxel;
    // Centered on x/y, resting on the floor of the grid (z=0).
    vec3 p = vec3(voxel_i) + 0.5 - vec3(grid_center.xy, 0);
    Voxel voxel = Voxel(vec3(0), vec3(0, 0, 1), 0.9, 0u);

    // Animation phase as a fraction of one full loop (0 at frame 0, wrapping
    // back to 0 at frame_count). sd_fern builds its sway purely from this
    // fraction (whole cycles per loop), so frame_count-1 -> 0 is a seamless cut.
    float loop_t = float(frame_index) / float(push.frame_count);

    // // Fixed shape seed: only the animation phase (loop_t) should change the
    // // result frame-to-frame, not the branch layout.
    // vec3 shape_seed = vec3(17.0, 3.0, 41.0);
    // voxel_pos = p * 0.4;
    // brush_fern(voxel, shape_seed, loop_t);

    voxel_pos = p * VOXEL_SIZE;
    bool solid = voxel.material_type != 0u;
    voxel.material_type = 0;

    vec4 tree_material;
    vec3 tree_leaf_normal;
    vec3 tree_space;
    float tree_dist = sd_oak_tree(voxel_pos.xzy, loop_t, /*out*/ tree_material, /*out*/ tree_leaf_normal, /*out*/ tree_space);
    if (tree_dist < 0) {
        solid = true;
        if (tree_material.x == MAT_OAK_LEAF) {
            voxel.albedo = vec3(0.045, 0.156, 0.032);
            voxel.normal = tree_leaf_normal.xzy;
        }
        if (tree_material.x == MAT_OAK_BARK) {
            voxel.normal = sd_oak_tree_normal(voxel_pos.xzy, loop_t, 0.001).xzy;
            voxel.albedo = vec3(0.102, 0.070, 0.045) * 2;

            float angle = tree_material.z * 8 + good_rand(tree_space);
            vec3 wavy_nrm = abs(vec3(sin(angle), cos(angle), 0));
            vec4 dnrm = sd_analytical_fractal_noise(vec3(tree_space * 20 * vec3(1, 1, 4)));
            voxel.normal = normalize(voxel.normal + dnrm.yzw * 15 + wavy_nrm * 0.3);
        }
    }

    deref(attribs_ptr).voxels[voxel_index] = pack_voxel(voxel);
    shared_solid[voxel_index] = solid;
    memoryBarrierShared();
    barrier();

    if (gl_LocalInvocationIndex < BRICK_SIZE * BRICK_SIZE) {
        uint z = gl_LocalInvocationIndex / BRICK_SIZE;
        uint y = gl_LocalInvocationIndex % BRICK_SIZE;
        uint local_byte = 0;
        for (uint x = 0; x < BRICK_SIZE; ++x) {
            uint voxel_index = x + y * BRICK_SIZE + z * BRICK_SIZE * BRICK_SIZE;
            if (shared_solid[voxel_index]) {
                local_byte |= (1u << x);
            }
        }
        deref(brick_ptr).bitmap[z * BRICK_SIZE + y] = uint8_t(local_byte);
    }
}
