#include "animation_playground.inl"
#include <voxels/pack_unpack.inl>

DAXA_DECL_PUSH_CONSTANT(AnimationPlaygroundGenPush, push)

#include "../brushes.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

shared bool shared_solid[512];

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
    float max_radius = float(min(min(grid_voxel_dims.x, grid_voxel_dims.y), grid_voxel_dims.z)) * 0.5;

    uvec3 local_voxel = gl_LocalInvocationID.xyz;
    uint voxel_index = local_voxel.x + local_voxel.y * BRICK_SIZE + local_voxel.z * BRICK_SIZE * BRICK_SIZE;
    uvec3 voxel_i = brick_i * BRICK_SIZE + local_voxel;
    vec3 p = vec3(voxel_i) + 0.5 - vec3(grid_center.xy, 0);

    // float radius = max_radius * (0.5 + 0.4 * sin(float(frame_index) * 0.8));
    // bool solid = length(p) < radius;
    bool solid = false;

    vec3 nrm = normalize(p + vec3(1.0e-5, 0.0, 0.0));
    // vec3 col = nrm * 0.5 + 0.5;
    vec3 col = vec3(1, 0, 0);
    voxel_pos = p;

    vec3 tree_pos = vec3(0);
    float space_scl = 1.0 / 13.0;
    TreeSDFNrm tree = sd_maple_tree((voxel_pos - tree_pos) * space_scl, tree_pos, float(frame_index));
    tree.wood /= space_scl;
    tree.leaves /= space_scl;

    Voxel voxel = Voxel(col, nrm, 0.6, 0u);
    float leaf_rand = good_rand(voxel_pos);

    if (tree.wood < 0) {
        voxel.material_type = 1;
        voxel.albedo = vec3(.22, .13, .05);
        voxel.roughness = 0.99;
        voxel.normal = vec3(0, 0, 1);
        solid = true;
    } else if (tree.leaves * 5.0 + leaf_rand * 15.0 < 0) {
        voxel.material_type = 1;
        // voxel.albedo = vec3(.28, .8, .15) * 0.5;
        voxel.albedo = hsv2rgb(vec3(0.0 + good_rand(tree_pos) * 0.05, 0.9, 0.9));
        voxel.roughness = 0.95;
        voxel.normal = tree.leaves_nrm;
        solid = true;
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