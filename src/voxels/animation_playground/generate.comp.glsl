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

    uvec3 local_voxel = gl_LocalInvocationID.xyz;
    uint voxel_index = local_voxel.x + local_voxel.y * BRICK_SIZE + local_voxel.z * BRICK_SIZE * BRICK_SIZE;
    uvec3 voxel_i = brick_i * BRICK_SIZE + local_voxel;
    // Centered on x/y, resting on the floor of the grid (z=0).
    vec3 p = vec3(voxel_i) + 0.5 - vec3(grid_center.xy, 0);

    // Animation phase as a fraction of one full loop (0 at frame 0, wrapping
    // back to 0 at frame_count). sd_fern builds its sway purely from this
    // fraction (whole cycles per loop), so frame_count-1 -> 0 is a seamless cut.
    float loop_t = float(frame_index) / float(push.frame_count);
    // Fixed shape seed: only the animation phase (loop_t) should change the
    // result frame-to-frame, not the branch layout.
    vec3 shape_seed = vec3(17.0, 3.0, 41.0);

    voxel_pos = p * 0.4;
    Voxel voxel = Voxel(vec3(0), vec3(0, 0, 1), 0.9, 0u);
    brush_fern(voxel, shape_seed, loop_t);
    bool solid = voxel.material_type != 0u;
    voxel.material_type = 0;

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