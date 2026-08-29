#ifndef VOXELS_ANIMATION_PLAYGROUND_ANIMATION_PLAYGROUND_INL
#define VOXELS_ANIMATION_PLAYGROUND_ANIMATION_PLAYGROUND_INL

#include <voxels/voxel.inl>

struct AnimationPlaygroundGenPush {
    daxa_RWBufferPtr(BrickPrimitive) bricks;
    daxa_RWBufferPtr(VoxelShadingAttribBrick) brick_attribs;
    daxa_i32vec3 grid_dims_bricks;
    daxa_u32 frame_count;
    daxa_f32 time;
};

#endif // VOXELS_ANIMATION_PLAYGROUND_ANIMATION_PLAYGROUND_INL
