#pragma once

#include <daxa/utils/task_graph.inl>
#include <voxels/voxel.inl>
#include <voxels/particles/grass/grass.inl>
#include <voxels/particles/flower/flower.inl>

// Spawns one piece of foliage for every set bit in a visible brick's foliage
// bitmask: one 8x8x8 workgroup per brick, reading `foliage_bricks` and
// malloc'ing into either `grass_allocator` or `flower_allocator` depending on a
// per-voxel noise roll. Both heaps are rebuilt from scratch every frame.

struct FoliageBrickInstance {
    daxa_BufferPtr(GpuVoxelObject) voxel_object;
    daxa_u32 brick_index;
    daxa_u32 _pad;
};
DAXA_DECL_BUFFER_PTR(FoliageBrickInstance);

struct FoliageGeneratePush {
    daxa_RWBufferPtr(FoliageBrickInstance) visible_foliage_bricks;
    daxa_RWBufferPtr(GrassStrandAllocator) grass_allocator;
    daxa_RWBufferPtr(FlowerAllocator) flower_allocator;
};

// One workgroup per candidate render voxel object: frustum + HiZ occlusion
// test its world-space AABB (GpuVoxelObject::aabb_min/aabb_max), and if
// visible, append its index (into voxel_object_manifests) to visible_foliage_bricks
// (slot 0 = atomic count, slots [1..] = object indices).
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(FoliageCull)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuVoxelObject), voxel_object_manifests)
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(FoliageBrickInstance), visible_foliage_bricks)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, hiz)
DAXA_DECL_TASK_HEAD_END

struct FoliageCullPush {
    DAXA_TH_BLOB(FoliageCull, uses)
    daxa_u32 object_count;
    daxa_u32 hiz_mip_count;
};

#if defined(__cplusplus)
struct FoliageCullInfo {
    RenderScene *scene;
};
#endif
