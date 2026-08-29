#pragma once

struct VoxelAllocator;
VoxelAllocator *create_voxel_allocator();
void destroy_voxel_allocator(VoxelAllocator *self);

struct VoxelBrick *alloc_brick(VoxelAllocator *self);
void free_brick(VoxelAllocator *self, VoxelBrick *brick);
struct VoxelShadingAttribBrick *alloc_render_brick(VoxelAllocator *self);
void free_render_brick(VoxelAllocator *self, struct VoxelShadingAttribBrick *render_brick);
