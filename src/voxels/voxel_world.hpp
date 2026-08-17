#pragma once

struct VoxelWorld;

auto create_voxel_world(struct Scene *scene) -> VoxelWorld *;
void destroy_voxel_world(VoxelWorld *self);
void update_voxel_world(struct GpuContext& gpu_context,struct Renderer &renderer, struct GpuInput& gpu_input, VoxelWorld *self);
