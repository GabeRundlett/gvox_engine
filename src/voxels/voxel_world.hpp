#pragma once

#include <glm/vec3.hpp>

struct VoxelWorld;

auto create_voxel_world(struct Scene *scene) -> VoxelWorld *;
void destroy_voxel_world(VoxelWorld *self);
void update_voxel_world(struct GpuContext &gpu_context, struct Renderer &renderer, struct GpuInput &gpu_input, VoxelWorld *self);

// Collision queries. Main thread only -- chunk generation mutates the chunk grid from
// worker threads inside update_voxel_world().
//
// Regions with no generated brick data fall back to the terrain density function, so
// these are correct even for chunks that haven't streamed in (or that the generator
// skipped because they hold no surface). Anything outside the world reads as empty.
bool voxel_world_is_solid(VoxelWorld *self, glm::ivec3 voxel_coord);
bool voxel_world_box_is_solid(VoxelWorld *self, glm::vec3 box_min, glm::vec3 box_max, glm::ivec3 *out_hit_voxel = nullptr);
glm::vec3 voxel_world_terrain_normal(VoxelWorld *self, glm::vec3 world_pos);
