#pragma once

#include <glm/fwd.hpp>
#include <glm/gtc/quaternion.hpp>

struct RenderVoxelObject *create_render_voxel_object(struct RenderScene *scene, bool has_foliage = false);
void destroy_render_voxel_object(struct GpuContext &gpu_context, struct RenderVoxelObject *object);
void update_render_voxel_object(struct GpuContext &gpu_context, struct VoxelObject *src);
void draw_voxel_object(struct VoxelObject *object, const glm::vec3 &pos, const glm::quat &rotation, float scale, const glm::vec3 &tint);
