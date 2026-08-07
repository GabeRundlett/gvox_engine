#pragma once

#include <vector>
class VoxelObject;

struct Scene
{
    std::vector<VoxelObject*> voxel_objects;
    struct RenderScene* render_scene;
    struct GpuContext& gpu_context;

    Scene(struct GpuContext& gpu_context);
    ~Scene();
};
