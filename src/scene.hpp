#pragma once

#include <vector>
struct VoxelObject;

struct Scene
{
    std::vector<VoxelObject*> voxel_objects;
    struct RenderScene* render_scene;
    struct GpuContext& gpu_context;

    Scene(struct GpuContext& gpu_context);
    ~Scene();

    void update(struct Renderer& renderer, struct GpuInput& gpu_input);
};
