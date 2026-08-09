#pragma once

#include <vector>
struct VoxelObject;

struct Scene
{
    std::vector<VoxelObject*> voxel_objects;
    struct RenderScene* render_scene;
    struct GpuContext& gpu_context;
    struct VoxelWorld* voxel_world;

    VoxelObject *ball_frames[8];

    Scene(struct GpuContext& gpu_context);
    ~Scene();

    void update(struct Renderer& renderer, struct GpuInput& gpu_input);
};
