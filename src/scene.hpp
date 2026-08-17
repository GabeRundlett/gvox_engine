#pragma once

#include <base/vec.hpp>
struct VoxelObject;
struct AnimationPlayground;

struct Scene {
    Vec<VoxelObject *> voxel_objects;
    struct RenderScene *render_scene;
    struct GpuContext &gpu_context;
    struct VoxelWorld *voxel_world;

    VoxelObject *ball_frames[8];

    AnimationPlayground *animation_playground = nullptr;

    Scene(struct GpuContext &gpu_context);
    ~Scene();

    void update(struct Renderer &renderer, struct GpuInput &gpu_input);
};
