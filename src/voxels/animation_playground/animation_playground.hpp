#pragma once

#include <renderer/gpu_context.hpp>
#include <glm/vec3.hpp>
#include <base/vec.hpp>

struct VoxelObject;
struct RenderScene;
struct GpuInput;

struct AnimationPlayground {
    GpuContext &gpu_context;
    RenderScene *render_scene;

    daxa::ComputePipeline pipeline;

    daxa::BufferId bricks_buffer{};
    daxa::BufferId brick_attribs_buffer{};
    daxa::BufferId bricks_readback_buffer{};
    daxa::BufferId brick_attribs_readback_buffer{};

    glm::ivec3 grid_dims_bricks{4, 4, 4};
    int frame_count = 32;
    bool dirty = true;

    size_t total_brick_count = 0;

    glm::vec3 playground_pos = glm::vec3{0.0f, 0.0f, 0.0f};

    Vec<VoxelObject *> frames;

    bool playing = true;
    float current_frame_f = 0.0f;
    float playback_fps = 24.0f;

    AnimationPlayground(GpuContext &gpu_context, RenderScene *render_scene);
    AnimationPlayground(AnimationPlayground const &) = delete;
    AnimationPlayground(AnimationPlayground &&) = delete;
    auto operator=(AnimationPlayground const &) -> AnimationPlayground & = delete;
    auto operator=(AnimationPlayground &&) -> AnimationPlayground & = delete;
    ~AnimationPlayground();

    // Dispatches the generate shader for all frames, reads the result back to the
    // CPU, and rebuilds `frames` from it.
    void regenerate(float time);

    // Advances playback, draws the ImGui control panel, regenerates if `dirty`, and
    // draws the current frame into the world.
    void update(struct Renderer &renderer, GpuInput const &gpu_input);
    void ui();
};
