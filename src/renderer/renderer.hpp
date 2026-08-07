#pragma once

#include <application/input.inl>
#include <memory>
// #include <voxels/voxel.inl>
// #include <voxels/particles/voxel_particles.inl>

struct RendererImpl;

struct Renderer {
    std::unique_ptr<RendererImpl> impl;

    Renderer();
    ~Renderer();

    void begin_frame(GpuInput &gpu_input);
    void end_frame(daxa::Device &device, float dt);
    auto render(GpuContext &gpu_context, struct RenderScene *scene, daxa::TaskImageView output_image, daxa::Format output_format) -> daxa::TaskImageView;
};
