#pragma once

#include <application/input.inl>

#include <voxels/voxels.inl>
#include <voxels/particles/voxel_particles.inl>

struct RendererImpl;

struct Renderer {
    std::unique_ptr<RendererImpl> impl;

    Renderer();
    ~Renderer();

    void begin_frame(GpuInput &gpu_input);
    void end_frame(daxa::Device &device, float dt);
    auto render(GpuContext &gpu_context, VoxelWorldBuffers &voxel_buffers, VoxelParticles &particles, daxa::TaskImageView output_image, daxa::Format output_format) -> daxa::TaskImageView;
};
