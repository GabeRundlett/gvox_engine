#pragma once

#include <application/input.inl>

#include <voxels/voxels.inl>
#include <voxels/voxel_particles.inl>

struct RendererImpl;

struct Renderer {
    std::unique_ptr<RendererImpl> impl;

    Renderer();
    ~Renderer();

    void create(daxa::Device &device);
    void destroy(daxa::Device &device);

    void begin_frame(GpuInput &gpu_input, GpuOutput &gpu_output);
    void end_frame(daxa::Device &device, RendererSettings const &settings, float dt);
    auto render(RecordContext &record_ctx, VoxelWorldBuffers &voxel_buffers, VoxelParticles &particles, daxa::TaskImageView output_image, daxa::Format output_format) -> daxa::TaskImageView;
};
