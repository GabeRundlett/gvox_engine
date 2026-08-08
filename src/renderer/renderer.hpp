#pragma once

#include <application/input.inl>
#include <memory>
// #include <voxels/voxel.inl>
// #include <voxels/particles/voxel_particles.inl>

struct RendererImpl;

struct Line {
    float p0_x, p0_y, p0_z;
    float p1_x, p1_y, p1_z;
    float r, g, b;
};
struct Point {
    float p0_x, p0_y, p0_z;
    float r, g, b;
    float s_x, s_y, type;
};
struct Box {
    float p0_x, p0_y, p0_z;
    float p1_x, p1_y, p1_z;
    float r, g, b;
};

struct Renderer {
    std::unique_ptr<RendererImpl> impl;

    Renderer();
    ~Renderer();

    void begin_frame(GpuInput &gpu_input);
    void end_frame(daxa::Device &device, float dt);
    auto render(GpuContext &gpu_context, struct RenderScene *scene, daxa::TaskImageView output_image, daxa::Format output_format) -> daxa::TaskImageView;

    void submit_debug_lines(Line const *lines, int line_n);
    void submit_debug_points(Point const *points, int point_n);
    void submit_debug_box_lines(Box const *cubes, int cube_n);
};
