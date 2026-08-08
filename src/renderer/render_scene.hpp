#pragma once

struct RenderScene *create_render_scene(struct GpuContext &gpu_context);
void destroy_render_scene(struct GpuContext &gpu_context, struct RenderScene *self);

void render_scene_begin(struct GpuContext &gpu_context, struct RenderScene *self);
void render_scene_end(struct GpuContext &gpu_context, struct RenderScene *self);
void record_render_scene(struct GpuContext &gpu_context, struct RenderScene *self);

#if RENDERER_INTERNAL

#include <utilities/gpu_context.hpp>
#include <voxels/voxel.inl>

struct RenderScene {
    VoxelWorldBuffers buffers;
    daxa::ExternalTaskBuffer task_voxel_object_slotmap;
    std::vector<GpuVoxelObject> drawn_voxel_object_manifests;
    std::vector<daxa_BlasInstanceData> drawn_voxel_objects_blas_instances;
};

#endif
