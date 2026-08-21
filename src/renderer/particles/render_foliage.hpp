#pragma once

#include <cstdint>
#include <daxa/daxa.hpp>
#include <daxa/utils/task_graph.hpp>
#include <renderer/pipeline_manager.hpp>

// TemporalBuffer is defined in gpu_context.hpp (which includes this header).
// GrassStrands is defined in voxels/particles/grass/grass.inl, whose inline
// methods need a complete GpuContext -- both are kept as pointers here (only
// fully included from render_foliage.cpp, after gpu_context.hpp is visible)
// so this header doesn't have to drag those dependencies into gpu_context.hpp.
struct TemporalBuffer;
struct GrassStrands;

#define MAX_FOLIAGE_BRICK_COUNT (1 << 20)

struct RenderFoliageBrick {
    // Points into the owning VoxelBrick::foliage_bitmask -- not owned here, so
    // there's exactly one copy of the bitmask (also the source uploaded into
    // the render voxel object's foliage section).
    uint64_t bitmask[BRICK_SIZE * BRICK_SIZE * BRICK_SIZE / 64];
    daxa_f32vec3 pos;
    bool dirty = false;
};

struct RenderFoliageBricks {
    daxa::ComputePipeline generate_pipeline;

    // Grass blades spawned from foliage bricks, drawn indirectly via the
    // existing particle cube/splat raster pipeline.
    GrassStrands *grass = nullptr;
    bool grass_initialized = false;
    TemporalBuffer *particles_state = nullptr;
    TemporalBuffer *cube_index_buffer = nullptr;

    // Slot 0 = atomic visible-object count, slots [1..] = indices into this
    // frame's voxel_object_manifests, written by cull_pipeline.
    TemporalBuffer *visible_foliage_bricks = nullptr;
};

void init_render_foliage_bricks(struct GpuContext &gpu_context);
void deinit_render_foliage_bricks(struct GpuContext &gpu_context);

// Registers task_buffer, uploads dirty foliage bricks, spawns grass strands
// for newly-uploaded foliage voxels, and runs the per-frame grass
// select/cull/build-draw-list pass.
void record_render_foliage_bricks(struct GpuContext &gpu_context, daxa::TaskGraph &task_graph, daxa::TaskImageView hiz, struct RenderScene *scene);

// Draws the currently-alive grass strands (cubes + splats) into the gbuffer,
// returning the shadow-map depth image they rendered into.
auto render_foliage_grass(struct GpuContext &gpu_context, struct GbufferDepth &gbuffer_depth, daxa::TaskImageView velocity_image) -> daxa::TaskImageView;
