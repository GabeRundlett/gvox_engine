#pragma once

#include <cstdint>
#include <daxa/daxa.hpp>
#include <daxa/utils/task_graph.hpp>
#include <renderer/pipeline_manager.hpp>

struct TemporalBuffer;
struct GrassStrands;

#define MAX_FOLIAGE_BRICK_COUNT (1 << 20)

struct RenderFoliageBricks {
    daxa::ComputePipeline generate_pipeline;

    GrassStrands *grass = nullptr;
    bool grass_initialized = false;
    TemporalBuffer *particles_state = nullptr;
    TemporalBuffer *cube_index_buffer = nullptr;
    TemporalBuffer *visible_foliage_bricks = nullptr;
};

void init_render_foliage_bricks(struct GpuContext &gpu_context);
void deinit_render_foliage_bricks(struct GpuContext &gpu_context);
void record_render_foliage_bricks(struct GpuContext &gpu_context, daxa::TaskGraph &task_graph, daxa::TaskImageView hiz, struct RenderScene *scene);

// Draws the currently-alive grass strands (cubes + splats) into the gbuffer,
// returning the shadow-map depth image they rendered into.
auto render_foliage_grass(struct GpuContext &gpu_context, struct GbufferDepth &gbuffer_depth, daxa::TaskImageView velocity_image) -> daxa::TaskImageView;
