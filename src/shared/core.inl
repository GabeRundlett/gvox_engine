#pragma once

#define DAXA_ENABLE_SHADER_NO_NAMESPACE 1
#define DAXA_ENABLE_IMAGE_OVERLOADS_BASIC 1
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#undef DAXA_ENABLE_SHADER_NO_NAMESPACE
#undef DAXA_ENABLE_IMAGE_OVERLOADS_BASIC

#define CHUNK_SIZE 64 // A chunk = 64^3 voxels

#define PALETTE_REGION_SIZE 8
#define PALETTE_REGION_TOTAL_SIZE (PALETTE_REGION_SIZE * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE)
#define PALETTE_MAX_COMPRESSED_VARIANT_N 367

#if PALETTE_REGION_SIZE != 8
#error Unsupported Palette Region Size
#endif

#define PALETTES_PER_CHUNK_AXIS (CHUNK_SIZE / PALETTE_REGION_SIZE)
#define PALETTES_PER_CHUNK (PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS)

#define MAX_CHUNK_UPDATES_PER_FRAME 64
#define MAX_QUEUED_BRUSH_INPUTS 128

// doing x << 9 is the same as doing x * 512. Written this way for clarity only.
// We multiply by 512 because that's how many times the brush can subdivide
#define MAX_CHUNK_WORK_ITEMS_L0 (MAX_QUEUED_BRUSH_INPUTS << 0)
#define MAX_CHUNK_WORK_ITEMS_L1 (MAX_QUEUED_BRUSH_INPUTS << 9)
#define MAX_CHUNK_WORK_ITEMS_L2 MAX_CHUNK_UPDATES_PER_FRAME

#define MAX_SIMULATED_VOXEL_PARTICLES 0 // (1 << 14)
#define MAX_RENDERED_VOXEL_PARTICLES 0 // (1 << 14)

#define L2_CHUNK_SIZE CHUNK_SIZE
#define L1_CHUNK_SIZE (L2_CHUNK_SIZE * 8)
#define L0_CHUNK_SIZE (L1_CHUNK_SIZE * 8)

#if defined(__cplusplus)
#include <memory>
#include <daxa/utils/pipeline_manager.hpp>
#endif

#define USE_POINTS 0
#define PREPASS_SCL 2
#define SHADING_SCL 2
