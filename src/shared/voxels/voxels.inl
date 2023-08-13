#pragma once

#include <shared/voxels/voxel_malloc.inl>
#include <shared/voxels/gvox_model.inl>

// 1364 u32's
// 10.65625 bytes per 8x8x8
struct VoxelChunkUniformity {
    u32 lod_x2[1024];
    u32 lod_x4[256];
    u32 lod_x8[64];
    u32 lod_x16[16];
    u32 lod_x32[4];
};

// 8 bytes per 8x8x8
struct PaletteHeader {
    u32 variant_n;
    VoxelMalloc_Pointer blob_ptr;
};

struct VoxelLeafChunk {
    u32 flags;
    VoxelChunkUniformity uniformity;
    VoxelMalloc_ChunkLocalPageSubAllocatorState sub_allocator_state;
    PaletteHeader palette_headers[PALETTES_PER_CHUNK];
};
DAXA_DECL_BUFFER_PTR(VoxelLeafChunk)

struct TempVoxel {
    u32 col_and_id;
};

struct TempVoxelChunk {
    TempVoxel voxels[CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE];
};
DAXA_DECL_BUFFER_PTR(TempVoxelChunk)

struct VoxelParentChunk {
    u32 is_uniform;
    u32 children[512];
    u32 is_leaf[16];
};
DAXA_DECL_BUFFER_PTR(VoxelParentChunk)

DECL_SIMPLE_ALLOCATOR(VoxelLeafChunkAllocator, VoxelLeafChunk, 1, u32, (MAX_CHUNK_WORK_ITEMS_L2))
DECL_SIMPLE_ALLOCATOR(VoxelParentChunkAllocator, VoxelParentChunk, 1, u32, (MAX_CHUNK_WORK_ITEMS_L0 + MAX_CHUNK_WORK_ITEMS_L1))

struct BrushInput {
    f32vec3 pos;
    f32vec3 prev_pos;
};

struct ChunkWorkItem {
    i32vec3 i;
    i32vec3 chunk_offset;
    u32 brush_id;           // Brush ID
    BrushInput brush_input; // Brush input parameters

    u32 children_finished[16]; // bitmask of completed children work items (16x32 = 512 children)
};

// Manages L0 and L1 ChunkWorkItems
// Values are reset between frames in perframe.comp.glsl in the following way:
// total_jobs_ran = l0_uncompleted + l1_uncompleted
// l0_queued = l0_uncompleted
// l1_queued = l1_uncompleted
// l0_completed = l0_uncompleted = l1_completed = l1_uncompleted = 0

struct ChunkThreadPoolState {
    u32 total_jobs_ran; // total work items to run for the current frame
    u32 queue_index;    // Current queue (0: default, 1: destination for repacking unfinished work items)

    u32 work_items_l0_queued;     // Number of L0 work items in queue (also L0 dispatch x)
    u32 work_items_l0_dispatch_y; // 1
    u32 work_items_l0_dispatch_z; // 1

    u32 work_items_l1_queued;     // Number of L1 work items in queue (also L1 dispatch x)
    u32 work_items_l1_dispatch_y; // 1
    u32 work_items_l1_dispatch_z; // 1

    u32 work_items_l0_completed;                                   // Number of L0 work items completed for the current frame
    u32 work_items_l0_uncompleted;                                 // Number of L0 work items left to do (one frame)
    ChunkWorkItem chunk_work_items_l0[2][MAX_CHUNK_WORK_ITEMS_L0]; // L0 work items list

    u32 work_items_l1_completed;                                   // Number of L1 work items completed (one frame)
    u32 work_items_l1_uncompleted;                                 // Number of L1 work items left to do (one frame)
    ChunkWorkItem chunk_work_items_l1[2][MAX_CHUNK_WORK_ITEMS_L1]; // L1 work items list
};

struct VoxelChunkUpdateInfo {
    i32vec3 i;
    i32vec3 chunk_offset;
    u32 flags; // brush flags
    BrushInput brush_input;
};

struct VoxelWorldGlobals {
    VoxelChunkUpdateInfo chunk_update_infos[MAX_CHUNK_UPDATES_PER_FRAME];
    u32 chunk_update_n; // Number of chunks to update
};
