#pragma once

#include <core.inl>
#include <voxels/brushes.inl>

#define VOXELS_ORIGINAL_IMPL

#define CHUNK_SIZE 64 // A chunk = 64^3 voxels
#define CHUNK_NX 32
#define CHUNK_NY 32
#define CHUNK_NZ 32

#define LOG2_VOXEL_SIZE (-4)
#if LOG2_VOXEL_SIZE <= 0
#define VOXEL_SCL (1 << (-LOG2_VOXEL_SIZE))
#define VOXEL_SIZE (1.0 / VOXEL_SCL)
#else
#define VOXEL_SIZE (1 << LOG2_VOXEL_SIZE)
#define VOXEL_SCL (1.0 / VOXEL_SIZE)
#endif
#define CHUNK_WORLDSPACE_SIZE (float(CHUNK_SIZE) * float(VOXEL_SIZE))
#if LOG2_VOXEL_SIZE < -6
#error "this is not currently supported"
#endif

#define PALETTE_REGION_SIZE 8
#define PALETTE_REGION_TOTAL_SIZE (PALETTE_REGION_SIZE * PALETTE_REGION_SIZE * PALETTE_REGION_SIZE)
#define PALETTE_MAX_COMPRESSED_VARIANT_N 367

#if PALETTE_REGION_SIZE != 8
#error Unsupported Palette Region Size
#endif

#define PALETTES_PER_CHUNK_AXIS (CHUNK_SIZE / PALETTE_REGION_SIZE)
#define PALETTES_PER_CHUNK (PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS * PALETTES_PER_CHUNK_AXIS)

#define MAX_CHUNK_UPDATES_PER_FRAME 128

#define PACKED_NULL_VOXEL 0xFFFF00FC
#define PACKED_NULL_VOXEL_MASK 0xFFFF00FF

// 8 bytes per 8x8x8
struct PaletteHeader {
    daxa_u32 variant_n;
    daxa_u32 blob_ptr;
};

struct VoxelParentChunk {
    daxa_u32 is_uniform;
    daxa_u32 children[512];
    daxa_u32 is_ptr[16];
};
DAXA_DECL_BUFFER_PTR(VoxelParentChunk)

#define VOXEL_BRICK_SIZE 8

struct VoxelBrickBitmask {
    daxa_u32 data[VOXEL_BRICK_SIZE * VOXEL_BRICK_SIZE * VOXEL_BRICK_SIZE / 32];
};
struct VoxelBrickAttribs {
    PackedVoxel packed_voxels[VOXEL_BRICK_SIZE * VOXEL_BRICK_SIZE * VOXEL_BRICK_SIZE];
};
DAXA_DECL_BUFFER_PTR(VoxelBrickAttribs)
DAXA_DECL_BUFFER_PTR(daxa_BufferPtr(VoxelBrickAttribs))
struct VoxelBlasTransform {
    daxa_f32vec3 pos;
    daxa_f32vec3 vel;
};
DAXA_DECL_BUFFER_PTR(VoxelBlasTransform)

struct VoxelLeafChunk {
    daxa_u32 flags;
    daxa_u32 update_index;
};
DAXA_DECL_BUFFER_PTR(VoxelLeafChunk)

// DECL_SIMPLE_ALLOCATOR(VoxelLeafChunkAllocator, VoxelLeafChunk, 1, daxa_u32, (MAX_CHUNK_WORK_ITEMS_L2))
// DECL_SIMPLE_ALLOCATOR(VoxelParentChunkAllocator, VoxelParentChunk, 1, daxa_u32, (MAX_CHUNK_WORK_ITEMS_L0 + MAX_CHUNK_WORK_ITEMS_L1))

struct TempVoxelChunk {
    PackedVoxel voxels[CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE];
};
DAXA_DECL_BUFFER_PTR(TempVoxelChunk)

struct VoxelChunkUpdateInfo {
    daxa_i32vec3 i;
    daxa_i32vec3 chunk_offset;
    daxa_u32 brush_flags;
    BrushInput brush_input;
};

struct VoxelWorldGpuIndirectDispatch {
    daxa_u32vec3 chunk_edit_dispatch;
    daxa_u32vec3 subchunk_x2x4_dispatch;
    daxa_u32vec3 subchunk_x8up_dispatch;
};

struct BrushState {
    daxa_u32 initial_frame;
    daxa_f32vec3 initial_ray;
    daxa_u32 is_editing;
};

struct VoxelWorldGlobals {
    BrushInput brush_input;
    BrushState brush_state;
    VoxelWorldGpuIndirectDispatch indirect_dispatch;

    VoxelChunkUpdateInfo chunk_update_infos[MAX_CHUNK_UPDATES_PER_FRAME];
    daxa_i32vec3 prev_offset;
    daxa_u32 chunk_update_n; // Number of chunks to update
    daxa_i32vec3 offset;
    daxa_u32 chunk_update_heap_alloc_n;
};
DAXA_DECL_BUFFER_PTR(VoxelWorldGlobals)

struct CpuChunkUpdateInfo {
    daxa_u32 chunk_index;
    daxa_u32 flags;
};
struct ChunkUpdate {
    CpuChunkUpdateInfo info;
    PaletteHeader palette_headers[PALETTES_PER_CHUNK];
};
DAXA_DECL_BUFFER_PTR(ChunkUpdate)

#define VOXELS_USE_BUFFERS(ptr_type, mode)                               \
    DAXA_TH_BUFFER_PTR(mode, ptr_type(VoxelWorldGlobals), voxel_globals) \
    DAXA_TH_BUFFER_PTR(mode, ptr_type(VoxelLeafChunk), voxel_chunks)

#define VOXELS_USE_BUFFERS_PUSH_USES(ptr_type)                           \
    ptr_type(VoxelWorldGlobals) voxel_globals = push.uses.voxel_globals; \
    ptr_type(VoxelLeafChunk) voxel_chunks = push.uses.voxel_chunks;

#define VOXELS_BUFFER_USES_ASSIGN(TaskHeadName, voxel_buffers)                                                   \
    daxa::TaskViewVariant{std::pair{TaskHeadName::AT.voxel_globals, voxel_buffers.voxel_globals.task_resource}}, \
        daxa::TaskViewVariant {                                                                                  \
        std::pair { TaskHeadName::AT.voxel_chunks, voxel_buffers.voxel_chunks.task_resource }                    \
    }

struct VoxelBufferPtrs {
    daxa_BufferPtr(VoxelLeafChunk) voxel_chunks_ptr;
    daxa_BufferPtr(VoxelWorldGlobals) globals;
};
struct VoxelRtBufferPtrs {
    daxa_BufferPtr(daxa_BufferPtr(BlasGeom)) geometry_pointers;
    daxa_BufferPtr(daxa_BufferPtr(VoxelBrickAttribs)) attribute_pointers;
    daxa_BufferPtr(VoxelBlasTransform) blas_transforms;
    daxa_u64 tlas;
};
struct VoxelRWBufferPtrs {
    daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunks_ptr;
    daxa_RWBufferPtr(VoxelWorldGlobals) globals;
};

#define VOXELS_BUFFER_PTRS VoxelBufferPtrs(daxa_BufferPtr(VoxelLeafChunk)(as_address(voxel_chunks)), daxa_BufferPtr(VoxelWorldGlobals)(as_address(voxel_globals)))
#define VOXELS_RW_BUFFER_PTRS VoxelRWBufferPtrs(daxa_RWBufferPtr(VoxelLeafChunk)(as_address(voxel_chunks)), daxa_RWBufferPtr(VoxelWorldGlobals)(as_address(voxel_globals)))

#define VOXELS_RT_BUFFER_PTRS VoxelRtBufferPtrs( \
    push.uses.geometry_pointers,                 \
    push.uses.attribute_pointers,                \
    push.uses.blas_transforms,                   \
    push.uses.tlas)

#define MAX_CHUNK_UPDATES_PER_FRAME_VOXEL_COUNT (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * MAX_CHUNK_UPDATES_PER_FRAME)

#if defined(__cplusplus)

#include <utilities/allocator.inl>

struct VoxelWorldBuffers {
    TemporalBuffer voxel_globals;
    TemporalBuffer voxel_chunks;
    TemporalBuffer chunk_updates;
    TemporalBuffer chunk_update_heap;

    TemporalBuffer blas_geom_pointers;
    TemporalBuffer blas_attr_pointers;
    TemporalBuffer blas_transforms;

    daxa::BufferId tlas_buffer;
    daxa::TlasId tlas;
    daxa::TaskTlas task_tlas;

    // AllocatorBufferState<VoxelLeafChunkAllocator> voxel_leaf_chunk_malloc;
    // AllocatorBufferState<VoxelParentChunkAllocator> voxel_parent_chunk_malloc;
};

#endif
