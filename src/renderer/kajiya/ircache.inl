#pragma once

#include <core.inl>
#include <application/input.inl>
#include <voxels/voxel.inl>
#include <renderer/kajiya/prefix_scan.inl>

struct Vertex {
    daxa_f32vec3 position;
    daxa_f32vec3 normal;
};
struct VertexPacked {
    daxa_f32vec4 data0;
};
DAXA_DECL_BUFFER_PTR(VertexPacked)
struct IrcacheMetadata {
    // Same as alloc_count, but frozen at the rt dispatch args stage.
    daxa_u32 tracing_alloc_count;
    daxa_u32 _pad0;
    daxa_u32 entry_count;
    daxa_u32 alloc_count;
};
DAXA_DECL_BUFFER_PTR(IrcacheMetadata)
struct IrcacheCell {
    daxa_u32 entry_index;
    daxa_u32 flags;
};
DAXA_DECL_BUFFER_PTR(IrcacheCell)
const daxa_u32 IRCACHE_OCTA_DIMS = 4;
const daxa_u32 IRCACHE_OCTA_DIMS2 = IRCACHE_OCTA_DIMS * IRCACHE_OCTA_DIMS;
struct IrcacheAux {
    daxa_u32vec2 reservoirs[IRCACHE_OCTA_DIMS2];
    daxa_f32vec4 values[IRCACHE_OCTA_DIMS2];
    VertexPacked vertexes[IRCACHE_OCTA_DIMS2];
};
DAXA_DECL_BUFFER_PTR(IrcacheAux)
struct IrcacheBuffers {
    daxa_RWBufferPtr(IrcacheMetadata) ircache_meta_buf;
    daxa_RWBufferPtr(IrcacheCell) ircache_grid_meta_buf;
    daxa_RWBufferPtr(daxa_u32) ircache_entry_cell_buf;
    daxa_RWBufferPtr(VertexPacked) ircache_spatial_buf;
    daxa_RWBufferPtr(daxa_f32vec4) ircache_irradiance_buf;
    daxa_RWBufferPtr(IrcacheAux) ircache_aux_buf;
    daxa_RWBufferPtr(daxa_u32) ircache_life_buf;
    daxa_RWBufferPtr(daxa_u32) ircache_pool_buf;
    daxa_RWBufferPtr(daxa_u32) ircache_entry_indirection_buf;
    daxa_RWBufferPtr(VertexPacked) ircache_reposition_proposal_buf;
    daxa_RWBufferPtr(daxa_u32) ircache_reposition_proposal_count_buf;
};
DAXA_DECL_BUFFER_PTR(IrcacheBuffers)

#define IRCACHE_BUFFER_USE_N 12

#define IRCACHE_USE_BUFFERS(SHADER_TYPE)                                                           \
    DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(IrcacheBuffers), ircache_buffers) \
    DAXA_TH_BUFFER(READ_WRITE_CONCURRENT, ircache_meta_buf)                   \
    DAXA_TH_BUFFER(READ_WRITE_CONCURRENT, ircache_grid_meta_buf)              \
    DAXA_TH_BUFFER(READ_WRITE_CONCURRENT, ircache_entry_cell_buf)             \
    DAXA_TH_BUFFER(READ_WRITE_CONCURRENT, ircache_spatial_buf)                \
    DAXA_TH_BUFFER(READ_WRITE_CONCURRENT, ircache_irradiance_buf)             \
    DAXA_TH_BUFFER(READ_WRITE_CONCURRENT, ircache_aux_buf)                    \
    DAXA_TH_BUFFER(READ_WRITE_CONCURRENT, ircache_life_buf)                   \
    DAXA_TH_BUFFER(READ_WRITE_CONCURRENT, ircache_pool_buf)                   \
    DAXA_TH_BUFFER(READ_WRITE_CONCURRENT, ircache_entry_indirection_buf)      \
    DAXA_TH_BUFFER(READ_WRITE_CONCURRENT, ircache_reposition_proposal_buf)    \
    DAXA_TH_BUFFER(READ_WRITE_CONCURRENT, ircache_reposition_proposal_count_buf)

#define IRCACHE_BUFFER_USES_ASSIGN(TaskHeadName, ircache)                                                       \
    .ircache_buffers = ircache.ircache_buffers,                                                                 \
        .ircache_meta_buf = ircache.ircache_meta_buf.view(),                                                    \
        .ircache_grid_meta_buf = ircache.ircache_grid_meta_buf.view(),                                          \
        .ircache_entry_cell_buf = ircache.ircache_entry_cell_buf.view(),                                        \
        .ircache_spatial_buf = ircache.ircache_spatial_buf.view(),                                              \
        .ircache_irradiance_buf = ircache.ircache_irradiance_buf.view(),                                        \
        .ircache_aux_buf = ircache.ircache_aux_buf.view(),                                                      \
        .ircache_life_buf = ircache.ircache_life_buf.view(),                                                    \
        .ircache_pool_buf = ircache.ircache_pool_buf.view(),                                                    \
        .ircache_entry_indirection_buf = ircache.ircache_entry_indirection_buf.view(),                          \
        .ircache_reposition_proposal_buf = ircache.ircache_reposition_proposal_buf.view(),                      \
        .ircache_reposition_proposal_count_buf = ircache.ircache_reposition_proposal_count_buf.view()

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(ClearIrcachePoolCompute)
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(daxa_u32), ircache_pool_buf)
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(daxa_u32), ircache_life_buf)
DAXA_DECL_TASK_HEAD_END
struct ClearIrcachePoolComputePush {
    DAXA_TH_BLOB(ClearIrcachePoolCompute, uses)
};
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(IrcacheScrollCascadesCompute)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(IrcacheCell), ircache_grid_meta_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(IrcacheCell), ircache_grid_meta_buf2)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_entry_cell_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_f32vec4), ircache_irradiance_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_pool_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_DECL_TASK_HEAD_END
struct IrcacheScrollCascadesComputePush {
    DAXA_TH_BLOB(IrcacheScrollCascadesCompute, uses)
};
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(IrcachePrepareAgeDispatchCompute)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(daxa_u32vec4), dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct IrcachePrepareAgeDispatchComputePush {
    DAXA_TH_BLOB(IrcachePrepareAgeDispatchCompute, uses)
};
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(AgeIrcacheEntriesCompute)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(IrcacheCell), ircache_grid_meta_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_entry_cell_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_pool_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(VertexPacked), ircache_spatial_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(VertexPacked), ircache_reposition_proposal_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_reposition_proposal_count_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_f32vec4), ircache_irradiance_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), entry_occupancy_buf)
DAXA_TH_BUFFER(READ, dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct AgeIrcacheEntriesComputePush {
    DAXA_TH_BLOB(AgeIrcacheEntriesCompute, uses)
};
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(IrcacheCompactEntriesCompute)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), entry_occupancy_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_entry_indirection_buf)
DAXA_TH_BUFFER(READ, dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct IrcacheCompactEntriesComputePush {
    DAXA_TH_BLOB(IrcacheCompactEntriesCompute, uses)
};
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(IrcachePrepareTraceDispatchCompute)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(daxa_u32vec4), dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct IrcachePrepareTraceDispatchComputePush {
    DAXA_TH_BLOB(IrcachePrepareTraceDispatchCompute, uses)
};
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(IrcacheResetCompute)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_f32vec4), ircache_irradiance_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(IrcacheAux), ircache_aux_buf)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), ircache_entry_indirection_buf)
DAXA_TH_BUFFER(READ, dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct IrcacheResetComputePush {
    DAXA_TH_BLOB(IrcacheResetCompute, uses)
};
DAXA_DECL_RAY_TRACING_TASK_HEAD_BEGIN(IrcacheTraceAccessRt)
// DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_BufferPtr(BlasGeom)), geometry_pointers)
// DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_BufferPtr(VoxelBrickAttribs)), attribute_pointers)
// DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VoxelBlasTransform), blas_transforms)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuVoxelObject), voxel_object_manifests)
DAXA_TH_TLAS_PTR(READ, tlas)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VertexPacked), ircache_spatial_buf)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(VertexPacked), ircache_reposition_proposal_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(IrcacheAux), ircache_aux_buf)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), ircache_entry_indirection_buf)
DAXA_TH_BUFFER(READ, dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct IrcacheTraceAccessRtPush {
    DAXA_TH_BLOB(IrcacheTraceAccessRt, uses)
};
DAXA_DECL_RAY_TRACING_TASK_HEAD_BEGIN(IrcacheValidateRt)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
// DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_BufferPtr(BlasGeom)), geometry_pointers)
// DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_BufferPtr(VoxelBrickAttribs)), attribute_pointers)
// DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VoxelBlasTransform), blas_transforms)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuVoxelObject), voxel_object_manifests)
DAXA_TH_TLAS_PTR(READ, tlas)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VertexPacked), ircache_spatial_buf)
DAXA_TH_IMAGE_INDEX(SAMPLE, CUBE, sky_cube_tex)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, transmittance_lut)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(IrcacheCell), ircache_grid_meta_buf)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(VertexPacked), ircache_reposition_proposal_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(daxa_u32), ircache_reposition_proposal_count_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(IrcacheAux), ircache_aux_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(daxa_u32), ircache_pool_buf)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), ircache_entry_indirection_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(daxa_u32), ircache_entry_cell_buf)
DAXA_TH_BUFFER(READ, dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct IrcacheValidateRtPush {
    DAXA_TH_BLOB(IrcacheValidateRt, uses)
};
DAXA_DECL_RAY_TRACING_TASK_HEAD_BEGIN(TraceIrradianceRt)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
// DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_BufferPtr(BlasGeom)), geometry_pointers)
// DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_BufferPtr(VoxelBrickAttribs)), attribute_pointers)
// DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VoxelBlasTransform), blas_transforms)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuVoxelObject), voxel_object_manifests)
DAXA_TH_TLAS_PTR(READ, tlas)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VertexPacked), ircache_spatial_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(IrcacheCell), ircache_grid_meta_buf)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(VertexPacked), ircache_reposition_proposal_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(daxa_u32), ircache_reposition_proposal_count_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(IrcacheAux), ircache_aux_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(daxa_u32), ircache_pool_buf)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), ircache_entry_indirection_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(daxa_u32), ircache_entry_cell_buf)
DAXA_TH_IMAGE_INDEX(SAMPLE, CUBE, sky_cube_tex)
DAXA_TH_IMAGE_INDEX(SAMPLE, REGULAR_2D, transmittance_lut)
DAXA_TH_BUFFER(READ, dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct TraceIrradianceRtPush {
    DAXA_TH_BLOB(TraceIrradianceRt, uses)
};
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(SumUpIrradianceCompute)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_f32vec4), ircache_irradiance_buf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(IrcacheAux), ircache_aux_buf)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), ircache_entry_indirection_buf)
DAXA_TH_BUFFER(READ, dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct SumUpIrradianceComputePush {
    DAXA_TH_BLOB(SumUpIrradianceCompute, uses)
};

#if defined(__cplusplus)

#include <array>
#include <glm/glm.hpp>
#include <utilities/ping_pong_resource.hpp>

struct IrcacheIrradiancePendingSummation {
    daxa::TaskBufferView indirect_args_buf;
};

struct IrcacheRenderState {
    daxa::ExternalTaskBuffer ircache_meta_buf;

    daxa::ExternalTaskBuffer ircache_grid_meta_buf;
    daxa::ExternalTaskBuffer ircache_grid_meta_buf2;

    daxa::ExternalTaskBuffer ircache_entry_cell_buf;
    daxa::ExternalTaskBuffer ircache_spatial_buf;
    daxa::ExternalTaskBuffer ircache_irradiance_buf;
    daxa::ExternalTaskBuffer ircache_aux_buf;

    daxa::ExternalTaskBuffer ircache_life_buf;
    daxa::ExternalTaskBuffer ircache_pool_buf;
    daxa::ExternalTaskBuffer ircache_entry_indirection_buf;

    daxa::ExternalTaskBuffer ircache_reposition_proposal_buf;
    daxa::ExternalTaskBuffer ircache_reposition_proposal_count_buf;

    daxa::TaskBufferView ircache_buffers;

    bool pending_irradiance_sum;

    auto trace_irradiance(GpuContext &gpu_context, VoxelWorldBuffers &voxel_buffers, daxa::TaskImageView sky_cube, daxa::TaskImageView transmittance_lut) -> IrcacheIrradiancePendingSummation;
    void sum_up_irradiance_for_sampling(GpuContext &gpu_context, IrcacheIrradiancePendingSummation pending);
};

struct IrcacheRenderer {
    bool initialized = false;
    glm::vec3 grid_center{};
    std::array<glm::ivec3, IRCACHE_CASCADE_COUNT> cur_scroll{};
    std::array<glm::ivec3, IRCACHE_CASCADE_COUNT> prev_scroll{};
    size_t parity = 0;
    bool enable_scroll = true;

    PingPongBuffer ping_pong_ircache_grid_meta_buf;

    void update_eye_position(GpuInput &gpu_input);
    void next_frame();
    auto prepare(GpuContext &gpu_context) -> IrcacheRenderState;
};

#endif
