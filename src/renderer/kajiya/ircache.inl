#pragma once

#include <core.inl>
#include <application/input.inl>
#include <voxels/voxels.inl>
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

#define IRCACHE_USE_BUFFERS()                                                                \
    DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(IrcacheBuffers), ircache_buffers) \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE_CONCURRENT, ircache_meta_buf)                   \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE_CONCURRENT, ircache_grid_meta_buf)              \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE_CONCURRENT, ircache_entry_cell_buf)             \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE_CONCURRENT, ircache_spatial_buf)                \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE_CONCURRENT, ircache_irradiance_buf)             \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE_CONCURRENT, ircache_aux_buf)                    \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE_CONCURRENT, ircache_life_buf)                   \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE_CONCURRENT, ircache_pool_buf)                   \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE_CONCURRENT, ircache_entry_indirection_buf)      \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE_CONCURRENT, ircache_reposition_proposal_buf)    \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE_CONCURRENT, ircache_reposition_proposal_count_buf)

#define IRCACHE_USE_BUFFERS_PUSH_USES()                                                                                      \
    daxa_BufferPtr(IrcacheBuffers) ircache_buffers = push.uses.ircache_buffers;                                              \
    daxa_RWBufferPtr(IrcacheMetadata) ircache_meta_buf = deref(ircache_buffers).ircache_meta_buf;                            \
    daxa_RWBufferPtr(IrcacheCell) ircache_grid_meta_buf = deref(ircache_buffers).ircache_grid_meta_buf;                      \
    daxa_RWBufferPtr(daxa_u32) ircache_entry_cell_buf = deref(ircache_buffers).ircache_entry_cell_buf;                       \
    daxa_RWBufferPtr(VertexPacked) ircache_spatial_buf = deref(ircache_buffers).ircache_spatial_buf;                         \
    daxa_RWBufferPtr(daxa_f32vec4) ircache_irradiance_buf = deref(ircache_buffers).ircache_irradiance_buf;                   \
    daxa_RWBufferPtr(IrcacheAux) ircache_aux_buf = deref(ircache_buffers).ircache_aux_buf;                                   \
    daxa_RWBufferPtr(daxa_u32) ircache_life_buf = deref(ircache_buffers).ircache_life_buf;                                   \
    daxa_RWBufferPtr(daxa_u32) ircache_pool_buf = deref(ircache_buffers).ircache_pool_buf;                                   \
    daxa_RWBufferPtr(daxa_u32) ircache_entry_indirection_buf = deref(ircache_buffers).ircache_entry_indirection_buf;         \
    daxa_RWBufferPtr(VertexPacked) ircache_reposition_proposal_buf = deref(ircache_buffers).ircache_reposition_proposal_buf; \
    daxa_RWBufferPtr(daxa_u32) ircache_reposition_proposal_count_buf = deref(ircache_buffers).ircache_reposition_proposal_count_buf;

#define IRCACHE_BUFFER_USES_ASSIGN(TaskHeadName, ircache)                                                                         \
    daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_buffers, ircache.ircache_buffers}},                                     \
        daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_meta_buf, ircache.ircache_meta_buf}},                               \
        daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_grid_meta_buf, ircache.ircache_grid_meta_buf}},                     \
        daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_entry_cell_buf, ircache.ircache_entry_cell_buf}},                   \
        daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_spatial_buf, ircache.ircache_spatial_buf}},                         \
        daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_irradiance_buf, ircache.ircache_irradiance_buf}},                   \
        daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_aux_buf, ircache.ircache_aux_buf}},                                 \
        daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_life_buf, ircache.ircache_life_buf}},                               \
        daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_pool_buf, ircache.ircache_pool_buf}},                               \
        daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_entry_indirection_buf, ircache.ircache_entry_indirection_buf}},     \
        daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_reposition_proposal_buf, ircache.ircache_reposition_proposal_buf}}, \
        daxa::TaskViewVariant {                                                                                                   \
        std::pair { TaskHeadName::ircache_reposition_proposal_count_buf, ircache.ircache_reposition_proposal_count_buf }          \
    }

DAXA_DECL_TASK_HEAD_BEGIN(ClearIrcachePoolCompute, 2)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_pool_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_life_buf)
DAXA_DECL_TASK_HEAD_END
struct ClearIrcachePoolComputePush {
    DAXA_TH_BLOB(ClearIrcachePoolCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(IrcacheScrollCascadesCompute, 8)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(IrcacheCell), ircache_grid_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheCell), ircache_grid_meta_buf2)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_entry_cell_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_f32vec4), ircache_irradiance_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_pool_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_DECL_TASK_HEAD_END
struct IrcacheScrollCascadesComputePush {
    DAXA_TH_BLOB(IrcacheScrollCascadesCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(IrcachePrepareAgeDispatchCompute, 2)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32vec4), dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct IrcachePrepareAgeDispatchComputePush {
    DAXA_TH_BLOB(IrcachePrepareAgeDispatchCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(AgeIrcacheEntriesCompute, 11)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheCell), ircache_grid_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_entry_cell_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_pool_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VertexPacked), ircache_spatial_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VertexPacked), ircache_reposition_proposal_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_reposition_proposal_count_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_f32vec4), ircache_irradiance_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), entry_occupancy_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ, dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct AgeIrcacheEntriesComputePush {
    DAXA_TH_BLOB(AgeIrcacheEntriesCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(IrcacheCompactEntriesCompute, 5)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), entry_occupancy_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_entry_indirection_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ, dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct IrcacheCompactEntriesComputePush {
    DAXA_TH_BLOB(IrcacheCompactEntriesCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(IrcachePrepareTraceDispatchCompute, 2)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32vec4), dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct IrcachePrepareTraceDispatchComputePush {
    DAXA_TH_BLOB(IrcachePrepareTraceDispatchCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(IrcacheResetCompute, 7)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32vec4), ircache_irradiance_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheAux), ircache_aux_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ircache_entry_indirection_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ, dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct IrcacheResetComputePush {
    DAXA_TH_BLOB(IrcacheResetCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(IrcacheTraceAccessCompute, 7 + VOXEL_BUFFER_USE_N)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VertexPacked), ircache_spatial_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(VertexPacked), ircache_reposition_proposal_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(IrcacheAux), ircache_aux_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ircache_entry_indirection_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ, dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct IrcacheTraceAccessComputePush {
    DAXA_TH_BLOB(IrcacheTraceAccessCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(IrcacheValidateCompute, 14 + VOXEL_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VertexPacked), ircache_spatial_buf)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, CUBE, sky_cube_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(IrcacheCell), ircache_grid_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(VertexPacked), ircache_reposition_proposal_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(daxa_u32), ircache_reposition_proposal_count_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(IrcacheAux), ircache_aux_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(daxa_u32), ircache_pool_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ircache_entry_indirection_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(daxa_u32), ircache_entry_cell_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ, dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct IrcacheValidateComputePush {
    DAXA_TH_BLOB(IrcacheValidateCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(TraceIrradianceCompute, 14 + VOXEL_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VertexPacked), ircache_spatial_buf)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, CUBE, sky_cube_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(IrcacheCell), ircache_grid_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(VertexPacked), ircache_reposition_proposal_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(daxa_u32), ircache_reposition_proposal_count_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(IrcacheAux), ircache_aux_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(daxa_u32), ircache_pool_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ircache_entry_indirection_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(daxa_u32), ircache_entry_cell_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ, dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct TraceIrradianceComputePush {
    DAXA_TH_BLOB(TraceIrradianceCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(SumUpIrradianceCompute, 7)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_f32vec4), ircache_irradiance_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheAux), ircache_aux_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ircache_entry_indirection_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ, dispatch_args)
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
    daxa::TaskBuffer ircache_meta_buf;

    daxa::TaskBuffer ircache_grid_meta_buf;
    daxa::TaskBuffer ircache_grid_meta_buf2;

    daxa::TaskBuffer ircache_entry_cell_buf;
    daxa::TaskBuffer ircache_spatial_buf;
    daxa::TaskBuffer ircache_irradiance_buf;
    daxa::TaskBuffer ircache_aux_buf;

    daxa::TaskBuffer ircache_life_buf;
    daxa::TaskBuffer ircache_pool_buf;
    daxa::TaskBuffer ircache_entry_indirection_buf;

    daxa::TaskBuffer ircache_reposition_proposal_buf;
    daxa::TaskBuffer ircache_reposition_proposal_count_buf;

    daxa::TaskBufferView ircache_buffers;

    bool pending_irradiance_sum;

    auto trace_irradiance(RecordContext &record_ctx, VoxelWorldBuffers &voxel_buffers, daxa::TaskImageView sky_cube, daxa::TaskImageView transmittance_lut) -> IrcacheIrradiancePendingSummation;
    void sum_up_irradiance_for_sampling(RecordContext &record_ctx, IrcacheIrradiancePendingSummation pending);
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
    auto prepare(RecordContext &record_ctx) -> IrcacheRenderState;
};

#endif
