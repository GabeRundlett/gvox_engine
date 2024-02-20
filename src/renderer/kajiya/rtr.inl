#pragma once

#include <core.inl>
#include <renderer/core.inl>
#include <renderer/kajiya/ircache.inl>
#include <renderer/kajiya/rtdgi.inl>

DAXA_DECL_TASK_HEAD_BEGIN(RtrTraceCompute, 14 + VOXEL_BUFFER_USE_N + IRCACHE_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_i32), ranking_tile_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_i32), scambling_tile_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_i32), sobol_buf)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
IRCACHE_USE_BUFFERS()
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, gbuffer_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, rtdgi_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, sky_lut)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, out0_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, out1_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, out2_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, rng_out_tex)
DAXA_DECL_TASK_HEAD_END
struct RtrTraceComputePush {
    daxa_f32vec4 gbuffer_tex_size;
    DAXA_TH_BLOB(RtrTraceCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(RtrValidateCompute, 13 + VOXEL_BUFFER_USE_N + IRCACHE_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
IRCACHE_USE_BUFFERS()
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, gbuffer_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, rtdgi_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, sky_lut)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, refl_restir_invalidity_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, ray_orig_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, ray_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, rng_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, irradiance_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, reservoir_history_tex)
DAXA_DECL_TASK_HEAD_END
struct RtrValidateComputePush {
    daxa_f32vec4 gbuffer_tex_size;
    DAXA_TH_BLOB(RtrValidateCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(RtrRestirTemporalCompute, 21)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, gbuffer_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, half_view_normal_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, candidate0_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, candidate1_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, candidate2_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, irradiance_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, ray_orig_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, ray_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, rng_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reservoir_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, hit_normal_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, irradiance_out_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, ray_orig_output_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, ray_output_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, rng_output_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, hit_normal_output_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, reservoir_out_tex)
DAXA_DECL_TASK_HEAD_END
struct RtrRestirTemporalComputePush {
    daxa_f32vec4 gbuffer_tex_size;
    DAXA_TH_BLOB(RtrRestirTemporalCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(RtrRestirResolveCompute, 21)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, gbuffer_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, hit0_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, hit1_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, hit2_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, half_view_normal_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, half_depth_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, ray_len_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, restir_irradiance_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, restir_ray_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, restir_reservoir_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, restir_ray_orig_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, restir_hit_normal_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_3D, blue_noise_vec2)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, output_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, ray_len_output_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, rtr_debug_image)
DAXA_DECL_TASK_HEAD_END
struct RtrRestirResolveComputePush {
    daxa_f32vec4 output_tex_size;
    DAXA_TH_BLOB(RtrRestirResolveCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(RtrTemporalFilterCompute, 10)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, ray_len_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, refl_restir_invalidity_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, gbuffer_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, output_tex)
DAXA_DECL_TASK_HEAD_END
struct RtrTemporalFilterComputePush {
    daxa_f32vec4 output_tex_size;
    DAXA_TH_BLOB(RtrTemporalFilterCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(RtrSpatialFilterCompute, 7)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_i32vec2), spatial_resolve_offsets)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, geometric_normal_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, output_tex)
DAXA_DECL_TASK_HEAD_END
struct RtrSpatialFilterComputePush {
    daxa_f32vec4 output_tex_size;
    DAXA_TH_BLOB(RtrSpatialFilterCompute, uses)
};

#if defined(__cplusplus)

#include <spp64.hpp>

const std::array<int, 2 * 16 * 4 * 8> SPATIAL_RESOLVE_OFFSETS = {
    // clang-format off
    0, 0,     -1, -1,    2, -1,     -2, -2,   -2, 2,    2, 2,     0, -3,     -3, 0,  
    3, 0,     3, 1,      -1, 3,     1, 3,     2, -3,    -3, -2,   1, 4,      -3, 3,   
    0, 0,     0, 1,      -1, 1,     1, 1,     0, -2,    -2, 1,    1, 2,      2, -2, 
    1, -3,    -4, 0,     4, 0,      -1, -4,   -4, -1,   -4, 1,    -3, -3,    3, 3,   
    0, 0,     0, -1,     -1, 0,     1, -1,    -2, -1,   2, 1,     -1, 2,     0, 3,   
    3, -1,    -2, -3,    3, -2,     -3, 2,    0, -4,    4, 1,     -1, 4,     3, -3,
    0, 0,     1, 0,      -2, 0,     2, 0,     0, 2,     -1, -2,   1, -2,     -1, -3,
    -3, -1,   -3, 1,     3, 2,      -2, 3,    2, 3,     0, 4,     1, -4,     4, -1,
    0, 0,     0, 1,      -1, -1,    1, -2,    2, -2,    3, 0,     1, 3,      -3, 2,
    -4, 0,    -1, -4,    -1, 4,     -3, -3,   4, 2,     1, -5,    -4, -4,    4, -4,
    0, 0,     0, -1,     1, 0,      -2, 1,    2, 2,     0, 3,     -3, -1,    -3, -2,
    0, -4,    4, -1,     -3, 3,     3, 3,     -2, 4,    3, -4,    5, -1,     -2, -5,
    0, 0,     -1, 1,     -1, -2,    2, 1,     0, -3,    3, -1,    -3, 1,     -1, 3,
    3, -2,    1, 4,      3, -3,     -4, -2,   0, -5,    4, 3,     -5, -1,    5, 1,
    0, 0,     -1, 0,     1, 1,      -2, 0,    2, -1,    1, 2,     -2, -2,    0, 4,
    1, -4,    -4, 1,     4, 1,      -2, -4,   2, -4,    -4, 2,    -4, -3,    -3, 4,
    0, 0,     0, -1,     1, 1,      -2, -2,   2, -2,    -3, 0,    3, 1,      -1, 3,
    -1, 4,    -2, -4,    5, 0,      -4, 3,    3, 4,     -5, -1,   2, -5,     4, -4,
    0, 0,     1, -1,     -2, 1,     1, 2,     1, -3,    -3, -2,   -3, 2,     4, 0,
    4, 2,     2, 4,      3, -4,     -1, -5,   -5, -3,   -3, 5,    6, -2,     -6, 2,
    0, 0,     0, 1,      0, -2,     -1, -2,   -2, 2,    3, -1,    -4, 0,     1, 4,
    3, 3,     4, -2,     0, -5,     -3, -4,   1, -5,    -5, 1,    -1, 5,     1, 5,
    0, 0,     -1, 0,     0, 2,      2, 1,     0, -3,    -3, -1,   2, -3,     4, -1,
    -3, -3,   -4, 2,     -2, 4,     5, 2,     4, 4,     5, -3,    -5, 3,     3, 5,
    0, 0,     0, 1,      1, -1,     0, -3,    -3, 0,    3, 1,     0, 4,      -4, -3,
    -4, 3,    4, 3,      5, -1,     -3, -5,   3, -5,    5, -3,    -3, 5,     -6, 0,
    0, 0,     -2, -1,    2, -1,     -2, -2,   -2, 2,    1, 3,     -4, 1,     4, 1,
    3, -3,    -1, -5,    2, -5,     -2, 5,    2, 5,     0, -6,    -6, -1,    6, 2,
    0, 0,     0, 2,      -2, 1,     1, -3,    3, -1,    3, 2,     -1, -4,    -4, -2,
    -2, 4,    5, -2,     -4, -4,    3, 5,     0, 6,     -6, -2,   -6, 2,     4, -5,
    0, 0,     0, -1,     1, 1,      -1, -2,   -2, 3,    1, 4,     -3, -4,    4, -3,
    5, 0,     1, -5,     -5, -1,    -5, 2,    5, 3,     6, 1,     1, 6,      -4, 6,
    0, 0,     1, -3,     1, 3,      -3, 2,    4, -1,    3, -4,    -5, 0,     -1, -5,
    5, 2,     -2, 5,     -4, -4,    1, 6,     -5, 5,    3, 7,     -8, 1,     7, 4,
    0, 0,     -1, -1,    2, -1,     -1, 2,    3, 1,     -4, -2,   2, 4,      -2, -5,
    -5, 2,    6, -2,     -2, 6,     5, -4,    6, 3,     0, -7,    5, 5,      4, -6,
    0, 0,     2, 2,      -1, -3,    -3, -1,   0, 4,     -3, 3,    4, -2,     1, -5,
    -5, -3,   6, 0,      -6, 1,     3, -6,    3, 6,     -1, 7,    -4, -7,    -7, -4,
    0, 0,     -1, 0,     0, 2,      -2, -2,   2, -2,    4, 1,     -2, 4,     4, 3,
    -3, -5,   5, -3,     -6, 0,     1, -6,    -5, 4,    7, 0,     1, 7,      -3, -7,
    0, 0,     0, -1,     -2, 2,     3, 2,     -3, -3,   -5, 2,    0, -6,     6, 1,
    -1, 6,    -6, -2,    6, -3,     3, 6,     4, -6,    -5, 6,    7, 4,      -8, 2,
    0, 0,     1, 2,      1, -3,     -4, 0,    4, -1,    -3, 4,    3, -5,     1, 6,
    -2, -6,   -5, -5,    -8, 0,     8, -1,    4, 7,     7, -5,    2, -9,     -7, -6,
    0, 0,     -1, 0,     -2, -2,    0, 3,     3, -2,    -1, -4,   4, 3,      5, -2,
    1, -6,    -6, 1,     -2, 6,     5, 4,     -6, -3,   -6, 4,    8, 0,      -4, -7,
    0, 0,     2, 0,      -2, 1,     -4, -1,   1, 4,     3, -4,    5, 1,      -4, 4,
    3, 5,     -4, -5,    -7, -2,    7, -2,    2, -8,    8, 3,     -3, 8,     5, -7,
    0, 0,     0, -1,     4, -3,     -4, 3,    0, 5,     4, 4,     -3, -5,    1, -6,
    -6, -2,   7, -1,     0, 8,      -8, 3,    8, 3,     5, 7,     7, -6,     -5, 8,
    0, 0,     2, 2,      0, -3,     3, 0,     -3, -1,   -1, 3,    -6, 1,     4, -5,
    6, 3,     -6, -5,    -6, 5,     0, -8,    -9, -1,   -2, 9,    5, -8,     -3, -9,
    0, 0,     -2, 1,     -1, -4,    5, 0,     2, 6,     -3, 6,    -4, -6,    3, -7,
    -7, -3,   7, -3,     8, 1,      8, 6,     -10, 1,   -7, 8,    -10, 4,    0, 11,
    0, 0,     1, 2,      2, -2,     -3, -3,   -3, 3,    3, -4,    -5, 0,     3, 4,
    -2, 6,    -1, -7,    6, 5,      -8, 2,    2, 8,     7, -5,    -7, 5,     2, -9,
    0, 0,     0, -2,     2, 2,      -4, -2,   -2, 5,    -7, 1,    4, -7,     7, -4,
    6, 7,     9, 3,      -6, -9,    -9, 6,    0, -11,   0, 11,    -10, -6,   -7, 10,
    0, 0,     -3, 1,     3, -2,     1, -7,    -6, 4,    2, 7,     -7, -4,    8, -1,
    -2, -8,   -10, 3,    10, -4,    -4, 10,   4, 10,    -11, -2,  12, 1,     10, 7,
    0, 0,     -1, 2,     -1, -4,    4, 1,     4, -4,    4, 5,     -5, -5,    7, 3,
    -3, 7,    -8, -1,    -6, 6,     0, 9,     3, -10,   7, -8,    -9, -8,    -12, 3,
    0, 0,     1, -5,     -5, 1,     1, 5,     -3, -5,   6, 0,     -4, -8,    -10, 0,
    10, -1,   5, 9,      8, -7,     9, 6,     -11, -4,  2, 12,    -7, -10,   -4, 12,
    // clang-format on
};

struct TracedRtr {
    daxa::TaskImageView resolved_tex;
    daxa::TaskImageView temporal_output_tex;
    daxa::TaskImageView history_tex;
    daxa::TaskImageView ray_len_tex;
    daxa::TaskImageView refl_restir_invalidity_tex;

    auto filter(
        RecordContext &record_ctx,
        GbufferDepth &gbuffer_depth,
        daxa::TaskImageView reprojection_map,
        daxa::TaskBufferView spatial_resolve_offsets) -> daxa::TaskImageView {
        record_ctx.add(ComputeTask<RtrTemporalFilterCompute, RtrTemporalFilterComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"kajiya/rtr/temporal_filter.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{RtrTemporalFilterCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
                daxa::TaskViewVariant{std::pair{RtrTemporalFilterCompute::globals, record_ctx.gpu_context->task_globals_buffer}},
                daxa::TaskViewVariant{std::pair{RtrTemporalFilterCompute::input_tex, this->resolved_tex}},
                daxa::TaskViewVariant{std::pair{RtrTemporalFilterCompute::history_tex, this->history_tex}},
                daxa::TaskViewVariant{std::pair{RtrTemporalFilterCompute::depth_tex, gbuffer_depth.depth.current()}},
                daxa::TaskViewVariant{std::pair{RtrTemporalFilterCompute::ray_len_tex, this->ray_len_tex}},
                daxa::TaskViewVariant{std::pair{RtrTemporalFilterCompute::reprojection_tex, reprojection_map}},
                daxa::TaskViewVariant{std::pair{RtrTemporalFilterCompute::refl_restir_invalidity_tex, this->refl_restir_invalidity_tex}},
                daxa::TaskViewVariant{std::pair{RtrTemporalFilterCompute::gbuffer_tex, gbuffer_depth.gbuffer}},
                daxa::TaskViewVariant{std::pair{RtrTemporalFilterCompute::output_tex, this->temporal_output_tex}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, RtrTemporalFilterComputePush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(RtrTemporalFilterCompute::input_tex).ids[0]).value();
                auto const out_image_info = ti.device.info_image(ti.get(RtrTemporalFilterCompute::output_tex).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                push.output_tex_size = extent_inv_extent_2d(out_image_info);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });
        debug_utils::DebugDisplay::add_pass({.name = "rtr temporal filter", .task_image_id = this->temporal_output_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        auto final_resolved_tex = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::B10G11R11_UFLOAT_PACK32,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "final_resolved_tex",
        });

        record_ctx.add(ComputeTask<RtrSpatialFilterCompute, RtrSpatialFilterComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"kajiya/rtr/spatial_cleanup.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{RtrSpatialFilterCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
                daxa::TaskViewVariant{std::pair{RtrSpatialFilterCompute::globals, record_ctx.gpu_context->task_globals_buffer}},
                daxa::TaskViewVariant{std::pair{RtrSpatialFilterCompute::spatial_resolve_offsets, spatial_resolve_offsets}},
                daxa::TaskViewVariant{std::pair{RtrSpatialFilterCompute::input_tex, this->temporal_output_tex}},
                daxa::TaskViewVariant{std::pair{RtrSpatialFilterCompute::depth_tex, gbuffer_depth.depth.current()}},
                daxa::TaskViewVariant{std::pair{RtrSpatialFilterCompute::geometric_normal_tex, gbuffer_depth.geometric_normal}},
                daxa::TaskViewVariant{std::pair{RtrSpatialFilterCompute::output_tex, final_resolved_tex}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, RtrSpatialFilterComputePush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(RtrSpatialFilterCompute::input_tex).ids[0]).value();
                auto const out_image_info = ti.device.info_image(ti.get(RtrSpatialFilterCompute::output_tex).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                push.output_tex_size = extent_inv_extent_2d(out_image_info);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });
        debug_utils::DebugDisplay::add_pass({.name = "rtr spatial cleanup", .task_image_id = final_resolved_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        return final_resolved_tex;
    }
};

struct RtrRenderer {
    PingPongImage temporal_tex;
    PingPongImage ray_len_tex;
    PingPongImage temporal_irradiance_tex;
    PingPongImage temporal_ray_orig_tex;
    PingPongImage temporal_ray_tex;
    PingPongImage pp_temporal_reservoir_tex;
    PingPongImage temporal_rng_tex;
    PingPongImage temporal_hit_normal_tex;
    daxa::TaskBuffer spatial_resolve_offsets_buf;
    daxa::TaskBuffer ranking_tile_buf;
    daxa::TaskBuffer scambling_tile_buf;
    daxa::TaskBuffer sobol_buf;

    bool buffers_uploaded = false;

    void next_frame() {
        temporal_tex.swap();
        ray_len_tex.swap();
        temporal_irradiance_tex.swap();
        temporal_ray_orig_tex.swap();
        temporal_ray_tex.swap();
        pp_temporal_reservoir_tex.swap();
        temporal_rng_tex.swap();
        temporal_hit_normal_tex.swap();
    }

    auto trace(
        RecordContext &record_ctx,
        GbufferDepth &gbuffer_depth,
        daxa::TaskImageView reprojection_map,
        daxa::TaskImageView sky_lut,
        daxa::TaskImageView transmittance_lut,
        VoxelWorldBuffers &voxel_buffers,
        daxa::TaskImageView rtdgi_irradiance,
        RtdgiCandidates rtdgi_candidates,
        IrcacheRenderState &ircache) -> TracedRtr {
        auto [refl0_tex, refl1_tex, refl2_tex] = rtdgi_candidates;

        if (!buffers_uploaded) {
            buffers_uploaded = true;

            spatial_resolve_offsets_buf = temporal_storage_buffer(record_ctx, "rtr.spatial_resolve_offsets_buf", sizeof(SPATIAL_RESOLVE_OFFSETS));
            ranking_tile_buf = temporal_storage_buffer(record_ctx, "rtr.ranking_tile_buf", sizeof(RANKING_TILE));
            scambling_tile_buf = temporal_storage_buffer(record_ctx, "rtr.scambling_tile_buf", sizeof(SCRAMBLING_TILE));
            sobol_buf = temporal_storage_buffer(record_ctx, "rtr.sobol_buf", sizeof(SOBOL));

            daxa::TaskGraph temp_task_graph = daxa::TaskGraph({.device = record_ctx.gpu_context->device, .name = "temp_task_graph"});
            temp_task_graph.use_persistent_buffer(spatial_resolve_offsets_buf);
            temp_task_graph.use_persistent_buffer(ranking_tile_buf);
            temp_task_graph.use_persistent_buffer(scambling_tile_buf);
            temp_task_graph.use_persistent_buffer(sobol_buf);
            temp_task_graph.add_task({
                .attachments = {
                    daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, spatial_resolve_offsets_buf),
                    daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, ranking_tile_buf),
                    daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, scambling_tile_buf),
                    daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, sobol_buf),
                },
                .task = [](daxa::TaskInterface const &ti) {
                    auto staging_buffer = ti.device.create_buffer({
                        .size = sizeof(SPATIAL_RESOLVE_OFFSETS) + sizeof(RANKING_TILE) + sizeof(SCRAMBLING_TILE) + sizeof(SOBOL),
                        .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                        .name = "staging_buffer",
                    });
                    auto *buffer_ptr = ti.device.get_host_address_as<int>(staging_buffer).value();
                    auto offset = size_t{0};

                    auto spatial_resolve_offset = offset * sizeof(int);
                    std::copy(SPATIAL_RESOLVE_OFFSETS.begin(), SPATIAL_RESOLVE_OFFSETS.end(), buffer_ptr + offset);
                    offset += SPATIAL_RESOLVE_OFFSETS.size();

                    auto ranking_tile_offset = offset * sizeof(int);
                    std::copy(RANKING_TILE.begin(), RANKING_TILE.end(), buffer_ptr + offset);
                    offset += RANKING_TILE.size();

                    auto scrambling_tile_offset = offset * sizeof(int);
                    std::copy(SCRAMBLING_TILE.begin(), SCRAMBLING_TILE.end(), buffer_ptr + offset);
                    offset += SCRAMBLING_TILE.size();

                    auto sobel_offset = offset * sizeof(int);
                    std::copy(SOBOL.begin(), SOBOL.end(), buffer_ptr + offset);
                    offset += SOBOL.size();

                    ti.recorder.pipeline_barrier({
                        .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
                    });
                    ti.recorder.destroy_buffer_deferred(staging_buffer);
                    ti.recorder.copy_buffer_to_buffer({
                        .src_buffer = staging_buffer,
                        .dst_buffer = ti.get(daxa::TaskBufferAttachmentIndex{0}).ids[0],
                        .src_offset = spatial_resolve_offset,
                        .size = SPATIAL_RESOLVE_OFFSETS.size() * sizeof(int),
                    });
                    ti.recorder.copy_buffer_to_buffer({
                        .src_buffer = staging_buffer,
                        .dst_buffer = ti.get(daxa::TaskBufferAttachmentIndex{1}).ids[0],
                        .src_offset = ranking_tile_offset,
                        .size = RANKING_TILE.size() * sizeof(int),
                    });
                    ti.recorder.copy_buffer_to_buffer({
                        .src_buffer = staging_buffer,
                        .dst_buffer = ti.get(daxa::TaskBufferAttachmentIndex{2}).ids[0],
                        .src_offset = scrambling_tile_offset,
                        .size = SCRAMBLING_TILE.size() * sizeof(int),
                    });
                    ti.recorder.copy_buffer_to_buffer({
                        .src_buffer = staging_buffer,
                        .dst_buffer = ti.get(daxa::TaskBufferAttachmentIndex{3}).ids[0],
                        .src_offset = sobel_offset,
                        .size = SOBOL.size() * sizeof(int),
                    });
                },
                .name = "upload_lookup_tables",
            });
            temp_task_graph.submit({});
            temp_task_graph.complete({});
            temp_task_graph.execute({});
        } else {
            record_ctx.task_graph.use_persistent_buffer(spatial_resolve_offsets_buf);
            record_ctx.task_graph.use_persistent_buffer(ranking_tile_buf);
            record_ctx.task_graph.use_persistent_buffer(scambling_tile_buf);
            record_ctx.task_graph.use_persistent_buffer(sobol_buf);
        }

        auto gbuffer_half_res = daxa_u32vec2{(record_ctx.render_resolution.x + 1) / 2, (record_ctx.render_resolution.y + 1) / 2};

        temporal_rng_tex = PingPongImage{};
        auto [rng_output_tex, rng_history_tex] = temporal_rng_tex.get(
            *record_ctx.gpu_context,
            {
                .format = daxa::Format::R32_UINT,
                .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = "temporal_rng_tex",
            });
        record_ctx.task_graph.use_persistent_image(rng_output_tex);
        record_ctx.task_graph.use_persistent_image(rng_history_tex);
        clear_task_images(record_ctx.gpu_context->device, std::array{rng_output_tex, rng_history_tex});

        record_ctx.add(ComputeTask<RtrTraceCompute, RtrTraceComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"kajiya/rtr/trace_reflection.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::globals, record_ctx.gpu_context->task_globals_buffer}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::ranking_tile_buf, ranking_tile_buf}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::scambling_tile_buf, scambling_tile_buf}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::sobol_buf, sobol_buf}},
                VOXELS_BUFFER_USES_ASSIGN(RtrTraceCompute, voxel_buffers),
                IRCACHE_BUFFER_USES_ASSIGN(RtrTraceCompute, ircache),
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::gbuffer_tex, gbuffer_depth.gbuffer}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::depth_tex, gbuffer_depth.depth.current()}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::rtdgi_tex, rtdgi_irradiance}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::sky_lut, sky_lut}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::transmittance_lut, transmittance_lut}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::out0_tex, refl0_tex}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::out1_tex, refl1_tex}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::out2_tex, refl2_tex}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::rng_out_tex, rng_output_tex}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, RtrTraceComputePush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(RtrTraceCompute::depth_tex).ids[0]).value();
                auto const out_image_info = ti.device.info_image(ti.get(RtrTraceCompute::out0_tex).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                push.gbuffer_tex_size = extent_inv_extent_2d(image_info);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(out_image_info.size.x + 7) / 8, (out_image_info.size.y + 7) / 8});
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "rtr trace", .task_image_id = refl0_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        auto half_view_normal_tex = gbuffer_depth.get_downscaled_view_normal(record_ctx);
        auto half_depth_tex = gbuffer_depth.get_downscaled_depth(record_ctx);

        temporal_ray_orig_tex = PingPongImage{};
        auto [ray_orig_output_tex, ray_orig_history_tex] = temporal_ray_orig_tex.get(
            *record_ctx.gpu_context,
            {
                .format = daxa::Format::R32G32B32A32_SFLOAT,
                .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = "rtr.temporal_ray_orig_tex",
            });
        record_ctx.task_graph.use_persistent_image(ray_orig_output_tex);
        record_ctx.task_graph.use_persistent_image(ray_orig_history_tex);
        clear_task_images(record_ctx.gpu_context->device, std::array{ray_orig_output_tex, ray_orig_history_tex});

        auto refl_restir_invalidity_tex = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R8_UNORM,
            .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
            .name = "refl_restir_invalidity_tex",
        });

        auto irradiance_tex = daxa::TaskImageView{};
        auto ray_tex = daxa::TaskImageView{};
        auto temporal_reservoir_tex = daxa::TaskImageView{};
        auto restir_hit_normal_tex = daxa::TaskImageView{};
        {
            temporal_hit_normal_tex = PingPongImage{};
            auto [hit_normal_output_tex, hit_normal_history_tex] = temporal_hit_normal_tex.get(
                *record_ctx.gpu_context,
                {
                    .format = daxa::Format::R8G8B8A8_UNORM,
                    .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                    .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                    .name = "rtr.temporal_hit_normal_tex",
                });
            record_ctx.task_graph.use_persistent_image(hit_normal_output_tex);
            record_ctx.task_graph.use_persistent_image(hit_normal_history_tex);
            clear_task_images(record_ctx.gpu_context->device, std::array{hit_normal_output_tex, hit_normal_history_tex});

            temporal_irradiance_tex = PingPongImage{};
            auto [irradiance_output_tex, irradiance_history_tex] = temporal_irradiance_tex.get(
                *record_ctx.gpu_context,
                {
                    .format = daxa::Format::R16G16B16A16_SFLOAT,
                    .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                    .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                    .name = "rtr.temporal_irradiance_tex",
                });
            record_ctx.task_graph.use_persistent_image(irradiance_output_tex);
            record_ctx.task_graph.use_persistent_image(irradiance_history_tex);
            clear_task_images(record_ctx.gpu_context->device, std::array{irradiance_output_tex, irradiance_history_tex});

            pp_temporal_reservoir_tex = PingPongImage{};
            auto [reservoir_output_tex, reservoir_history_tex] = pp_temporal_reservoir_tex.get(
                *record_ctx.gpu_context,
                {
                    .format = daxa::Format::R32G32_UINT,
                    .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                    .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                    .name = "rtr.temporal_reservoir_tex",
                });
            record_ctx.task_graph.use_persistent_image(reservoir_output_tex);
            record_ctx.task_graph.use_persistent_image(reservoir_history_tex);
            clear_task_images(record_ctx.gpu_context->device, std::array{reservoir_output_tex, reservoir_history_tex});

            temporal_ray_tex = PingPongImage{};
            auto [ray_output_tex, ray_history_tex] = temporal_ray_tex.get(
                *record_ctx.gpu_context,
                {
                    .format = daxa::Format::R16G16B16A16_SFLOAT,
                    .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                    .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                    .name = "rtr.temporal_ray_tex",
                });
            record_ctx.task_graph.use_persistent_image(ray_output_tex);
            record_ctx.task_graph.use_persistent_image(ray_history_tex);
            clear_task_images(record_ctx.gpu_context->device, std::array{ray_output_tex, ray_history_tex});

            record_ctx.add(ComputeTask<RtrValidateCompute, RtrValidateComputePush, NoTaskInfo>{
                .source = daxa::ShaderFile{"kajiya/rtr/reflection_validate.comp.glsl"},
                .views = std::array{
                    daxa::TaskViewVariant{std::pair{RtrValidateCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
                    daxa::TaskViewVariant{std::pair{RtrValidateCompute::globals, record_ctx.gpu_context->task_globals_buffer}},
                    VOXELS_BUFFER_USES_ASSIGN(RtrValidateCompute, voxel_buffers),
                    IRCACHE_BUFFER_USES_ASSIGN(RtrValidateCompute, ircache),
                    daxa::TaskViewVariant{std::pair{RtrValidateCompute::gbuffer_tex, gbuffer_depth.gbuffer}},
                    daxa::TaskViewVariant{std::pair{RtrValidateCompute::depth_tex, gbuffer_depth.depth.current()}},
                    daxa::TaskViewVariant{std::pair{RtrValidateCompute::rtdgi_tex, rtdgi_irradiance}},
                    daxa::TaskViewVariant{std::pair{RtrValidateCompute::sky_lut, sky_lut}},
                    daxa::TaskViewVariant{std::pair{RtrValidateCompute::transmittance_lut, transmittance_lut}},
                    daxa::TaskViewVariant{std::pair{RtrValidateCompute::refl_restir_invalidity_tex, refl_restir_invalidity_tex}},
                    daxa::TaskViewVariant{std::pair{RtrValidateCompute::ray_orig_history_tex, ray_orig_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtrValidateCompute::ray_history_tex, ray_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtrValidateCompute::rng_history_tex, rng_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtrValidateCompute::irradiance_history_tex, irradiance_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtrValidateCompute::reservoir_history_tex, reservoir_history_tex}},
                },
                .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, RtrValidateComputePush &push, NoTaskInfo const &) {
                    auto const image_info = ti.device.info_image(ti.get(RtrValidateCompute::depth_tex).ids[0]).value();
                    auto const out_image_info = ti.device.info_image(ti.get(RtrValidateCompute::reservoir_history_tex).ids[0]).value();
                    ti.recorder.set_pipeline(pipeline);
                    push.gbuffer_tex_size = extent_inv_extent_2d(image_info);
                    set_push_constant(ti, push);
                    ti.recorder.dispatch({(out_image_info.size.x + 15) / 16, (out_image_info.size.y + 15) / 16});
                },
            });

            debug_utils::DebugDisplay::add_pass({.name = "rtr validate", .task_image_id = refl_restir_invalidity_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            record_ctx.add(ComputeTask<RtrRestirTemporalCompute, RtrRestirTemporalComputePush, NoTaskInfo>{
                .source = daxa::ShaderFile{"kajiya/rtr/rtr_restir_temporal.comp.glsl"},
                .views = std::array{
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::globals, record_ctx.gpu_context->task_globals_buffer}},
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::gbuffer_tex, gbuffer_depth.gbuffer}},
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::half_view_normal_tex, half_view_normal_tex}},
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::depth_tex, gbuffer_depth.depth.current()}},
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::candidate0_tex, refl0_tex}},
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::candidate1_tex, refl1_tex}},
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::candidate2_tex, refl2_tex}},
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::irradiance_history_tex, irradiance_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::ray_orig_history_tex, ray_orig_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::ray_history_tex, ray_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::rng_history_tex, rng_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::reservoir_history_tex, reservoir_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::reprojection_tex, reprojection_map}},
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::hit_normal_history_tex, hit_normal_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::irradiance_out_tex, irradiance_output_tex}},
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::ray_orig_output_tex, ray_orig_output_tex}},
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::ray_output_tex, ray_output_tex}},
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::rng_output_tex, rng_output_tex}},
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::hit_normal_output_tex, hit_normal_output_tex}},
                    daxa::TaskViewVariant{std::pair{RtrRestirTemporalCompute::reservoir_out_tex, reservoir_output_tex}},
                },
                .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, RtrRestirTemporalComputePush &push, NoTaskInfo const &) {
                    auto const image_info = ti.device.info_image(ti.get(RtrRestirTemporalCompute::gbuffer_tex).ids[0]).value();
                    auto const out_image_info = ti.device.info_image(ti.get(RtrRestirTemporalCompute::irradiance_out_tex).ids[0]).value();
                    ti.recorder.set_pipeline(pipeline);
                    push.gbuffer_tex_size = extent_inv_extent_2d(image_info);
                    set_push_constant(ti, push);
                    ti.recorder.dispatch({(out_image_info.size.x + 7) / 8, (out_image_info.size.y + 7) / 8});
                },
            });

            debug_utils::DebugDisplay::add_pass({.name = "rtr restir temporal", .task_image_id = irradiance_output_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            irradiance_tex = irradiance_output_tex;
            ray_tex = ray_output_tex;
            temporal_reservoir_tex = reservoir_output_tex;
            restir_hit_normal_tex = hit_normal_output_tex;
        }

        auto resolved_tex = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::B10G11R11_UFLOAT_PACK32,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "resolved_tex",
        });

        temporal_tex = PingPongImage{};
        auto [temporal_output_tex, history_tex] = temporal_tex.get(
            *record_ctx.gpu_context,
            {
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = "rtr.temporal_tex",
            });
        record_ctx.task_graph.use_persistent_image(temporal_output_tex);
        record_ctx.task_graph.use_persistent_image(history_tex);
        clear_task_images(record_ctx.gpu_context->device, std::array{temporal_output_tex, history_tex});

        ray_len_tex = PingPongImage{};
        auto [ray_len_output_tex, ray_len_history_tex] = ray_len_tex.get(
            *record_ctx.gpu_context,
            {
                .format = daxa::Format::R16G16_SFLOAT,
                .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = "rtr.ray_len_tex",
            });
        record_ctx.task_graph.use_persistent_image(ray_len_output_tex);
        record_ctx.task_graph.use_persistent_image(ray_len_history_tex);
        clear_task_images(record_ctx.gpu_context->device, std::array{ray_len_output_tex, ray_len_history_tex});

        auto rtr_debug_image = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R32G32B32A32_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "rtr_debug_image",
        });

        record_ctx.add(ComputeTask<RtrRestirResolveCompute, RtrRestirResolveComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"kajiya/rtr/resolve.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::globals, record_ctx.gpu_context->task_globals_buffer}},
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::gbuffer_tex, gbuffer_depth.gbuffer}},
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::depth_tex, gbuffer_depth.depth.current()}},
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::hit0_tex, refl0_tex}},
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::hit1_tex, refl1_tex}},
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::hit2_tex, refl2_tex}},
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::history_tex, history_tex}},
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::reprojection_tex, reprojection_map}},
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::half_view_normal_tex, half_view_normal_tex}},
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::half_depth_tex, half_depth_tex}},
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::ray_len_history_tex, ray_len_history_tex}},
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::restir_irradiance_tex, irradiance_tex}},
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::restir_ray_tex, ray_tex}},
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::restir_reservoir_tex, temporal_reservoir_tex}},
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::restir_ray_orig_tex, ray_orig_output_tex}},
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::restir_hit_normal_tex, restir_hit_normal_tex}},
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::blue_noise_vec2, record_ctx.gpu_context->task_blue_noise_vec2_image}},
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::output_tex, resolved_tex}},
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::ray_len_output_tex, ray_len_output_tex}},
                daxa::TaskViewVariant{std::pair{RtrRestirResolveCompute::rtr_debug_image, rtr_debug_image}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, RtrRestirResolveComputePush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(RtrRestirResolveCompute::output_tex).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                push.output_tex_size = extent_inv_extent_2d(image_info);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "rtr restir resolve", .task_image_id = resolved_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "rtr debug", .task_image_id = rtr_debug_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        return {
            resolved_tex,
            temporal_output_tex,
            history_tex,
            ray_len_output_tex,
            refl_restir_invalidity_tex,
        };
    }
};

#endif
