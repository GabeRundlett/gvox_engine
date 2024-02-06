#pragma once

#include <shared/core.inl>
#include <shared/renderer/core.inl>
#include <shared/renderer/ircache.inl>
#include <shared/renderer/rtdgi.inl>

DAXA_DECL_TASK_HEAD_BEGIN(RtrTraceCompute, 13 + VOXEL_BUFFER_USE_N + IRCACHE_BUFFER_USE_N)
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
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, sky_cube_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, out0_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, out1_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, out2_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, rng_out_tex)
DAXA_DECL_TASK_HEAD_END
struct RtrTraceComputePush {
    daxa_f32vec4 gbuffer_tex_size;
    daxa_u32 reuse_rtdgi_rays;
    DAXA_TH_BLOB(RtrTraceCompute, uses)
};

#if defined(__cplusplus)

#include <shared/renderer/spp64.hpp>

struct TracedRtr {
    daxa::TaskImageView resolved_tex;
    daxa::TaskImageView temporal_output_tex;
    daxa::TaskImageView history_tex;
    daxa::TaskImageView ray_len_tex;
    daxa::TaskImageView refl_restir_invalidity_tex;
};

struct RtrRenderer {
    PingPongImage temporal_tex;
    PingPongImage ray_len_tex;
    PingPongImage temporal_irradiance_tex;
    PingPongImage temporal_ray_orig_tex;
    PingPongImage temporal_ray_tex;
    PingPongImage temporal_reservoir_tex;
    PingPongImage temporal_rng_tex;
    PingPongImage temporal_hit_normal_tex;
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
        temporal_reservoir_tex.swap();
        temporal_rng_tex.swap();
        temporal_hit_normal_tex.swap();
    }

    auto trace(
        RecordContext &record_ctx,
        GbufferDepth &gbuffer_depth,
        daxa::TaskImageView reprojection_map,
        daxa::TaskImageView sky_cube,
        VoxelWorld::Buffers &voxel_buffers,
        daxa::TaskImageView rtdgi_irradiance,
        RtdgiCandidates rtdgi_candidates,
        IrcacheRenderState &ircache) -> TracedRtr {
        auto [refl0_tex, refl1_tex, refl2_tex] = rtdgi_candidates;

        if (!buffers_uploaded) {
            buffers_uploaded = true;

            ranking_tile_buf = temporal_storage_buffer(record_ctx, "rtr.ranking_tile_buf", sizeof(RANKING_TILE));
            scambling_tile_buf = temporal_storage_buffer(record_ctx, "rtr.scambling_tile_buf", sizeof(SCRAMBLING_TILE));
            sobol_buf = temporal_storage_buffer(record_ctx, "rtr.sobol_buf", sizeof(SOBOL));

            daxa::TaskGraph temp_task_graph = daxa::TaskGraph({.device = record_ctx.device, .name = "temp_task_graph"});
            temp_task_graph.use_persistent_buffer(ranking_tile_buf);
            temp_task_graph.use_persistent_buffer(scambling_tile_buf);
            temp_task_graph.use_persistent_buffer(sobol_buf);
            temp_task_graph.add_task({
                .attachments = {
                    daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, ranking_tile_buf),
                    daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, scambling_tile_buf),
                    daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, sobol_buf),
                },
                .task = [](daxa::TaskInterface const &ti) {
                    auto staging_buffer = ti.device.create_buffer({
                        .size = sizeof(RANKING_TILE) + sizeof(SCRAMBLING_TILE) + sizeof(SOBOL),
                        .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                        .name = "staging_buffer",
                    });
                    auto *buffer_ptr = ti.device.get_host_address_as<int>(staging_buffer).value();
                    auto offset = size_t{0};

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
                        .src_offset = ranking_tile_offset,
                        .size = RANKING_TILE.size() * sizeof(int),
                    });
                    ti.recorder.copy_buffer_to_buffer({
                        .src_buffer = staging_buffer,
                        .dst_buffer = ti.get(daxa::TaskBufferAttachmentIndex{1}).ids[0],
                        .src_offset = scrambling_tile_offset,
                        .size = SCRAMBLING_TILE.size() * sizeof(int),
                    });
                    ti.recorder.copy_buffer_to_buffer({
                        .src_buffer = staging_buffer,
                        .dst_buffer = ti.get(daxa::TaskBufferAttachmentIndex{2}).ids[0],
                        .src_offset = sobel_offset,
                        .size = SOBOL.size() * sizeof(int),
                    });
                },
                .name = "upload_lookup_tables",
            });
            temp_task_graph.submit({});
            temp_task_graph.complete({});
            temp_task_graph.execute({});
        }

        auto gbuffer_half_res = daxa_u32vec2{(record_ctx.render_resolution.x + 1) / 2, (record_ctx.render_resolution.y + 1) / 2};

        temporal_rng_tex = PingPongImage{};
        auto [rng_output_tex, rng_history_tex] = temporal_rng_tex.get(
            record_ctx.device,
            {
                .format = daxa::Format::R32_UINT,
                .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                .name = "temporal_rng_tex",
            });
        record_ctx.task_graph.use_persistent_image(rng_output_tex);
        record_ctx.task_graph.use_persistent_image(rng_history_tex);

        record_ctx.add(ComputeTask<RtrTraceCompute, RtrTraceComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"kajiya/rtdgi/trace_reflection.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::gpu_input, record_ctx.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::globals, record_ctx.task_globals_buffer}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::ranking_tile_buf, ranking_tile_buf}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::scambling_tile_buf, scambling_tile_buf}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::sobol_buf, sobol_buf}},
                VOXELS_BUFFER_USES_ASSIGN(RtdgiTraceCompute, voxel_buffers),
                IRCACHE_BUFFER_USES_ASSIGN(RtdgiTraceCompute, ircache),
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::gbuffer_tex, gbuffer_depth.gbuffer}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::depth_tex, gbuffer_depth.depth.current()}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::rtdgi_tex, rtdgi_irradiance}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::sky_cube_tex, sky_cube}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::out0_tex, refl0_tex}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::out1_tex, refl1_tex}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::out2_tex, refl2_tex}},
                daxa::TaskViewVariant{std::pair{RtrTraceCompute::rng_out_tex, rng_output_tex}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, RtrTraceComputePush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(RtrTraceCompute::depth_tex).ids[0]).value();
                auto const out_image_info = ti.device.info_image(ti.get(RtrTraceCompute::gbuffer_tex).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                push.gbuffer_tex_size = extent_inv_extent_2d(image_info);
                push.reuse_rtdgi_rays = 1;
                set_push_constant(ti, push);
                ti.recorder.dispatch({(out_image_info.size.x + 7) / 8, (out_image_info.size.y + 7) / 8});
            },
        });

        return {};
    }
};

#endif
