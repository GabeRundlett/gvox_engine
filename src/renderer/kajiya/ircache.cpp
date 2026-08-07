#include "ircache.inl"
#include <fmt/format.h>

#include <application/settings.hpp>

auto IrcacheRenderState::trace_irradiance(GpuContext &gpu_context, VoxelWorldBuffers &voxel_buffers, daxa::TaskImageView sky_cube, daxa::TaskImageView transmittance_lut) -> IrcacheIrradiancePendingSummation {
    auto indirect_args_buf = gpu_context.frame_task_graph.create_transient_buffer({
        .size = sizeof(uint32_t) * 4 * 4,
        .name = "ircache.trace_indirect_args_buf",
    });

    gpu_context.add(ComputeTask<IrcachePrepareTraceDispatchCompute::Info, IrcachePrepareTraceDispatchComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"kajiya/ircache/prepare_trace_dispatch_args.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{IrcachePrepareTraceDispatchCompute::AT.ircache_meta_buf, this->ircache_meta_buf}},
            daxa::TaskViewVariant{std::pair{IrcachePrepareTraceDispatchCompute::AT.dispatch_args, indirect_args_buf}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, IrcachePrepareTraceDispatchComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({1, 1, 1});
        },
    });

    gpu_context.add(ComputeTask<IrcacheResetCompute::Info, IrcacheResetComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"kajiya/ircache/reset_entry.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{IrcacheResetCompute::AT.gpu_input, gpu_context.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{IrcacheResetCompute::AT.ircache_life_buf, this->ircache_life_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheResetCompute::AT.ircache_meta_buf, this->ircache_meta_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheResetCompute::AT.ircache_irradiance_buf, this->ircache_irradiance_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheResetCompute::AT.ircache_aux_buf, this->ircache_aux_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheResetCompute::AT.ircache_entry_indirection_buf, this->ircache_entry_indirection_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheResetCompute::AT.dispatch_args, indirect_args_buf}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, IrcacheResetComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.get(IrcacheResetCompute::AT.dispatch_args).ids[0],
                .offset = sizeof(daxa_u32vec4) * 2,
            });
        },
    });

    gpu_context.add(RayTracingTask<IrcacheTraceAccessRt::Info, IrcacheTraceAccessRtPush, NoTaskInfo>{
        .source = daxa::ShaderFile{"kajiya/ircache/trace_accessibility.rt.glsl"},
        .views = std::array{
            // daxa::TaskViewVariant{std::pair{IrcacheTraceAccessRt::AT.geometry_pointers, voxel_buffers.blas_geom_pointers.task_resource}},
            // daxa::TaskViewVariant{std::pair{IrcacheTraceAccessRt::AT.attribute_pointers, voxel_buffers.blas_attr_pointers.task_resource}},
            // daxa::TaskViewVariant{std::pair{IrcacheTraceAccessRt::AT.blas_transforms, voxel_buffers.blas_transforms.task_resource}},
            daxa::TaskViewVariant{std::pair{IrcacheTraceAccessRt::AT.chunk_primitive_pointers, voxel_buffers.brick_primitive_pointers.task_resource}},
            daxa::TaskViewVariant{std::pair{IrcacheTraceAccessRt::AT.tlas, voxel_buffers.task_tlas}},
            daxa::TaskViewVariant{std::pair{IrcacheTraceAccessRt::AT.ircache_spatial_buf, this->ircache_spatial_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheTraceAccessRt::AT.ircache_life_buf, this->ircache_life_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheTraceAccessRt::AT.ircache_reposition_proposal_buf, this->ircache_reposition_proposal_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheTraceAccessRt::AT.ircache_meta_buf, this->ircache_meta_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheTraceAccessRt::AT.ircache_aux_buf, this->ircache_aux_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheTraceAccessRt::AT.ircache_entry_indirection_buf, this->ircache_entry_indirection_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheTraceAccessRt::AT.dispatch_args, indirect_args_buf}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::RayTracingPipeline &pipeline, IrcacheTraceAccessRtPush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.trace_rays_indirect({
                .indirect_device_address = ti.device.device_address(ti.get(IrcacheTraceAccessRt::AT.dispatch_args).ids[0]).value() + sizeof(daxa_u32vec4) * 1,
            });
        },
    });

    gpu_context.add(RayTracingTask<IrcacheValidateRt::Info, IrcacheValidateRtPush, NoTaskInfo>{
        .source = daxa::ShaderFile{"kajiya/ircache/ircache_validate.rt.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{IrcacheValidateRt::AT.gpu_input, gpu_context.task_input_buffer}},
            // daxa::TaskViewVariant{std::pair{IrcacheValidateRt::AT.geometry_pointers, voxel_buffers.blas_geom_pointers.task_resource}},
            // daxa::TaskViewVariant{std::pair{IrcacheValidateRt::AT.attribute_pointers, voxel_buffers.blas_attr_pointers.task_resource}},
            // daxa::TaskViewVariant{std::pair{IrcacheValidateRt::AT.blas_transforms, voxel_buffers.blas_transforms.task_resource}},
            daxa::TaskViewVariant{std::pair{IrcacheValidateRt::AT.chunk_primitive_pointers, voxel_buffers.brick_primitive_pointers.task_resource}},
            daxa::TaskViewVariant{std::pair{IrcacheValidateRt::AT.tlas, voxel_buffers.task_tlas}},
            daxa::TaskViewVariant{std::pair{IrcacheValidateRt::AT.ircache_spatial_buf, this->ircache_spatial_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheValidateRt::AT.sky_cube_tex, sky_cube}},
            daxa::TaskViewVariant{std::pair{IrcacheValidateRt::AT.transmittance_lut, transmittance_lut}},
            daxa::TaskViewVariant{std::pair{IrcacheValidateRt::AT.ircache_grid_meta_buf, this->ircache_grid_meta_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheValidateRt::AT.ircache_life_buf, this->ircache_life_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheValidateRt::AT.ircache_reposition_proposal_buf, this->ircache_reposition_proposal_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheValidateRt::AT.ircache_reposition_proposal_count_buf, this->ircache_reposition_proposal_count_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheValidateRt::AT.ircache_meta_buf, this->ircache_meta_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheValidateRt::AT.ircache_aux_buf, this->ircache_aux_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheValidateRt::AT.ircache_pool_buf, this->ircache_pool_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheValidateRt::AT.ircache_entry_indirection_buf, this->ircache_entry_indirection_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheValidateRt::AT.ircache_entry_cell_buf, this->ircache_entry_cell_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheValidateRt::AT.dispatch_args, indirect_args_buf}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::RayTracingPipeline &pipeline, IrcacheValidateRtPush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.trace_rays_indirect({
                .indirect_device_address = ti.device.device_address(ti.get(IrcacheValidateRt::AT.dispatch_args).ids[0]).value() + sizeof(daxa_u32vec4) * 3,
            });
        },
    });

    gpu_context.add(RayTracingTask<TraceIrradianceRt::Info, TraceIrradianceRtPush, NoTaskInfo>{
        .source = daxa::ShaderFile{"kajiya/ircache/trace_irradiance.rt.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{TraceIrradianceRt::AT.gpu_input, gpu_context.task_input_buffer}},
            // daxa::TaskViewVariant{std::pair{TraceIrradianceRt::AT.geometry_pointers, voxel_buffers.blas_geom_pointers.task_resource}},
            // daxa::TaskViewVariant{std::pair{TraceIrradianceRt::AT.attribute_pointers, voxel_buffers.blas_attr_pointers.task_resource}},
            // daxa::TaskViewVariant{std::pair{TraceIrradianceRt::AT.blas_transforms, voxel_buffers.blas_transforms.task_resource}},
            daxa::TaskViewVariant{std::pair{TraceIrradianceRt::AT.chunk_primitive_pointers, voxel_buffers.brick_primitive_pointers.task_resource}},
            daxa::TaskViewVariant{std::pair{TraceIrradianceRt::AT.tlas, voxel_buffers.task_tlas}},
            daxa::TaskViewVariant{std::pair{TraceIrradianceRt::AT.ircache_spatial_buf, this->ircache_spatial_buf}},
            daxa::TaskViewVariant{std::pair{TraceIrradianceRt::AT.sky_cube_tex, sky_cube}},
            daxa::TaskViewVariant{std::pair{TraceIrradianceRt::AT.transmittance_lut, transmittance_lut}},
            daxa::TaskViewVariant{std::pair{TraceIrradianceRt::AT.ircache_grid_meta_buf, this->ircache_grid_meta_buf}},
            daxa::TaskViewVariant{std::pair{TraceIrradianceRt::AT.ircache_life_buf, this->ircache_life_buf}},
            daxa::TaskViewVariant{std::pair{TraceIrradianceRt::AT.ircache_reposition_proposal_buf, this->ircache_reposition_proposal_buf}},
            daxa::TaskViewVariant{std::pair{TraceIrradianceRt::AT.ircache_reposition_proposal_count_buf, this->ircache_reposition_proposal_count_buf}},
            daxa::TaskViewVariant{std::pair{TraceIrradianceRt::AT.ircache_meta_buf, this->ircache_meta_buf}},
            daxa::TaskViewVariant{std::pair{TraceIrradianceRt::AT.ircache_aux_buf, this->ircache_aux_buf}},
            daxa::TaskViewVariant{std::pair{TraceIrradianceRt::AT.ircache_pool_buf, this->ircache_pool_buf}},
            daxa::TaskViewVariant{std::pair{TraceIrradianceRt::AT.ircache_entry_indirection_buf, this->ircache_entry_indirection_buf}},
            daxa::TaskViewVariant{std::pair{TraceIrradianceRt::AT.ircache_entry_cell_buf, this->ircache_entry_cell_buf}},
            daxa::TaskViewVariant{std::pair{TraceIrradianceRt::AT.dispatch_args, indirect_args_buf}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::RayTracingPipeline &pipeline, TraceIrradianceRtPush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.trace_rays_indirect({
                .indirect_device_address = ti.device.device_address(ti.get(TraceIrradianceRt::AT.dispatch_args).ids[0]).value() + sizeof(daxa_u32vec4) * 0,
            });
        },
    });

    return {indirect_args_buf};
}

void IrcacheRenderState::sum_up_irradiance_for_sampling(GpuContext &gpu_context, IrcacheIrradiancePendingSummation pending) {
    gpu_context.add(ComputeTask<SumUpIrradianceCompute::Info, SumUpIrradianceComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"kajiya/ircache/sum_up_irradiance.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{SumUpIrradianceCompute::AT.gpu_input, gpu_context.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{SumUpIrradianceCompute::AT.ircache_life_buf, this->ircache_life_buf}},
            daxa::TaskViewVariant{std::pair{SumUpIrradianceCompute::AT.ircache_meta_buf, this->ircache_meta_buf}},
            daxa::TaskViewVariant{std::pair{SumUpIrradianceCompute::AT.ircache_irradiance_buf, this->ircache_irradiance_buf}},
            daxa::TaskViewVariant{std::pair{SumUpIrradianceCompute::AT.ircache_aux_buf, this->ircache_aux_buf}},
            daxa::TaskViewVariant{std::pair{SumUpIrradianceCompute::AT.ircache_entry_indirection_buf, this->ircache_entry_indirection_buf}},
            daxa::TaskViewVariant{std::pair{SumUpIrradianceCompute::AT.dispatch_args, pending.indirect_args_buf}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SumUpIrradianceComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.get(SumUpIrradianceCompute::AT.dispatch_args).ids[0],
                .offset = sizeof(daxa_u32vec4) * 2,
            });
        },
    });

    this->pending_irradiance_sum = false;
}

inline auto temporal_storage_buffer(GpuContext &gpu_context, std::string_view name, size_t size) -> daxa::TaskBuffer {
    auto result = gpu_context.find_or_add_temporal_buffer({
        .size = size,
        .name = name,
    });

    gpu_context.frame_task_graph.register_buffer(result.task_resource);

    return result.task_resource;
}

void IrcacheRenderer::update_eye_position(GpuInput &gpu_input) {
    if (!this->enable_scroll) {
        return;
    }

    gpu_input.ircache_grid_center = daxa_f32vec3{
        gpu_input.player.pos.x + gpu_input.player.player_unit_offset.x,
        gpu_input.player.pos.y + gpu_input.player.player_unit_offset.y,
        gpu_input.player.pos.z + gpu_input.player.player_unit_offset.z,
    };

    this->grid_center = glm::vec3(gpu_input.ircache_grid_center.x, gpu_input.ircache_grid_center.y, gpu_input.ircache_grid_center.z);

    for (size_t cascade = 0; cascade < IRCACHE_CASCADE_COUNT; ++cascade) {
        auto cell_diameter = IRCACHE_GRID_CELL_DIAMETER * static_cast<float>(1 << cascade);
        auto cascade_center = glm::ivec3(glm::floor(this->grid_center / cell_diameter));
        auto cascade_origin = cascade_center - glm::ivec3(IRCACHE_CASCADE_SIZE / 2);

        this->prev_scroll[cascade] = this->cur_scroll[cascade];
        this->cur_scroll[cascade] = cascade_origin;

        gpu_input.ircache_cascades[cascade].origin = {
            this->cur_scroll[cascade].x,
            this->cur_scroll[cascade].y,
            this->cur_scroll[cascade].z,
            0,
        };
        gpu_input.ircache_cascades[cascade].voxels_scrolled_this_frame = {
            this->cur_scroll[cascade].x - this->prev_scroll[cascade].x,
            this->cur_scroll[cascade].y - this->prev_scroll[cascade].y,
            this->cur_scroll[cascade].z - this->prev_scroll[cascade].z,
            0,
        };
    }
}

void IrcacheRenderer::next_frame() {
    ping_pong_ircache_grid_meta_buf.swap();
    this->parity = (this->parity + 1) % 2;
}

auto IrcacheRenderer::prepare(GpuContext &gpu_context) -> IrcacheRenderState {
    constexpr auto INDIRECTION_BUF_ELEM_COUNT = size_t{1024 * 1024};

    auto [ircache_grid_meta_buf_, ircache_grid_meta_buf2_] = ping_pong_ircache_grid_meta_buf.get(
        gpu_context,
        daxa::BufferInfo{
            .size = sizeof(IrcacheCell) * MAX_GRID_CELLS,
            .name = "ircache.grid_meta_buf",
        });
    gpu_context.frame_task_graph.register_buffer(ircache_grid_meta_buf_);
    gpu_context.frame_task_graph.register_buffer(ircache_grid_meta_buf2_);

    auto state = IrcacheRenderState{
        // 0: hash grid cell count
        // 1: entry count
        .ircache_meta_buf = temporal_storage_buffer(gpu_context, "ircache.meta_buf", sizeof(IrcacheMetadata)),
        .ircache_grid_meta_buf = ircache_grid_meta_buf_,
        .ircache_grid_meta_buf2 = ircache_grid_meta_buf2_,
        .ircache_entry_cell_buf = temporal_storage_buffer(
            gpu_context,
            "ircache.entry_cell_buf",
            sizeof(daxa_u32) * MAX_ENTRIES),
        .ircache_spatial_buf = temporal_storage_buffer(
            gpu_context,
            "ircache.spatial_buf",
            sizeof(daxa_f32vec4) * MAX_ENTRIES),
        .ircache_irradiance_buf = temporal_storage_buffer(
            gpu_context,
            "ircache.irradiance_buf",
            3 * sizeof(daxa_f32vec4) * MAX_ENTRIES),
        .ircache_aux_buf = temporal_storage_buffer(
            gpu_context,
            "ircache.aux_buf",
            sizeof(IrcacheAux) * MAX_ENTRIES),
        .ircache_life_buf = temporal_storage_buffer(
            gpu_context,
            "ircache.life_buf",
            sizeof(daxa_u32) * MAX_ENTRIES),
        .ircache_pool_buf = temporal_storage_buffer(
            gpu_context,
            "ircache.pool_buf",
            sizeof(daxa_u32) * MAX_ENTRIES),
        .ircache_entry_indirection_buf = temporal_storage_buffer(
            gpu_context,
            "ircache.entry_indirection_buf",
            sizeof(daxa_u32) * INDIRECTION_BUF_ELEM_COUNT),
        .ircache_reposition_proposal_buf = temporal_storage_buffer(
            gpu_context,
            "ircache.reposition_proposal_buf",
            sizeof(daxa_f32vec4) * MAX_ENTRIES),
        .ircache_reposition_proposal_count_buf = temporal_storage_buffer(
            gpu_context,
            "ircache.reposition_proposal_count_buf",
            sizeof(daxa_u32) * MAX_ENTRIES),
        .pending_irradiance_sum = false,
    };

    if (!this->initialized) {
        auto temp_task_graph = daxa::TaskGraph({
            .device = gpu_context.device,
            .name = "temp_task_graph",
        });

        temp_task_graph.register_buffer(state.ircache_pool_buf);
        temp_task_graph.register_buffer(state.ircache_life_buf);

        gpu_context.add(ComputeTask<ClearIrcachePoolCompute::Info, ClearIrcachePoolComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"kajiya/ircache/clear_ircache_pool.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{ClearIrcachePoolCompute::AT.ircache_pool_buf, state.ircache_pool_buf}},
                daxa::TaskViewVariant{std::pair{ClearIrcachePoolCompute::AT.ircache_life_buf, state.ircache_life_buf}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ClearIrcachePoolComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(MAX_ENTRIES + 63) / 64});
            },
            .task_graph_ptr = &temp_task_graph,
        });

        temp_task_graph.submit({});
        temp_task_graph.complete({});
        temp_task_graph.execute({});

        this->initialized = true;
    }

    gpu_context.add(ComputeTask<IrcacheScrollCascadesCompute::Info, IrcacheScrollCascadesComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"kajiya/ircache/scroll_cascades.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{IrcacheScrollCascadesCompute::AT.gpu_input, gpu_context.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{IrcacheScrollCascadesCompute::AT.ircache_grid_meta_buf, state.ircache_grid_meta_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheScrollCascadesCompute::AT.ircache_grid_meta_buf2, state.ircache_grid_meta_buf2}},
            daxa::TaskViewVariant{std::pair{IrcacheScrollCascadesCompute::AT.ircache_entry_cell_buf, state.ircache_entry_cell_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheScrollCascadesCompute::AT.ircache_irradiance_buf, state.ircache_irradiance_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheScrollCascadesCompute::AT.ircache_life_buf, state.ircache_life_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheScrollCascadesCompute::AT.ircache_pool_buf, state.ircache_pool_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheScrollCascadesCompute::AT.ircache_meta_buf, state.ircache_meta_buf}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, IrcacheScrollCascadesComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({(IRCACHE_CASCADE_SIZE + 31) / 32, IRCACHE_CASCADE_SIZE, IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_COUNT});
        },
    });

    std::swap(state.ircache_grid_meta_buf, state.ircache_grid_meta_buf2);

    auto indirect_args_buf = gpu_context.frame_task_graph.create_transient_buffer({
        .size = sizeof(uint32_t) * 4 * 2,
        .name = "ircache.age_indirect_args_buf",
    });

    gpu_context.add(ComputeTask<IrcachePrepareAgeDispatchCompute::Info, IrcachePrepareAgeDispatchComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"kajiya/ircache/prepare_age_dispatch_args.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{IrcachePrepareAgeDispatchCompute::AT.ircache_meta_buf, state.ircache_meta_buf}},
            daxa::TaskViewVariant{std::pair{IrcachePrepareAgeDispatchCompute::AT.dispatch_args, indirect_args_buf}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, IrcachePrepareAgeDispatchComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({1, 1, 1});
        },
    });

    auto entry_occupancy_buf = gpu_context.frame_task_graph.create_transient_buffer({
        .size = sizeof(uint32_t) * MAX_ENTRIES,
        .name = "ircache.entry_occupancy_buf",
    });
    gpu_context.add(ComputeTask<AgeIrcacheEntriesCompute::Info, AgeIrcacheEntriesComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"kajiya/ircache/age_ircache_entries.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::AT.ircache_meta_buf, state.ircache_meta_buf}},
            daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::AT.ircache_grid_meta_buf, state.ircache_grid_meta_buf}},
            daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::AT.ircache_entry_cell_buf, state.ircache_entry_cell_buf}},
            daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::AT.ircache_life_buf, state.ircache_life_buf}},
            daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::AT.ircache_pool_buf, state.ircache_pool_buf}},
            daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::AT.ircache_spatial_buf, state.ircache_spatial_buf}},
            daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::AT.ircache_reposition_proposal_buf, state.ircache_reposition_proposal_buf}},
            daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::AT.ircache_reposition_proposal_count_buf, state.ircache_reposition_proposal_count_buf}},
            daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::AT.ircache_irradiance_buf, state.ircache_irradiance_buf}},
            daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::AT.entry_occupancy_buf, entry_occupancy_buf}},
            daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::AT.dispatch_args, indirect_args_buf}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, AgeIrcacheEntriesComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.get(AgeIrcacheEntriesCompute::AT.dispatch_args).ids[0],
                .offset = 0,
            });
        },
    });

    inclusive_prefix_scan_u32_1m(gpu_context, entry_occupancy_buf);

    gpu_context.add(ComputeTask<IrcacheCompactEntriesCompute::Info, IrcacheCompactEntriesComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"kajiya/ircache/ircache_compact_entries.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{IrcacheCompactEntriesCompute::AT.ircache_meta_buf, state.ircache_meta_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheCompactEntriesCompute::AT.ircache_life_buf, state.ircache_life_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheCompactEntriesCompute::AT.entry_occupancy_buf, entry_occupancy_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheCompactEntriesCompute::AT.ircache_entry_indirection_buf, state.ircache_entry_indirection_buf}},
            daxa::TaskViewVariant{std::pair{IrcacheCompactEntriesCompute::AT.dispatch_args, indirect_args_buf}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, IrcacheCompactEntriesComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.get(IrcacheCompactEntriesCompute::AT.dispatch_args).ids[0],
                .offset = 0,
            });
        },
    });

    state.ircache_buffers = gpu_context.frame_task_graph.create_transient_buffer({
        .size = sizeof(IrcacheBuffers),
        .name = "ircache.buffers",
    });
    gpu_context.frame_task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, state.ircache_buffers),
            daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_meta_buf),
            daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_grid_meta_buf),
            daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_entry_cell_buf),
            daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_spatial_buf),
            daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_irradiance_buf),
            daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_aux_buf),
            daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_life_buf),
            daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_pool_buf),
            daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_entry_indirection_buf),
            daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_reposition_proposal_buf),
            daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_reposition_proposal_count_buf),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            auto staging_buffer = ti.device.create_buffer({
                .size = sizeof(IrcacheBuffers),
                .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = "staging_buffer",
            });
            ti.recorder.destroy_buffer_deferred(staging_buffer);
            auto *buffer_ptr = ti.device.buffer_host_address_as<IrcacheBuffers>(staging_buffer).value();
            *buffer_ptr = {
                .ircache_meta_buf = ti.device.device_address(ti.get(daxa::TaskBufferAttachmentIndex{1}).ids[0]).value(),
                .ircache_grid_meta_buf = ti.device.device_address(ti.get(daxa::TaskBufferAttachmentIndex{2}).ids[0]).value(),
                .ircache_entry_cell_buf = ti.device.device_address(ti.get(daxa::TaskBufferAttachmentIndex{3}).ids[0]).value(),
                .ircache_spatial_buf = ti.device.device_address(ti.get(daxa::TaskBufferAttachmentIndex{4}).ids[0]).value(),
                .ircache_irradiance_buf = ti.device.device_address(ti.get(daxa::TaskBufferAttachmentIndex{5}).ids[0]).value(),
                .ircache_aux_buf = ti.device.device_address(ti.get(daxa::TaskBufferAttachmentIndex{6}).ids[0]).value(),
                .ircache_life_buf = ti.device.device_address(ti.get(daxa::TaskBufferAttachmentIndex{7}).ids[0]).value(),
                .ircache_pool_buf = ti.device.device_address(ti.get(daxa::TaskBufferAttachmentIndex{8}).ids[0]).value(),
                .ircache_entry_indirection_buf = ti.device.device_address(ti.get(daxa::TaskBufferAttachmentIndex{9}).ids[0]).value(),
                .ircache_reposition_proposal_buf = ti.device.device_address(ti.get(daxa::TaskBufferAttachmentIndex{10}).ids[0]).value(),
                .ircache_reposition_proposal_count_buf = ti.device.device_address(ti.get(daxa::TaskBufferAttachmentIndex{11}).ids[0]).value(),
            };
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer = staging_buffer,
                .dst_buffer = ti.get(daxa::TaskBufferAttachmentIndex{0}).ids[0],
                .size = sizeof(IrcacheBuffers),
            });
        },
        .name = "UploadIrcacheBuffers",
    });

    return state;
}
