#include "voxel_world.inl"

void VoxelWorld::init_gpu_malloc(GpuContext &gpu_context) {
    if (!gpu_malloc_initialized) {
        gpu_malloc_initialized = true;
        buffers.voxel_malloc.create(gpu_context);
        // buffers.voxel_leaf_chunk_malloc.create(device);
        // buffers.voxel_parent_chunk_malloc.create(device);
    }
}

void VoxelWorld::record_startup(RecordContext &record_ctx) {
    record_ctx.task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_globals.task_resource),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_chunks.task_resource),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_malloc.task_element_buffer),
            // daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_leaf_chunk_malloc.task_element_buffer),
            // daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_parent_chunk_malloc.task_element_buffer),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            ti.recorder.clear_buffer({
                .buffer = buffers.voxel_globals.task_resource.get_state().buffers[0],
                .offset = 0,
                .size = sizeof(VoxelWorldGlobals),
                .clear_value = 0,
            });

            auto chunk_n = (1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);
            chunk_n = chunk_n * chunk_n * chunk_n * CHUNK_LOD_LEVELS;
            ti.recorder.clear_buffer({
                .buffer = buffers.voxel_chunks.task_resource.get_state().buffers[0],
                .offset = 0,
                .size = sizeof(VoxelLeafChunk) * chunk_n,
                .clear_value = 0,
            });

            buffers.voxel_malloc.clear_buffers(ti.recorder);
            // buffers.voxel_leaf_chunk_malloc.clear_buffers(ti.recorder);
            // buffers.voxel_parent_chunk_malloc.clear_buffers(ti.recorder);
        },
        .name = "clear chunk editor",
    });

    record_ctx.task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_malloc.task_allocator_buffer),
            // daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_leaf_chunk_malloc.task_allocator_buffer),
            // daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_parent_chunk_malloc.task_allocator_buffer),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            buffers.voxel_malloc.init(ti.device, ti.recorder);
            // buffers.voxel_leaf_chunk_malloc.init(ti.device, ti.recorder);
            // buffers.voxel_parent_chunk_malloc.init(ti.device, ti.recorder);
        },
        .name = "Initialize",
    });
}

void VoxelWorld::begin_frame(daxa::Device &device, VoxelWorldOutput const &gpu_output) {
    buffers.voxel_malloc.check_for_realloc(device, gpu_output.voxel_malloc_output.current_element_count);
    // buffers.voxel_leaf_chunk_malloc.check_for_realloc(device, gpu_output.voxel_leaf_chunk_output.current_element_count);
    // buffers.voxel_parent_chunk_malloc.check_for_realloc(device, gpu_output.voxel_parent_chunk_output.current_element_count);

    bool needs_realloc = false;
    needs_realloc = needs_realloc || buffers.voxel_malloc.needs_realloc();
    // needs_realloc = needs_realloc || buffers.voxel_leaf_chunk_malloc.needs_realloc();
    // needs_realloc = needs_realloc || buffers.voxel_parent_chunk_malloc.needs_realloc();

    debug_gpu_heap_usage = gpu_output.voxel_malloc_output.current_element_count * VOXEL_MALLOC_PAGE_SIZE_BYTES;
    debug_page_count = buffers.voxel_malloc.current_element_count;

    if (needs_realloc) {
        auto temp_task_graph = daxa::TaskGraph({
            .device = device,
            .name = "temp_task_graph",
        });

        buffers.voxel_malloc.for_each_task_buffer([&temp_task_graph](auto &task_buffer) { temp_task_graph.use_persistent_buffer(task_buffer); });
        // buffers.voxel_leaf_chunk_malloc.for_each_task_buffer([&temp_task_graph](auto &task_buffer) { temp_task_graph.use_persistent_buffer(task_buffer); });
        // buffers.voxel_parent_chunk_malloc.for_each_task_buffer([&temp_task_graph](auto &task_buffer) { temp_task_graph.use_persistent_buffer(task_buffer); });
        temp_task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_READ, buffers.voxel_malloc.task_old_element_buffer),
                daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_malloc.task_element_buffer),
                // daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_READ, buffers.voxel_leaf_chunk_malloc.task_old_element_buffer),
                // daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_leaf_chunk_malloc.task_element_buffer),
                // daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_READ, buffers.voxel_parent_chunk_malloc.task_old_element_buffer),
                // daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_parent_chunk_malloc.task_element_buffer),
            },
            .task = [this](daxa::TaskInterface const &ti) {
                if (buffers.voxel_malloc.needs_realloc()) {
                    buffers.voxel_malloc.realloc(ti.device, ti.recorder);
                }
                // if (buffers.voxel_leaf_chunk_malloc.needs_realloc()) {
                //     buffers.voxel_leaf_chunk_malloc.realloc(ti.device, ti.recorder);
                // }
                // if (buffers.voxel_parent_chunk_malloc.needs_realloc()) {
                //     buffers.voxel_parent_chunk_malloc.realloc(ti.device, ti.recorder);
                // }
            },
            .name = "Transfer Task",
        });

        temp_task_graph.submit({});
        temp_task_graph.complete({});
        temp_task_graph.execute({});
    }
}

void VoxelWorld::use_buffers(RecordContext &record_ctx) {
    buffers.voxel_globals = record_ctx.gpu_context->find_or_add_temporal_buffer({
        .size = static_cast<daxa_u32>(sizeof(VoxelWorldGlobals)),
        .name = "voxel_globals",
    });

    auto chunk_n = (1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);
    chunk_n = chunk_n * chunk_n * chunk_n * CHUNK_LOD_LEVELS;
    buffers.voxel_chunks = record_ctx.gpu_context->find_or_add_temporal_buffer({
        .size = static_cast<daxa_u32>(sizeof(VoxelLeafChunk)) * chunk_n,
        .name = "voxel_chunks",
    });

    init_gpu_malloc(*record_ctx.gpu_context);

    record_ctx.task_graph.use_persistent_buffer(buffers.voxel_globals.task_resource);
    record_ctx.task_graph.use_persistent_buffer(buffers.voxel_chunks.task_resource);
    buffers.voxel_malloc.for_each_task_buffer([&record_ctx](auto &task_buffer) { record_ctx.task_graph.use_persistent_buffer(task_buffer); });
    // buffers.voxel_leaf_chunk_malloc.for_each_task_buffer([&record_ctx](auto &task_buffer) { record_ctx.task_graph.use_persistent_buffer(task_buffer); });
    // buffers.voxel_parent_chunk_malloc.for_each_task_buffer([&record_ctx](auto &task_buffer) { record_ctx.task_graph.use_persistent_buffer(task_buffer); });
}

void VoxelWorld::record_frame(RecordContext &record_ctx, daxa::TaskBufferView task_gvox_model_buffer, daxa::TaskImageView task_value_noise_image) {
    record_ctx.add(ComputeTask<PerChunkCompute, PerChunkComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{PerChunkCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
            daxa::TaskViewVariant{std::pair{PerChunkCompute::gvox_model, task_gvox_model_buffer}},
            daxa::TaskViewVariant{std::pair{PerChunkCompute::globals, record_ctx.gpu_context->task_globals_buffer}},
            daxa::TaskViewVariant{std::pair{PerChunkCompute::voxel_globals, buffers.voxel_globals.task_resource}},
            daxa::TaskViewVariant{std::pair{PerChunkCompute::voxel_chunks, buffers.voxel_chunks.task_resource}},
            daxa::TaskViewVariant{std::pair{PerChunkCompute::value_noise_texture, task_value_noise_image.view({.layer_count = 256})}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, PerChunkComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            auto const dispatch_size = 1 << LOG2_CHUNKS_DISPATCH_SIZE;
            ti.recorder.dispatch({dispatch_size, dispatch_size, dispatch_size * CHUNK_LOD_LEVELS});
        },
    });

    auto task_temp_voxel_chunks_buffer = record_ctx.task_graph.create_transient_buffer({
        .size = sizeof(TempVoxelChunk) * MAX_CHUNK_UPDATES_PER_FRAME,
        .name = "temp_voxel_chunks_buffer",
    });

    record_ctx.add(ComputeTask<ChunkEditCompute, ChunkEditComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::globals, record_ctx.gpu_context->task_globals_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::gvox_model, task_gvox_model_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::voxel_globals, buffers.voxel_globals.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::voxel_chunks, buffers.voxel_chunks.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::voxel_malloc_page_allocator, buffers.voxel_malloc.task_allocator_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::temp_voxel_chunks, task_temp_voxel_chunks_buffer}},
            // daxa::TaskViewVariant{std::pair{ChunkEditCompute::simulated_voxel_particles, task_simulated_voxel_particles_buffer}},
            // daxa::TaskViewVariant{std::pair{ChunkEditCompute::placed_voxel_particles, task_placed_voxel_particles_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::value_noise_texture, task_value_noise_image.view({.layer_count = 256})}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::test_texture, record_ctx.gpu_context->task_test_texture}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::test_texture2, record_ctx.gpu_context->task_test_texture2}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ChunkEditComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.get(ChunkEditCompute::globals).ids[0],
                .offset = offsetof(GpuGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, chunk_edit_dispatch),
            });
        },
    });

    record_ctx.add(ComputeTask<ChunkEditPostProcessCompute, ChunkEditPostProcessComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::globals, record_ctx.gpu_context->task_globals_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::gvox_model, task_gvox_model_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::voxel_globals, buffers.voxel_globals.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::voxel_chunks, buffers.voxel_chunks.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::voxel_malloc_page_allocator, buffers.voxel_malloc.task_allocator_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::temp_voxel_chunks, task_temp_voxel_chunks_buffer}},
            // daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::simulated_voxel_particles, task_simulated_voxel_particles_buffer}},
            // daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::placed_voxel_particles, task_placed_voxel_particles_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::value_noise_texture, task_value_noise_image.view({.layer_count = 256})}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ChunkEditPostProcessComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.get(ChunkEditPostProcessCompute::globals).ids[0],
                .offset = offsetof(GpuGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, chunk_edit_dispatch),
            });
        },
    });

    record_ctx.add(ComputeTask<ChunkOptCompute, ChunkOptComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
        .extra_defines = {{"CHUNK_OPT_STAGE", "0"}},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{ChunkOptCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkOptCompute::globals, record_ctx.gpu_context->task_globals_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkOptCompute::voxel_globals, buffers.voxel_globals.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkOptCompute::temp_voxel_chunks, task_temp_voxel_chunks_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkOptCompute::voxel_chunks, buffers.voxel_chunks.task_resource}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ChunkOptComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.get(ChunkOptCompute::globals).ids[0],
                .offset = offsetof(GpuGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, subchunk_x2x4_dispatch),
            });
        },
    });

    record_ctx.add(ComputeTask<ChunkOptCompute, ChunkOptComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
        .extra_defines = {{"CHUNK_OPT_STAGE", "1"}},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{ChunkOptCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkOptCompute::globals, record_ctx.gpu_context->task_globals_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkOptCompute::voxel_globals, buffers.voxel_globals.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkOptCompute::temp_voxel_chunks, task_temp_voxel_chunks_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkOptCompute::voxel_chunks, buffers.voxel_chunks.task_resource}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ChunkOptComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.get(ChunkOptCompute::globals).ids[0],
                .offset = offsetof(GpuGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, subchunk_x8up_dispatch),
            });
        },
    });

    record_ctx.add(ComputeTask<ChunkAllocCompute, ChunkAllocComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{ChunkAllocCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkAllocCompute::globals, record_ctx.gpu_context->task_globals_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkAllocCompute::voxel_globals, buffers.voxel_globals.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkAllocCompute::temp_voxel_chunks, task_temp_voxel_chunks_buffer}},
            daxa::TaskViewVariant{std::pair{ChunkAllocCompute::voxel_chunks, buffers.voxel_chunks.task_resource}},
            daxa::TaskViewVariant{std::pair{ChunkAllocCompute::voxel_malloc_page_allocator, buffers.voxel_malloc.task_allocator_buffer}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ChunkAllocComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.get(ChunkAllocCompute::globals).ids[0],
                // NOTE: This should always have the same value as the chunk edit dispatch, so we're re-using it here
                .offset = offsetof(GpuGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, chunk_edit_dispatch),
            });
        },
    });
}
