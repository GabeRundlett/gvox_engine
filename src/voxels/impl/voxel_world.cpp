#include <application/settings.inl>
#include "voxel_world.inl"

#include <fmt/format.h>

#if CPU_VOXEL_GEN

void VoxelWorld::startup() {
    voxel_globals.chunk_update_n = 0;
}
void VoxelWorld::per_frame() {
    for (uint32_t i = 0; i < MAX_CHUNK_UPDATES_PER_FRAME; ++i) {
        voxel_globals.chunk_update_infos[i].brush_flags = 0;
        voxel_globals.chunk_update_infos[i].i = {}; // INVALID_CHUNK_I;
    }
    voxel_globals.chunk_update_n = 0;
    buffers.voxel_malloc.per_frame();

    // TODO: Brush stuff...
}

uint32_t calc_chunk_index_from_worldspace(ivec3 chunk_i, uvec3 chunk_n) {
    chunk_i = chunk_i % ivec3(chunk_n) + ivec3(chunk_i.x < 0, chunk_i.y < 0, chunk_i.z < 0) * ivec3(chunk_n);
    uint32_t chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;
    return chunk_index;
}

auto atomicAdd(auto &x, auto &val) {
    auto prev = x;
    x += val;
    return prev;
}

void try_elect(VoxelWorldGlobals &VOXEL_WORLD, VoxelChunkUpdateInfo &work_item, uint32_t &update_index) {
    uint32_t prev_update_n = atomicAdd(VOXEL_WORLD.chunk_update_n, 1);

    // Check if the work item can be added
    if (prev_update_n < MAX_CHUNK_UPDATES_PER_FRAME) {
        // Set the chunk edit dispatch z axis (64/8, 64/8, 64 x 8 x 8 / 8 = 64 x 8) = (8, 8, 512)
        // atomicAdd(INDIRECT.chunk_edit_dispatch.z, CHUNK_SIZE / 8);
        // atomicAdd(INDIRECT.subchunk_x2x4_dispatch.z, 1);
        // atomicAdd(INDIRECT.subchunk_x8up_dispatch.z, 1);
        // Set the chunk update info
        VOXEL_WORLD.chunk_update_infos[prev_update_n] = work_item;
        update_index = prev_update_n + 1;
    }
}

void VoxelWorld::per_chunk(uvec3 gl_GlobalInvocationID) {
    ivec3 chunk_n = ivec3(1 << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);

    VoxelChunkUpdateInfo terrain_work_item;
    terrain_work_item.i = std::bit_cast<daxa_i32vec3>(ivec3(gl_GlobalInvocationID) & (chunk_n - 1));

    ivec3 offset = (std::bit_cast<ivec3>(voxel_globals.offset) >> ivec3(6 + LOG2_VOXEL_SIZE));
    ivec3 prev_offset = (std::bit_cast<ivec3>(voxel_globals.prev_offset) >> ivec3(6 + LOG2_VOXEL_SIZE));

    terrain_work_item.chunk_offset = std::bit_cast<daxa_i32vec3>(offset);
    terrain_work_item.brush_flags = BRUSH_FLAGS_WORLD_BRUSH;

    // (const) number of chunks in each axis
    uint32_t chunk_index = calc_chunk_index_from_worldspace(terrain_work_item.i, chunk_n);

    uint32_t update_index = 0;

    if ((voxel_chunks[chunk_index].flags & CHUNK_FLAGS_ACCEL_GENERATED) == 0) {
        try_elect(terrain_work_item, update_index);
    } else if (offset != prev_offset) {
        // invalidate chunks outside the chunk_offset
        ivec3 diff = clamp(ivec3(offset - prev_offset), -chunk_n, chunk_n);

        ivec3 start;
        ivec3 end;

        start.x = diff.x < 0 ? 0 : chunk_n.x - diff.x;
        end.x = diff.x < 0 ? -diff.x : chunk_n.x;

        start.y = diff.y < 0 ? 0 : chunk_n.y - diff.y;
        end.y = diff.y < 0 ? -diff.y : chunk_n.y;

        start.z = diff.z < 0 ? 0 : chunk_n.z - diff.z;
        end.z = diff.z < 0 ? -diff.z : chunk_n.z;

        uvec3 temp_chunk_i = uvec3((std::bit_cast<ivec3>(terrain_work_item.i) - offset) % ivec3(chunk_n));

        if ((temp_chunk_i.x >= start.x && temp_chunk_i.x < end.x) ||
            (temp_chunk_i.y >= start.y && temp_chunk_i.y < end.y) ||
            (temp_chunk_i.z >= start.z && temp_chunk_i.z < end.z)) {
            voxel_chunks[chunk_index].flags &= ~CHUNK_FLAGS_ACCEL_GENERATED;
            try_elect(voxel_globals, terrain_work_item, update_index);
        }
    } else {
        // Wrapped chunk index in leaf chunk space (0^3 - 31^3)
        ivec3 wrapped_chunk_i = imod3(terrain_work_item.i - imod3(std::bit_cast<ivec3>(terrain_work_item.chunk_offset) - ivec3(chunk_n), ivec3(chunk_n)), ivec3(chunk_n));
        // Leaf chunk position in world space
        ivec3 world_chunk = std::bit_cast<ivec3>(terrain_work_item.chunk_offset) + wrapped_chunk_i - ivec3(chunk_n / 2);

        terrain_work_item.brush_input = voxel_globals.brush_input;

        ivec3 brush_chunk = (ivec3(floor(voxel_globals.brush_input.pos)) + voxel_globals.brush_input.pos_offset) >> (6 + LOG2_VOXEL_SIZE);
        bool is_near_brush = all(greaterThanEqual(world_chunk, brush_chunk - 1)) && all(lessThanEqual(world_chunk, brush_chunk + 1));

        if (is_near_brush && deref(gpu_input).actions[GAME_ACTION_BRUSH_A] != 0) {
            terrain_work_item.brush_flags = BRUSH_FLAGS_USER_BRUSH_A;
            try_elect(voxel_globals, terrain_work_item, update_index);
        } else if (is_near_brush && deref(gpu_input).actions[GAME_ACTION_BRUSH_B] != 0) {
            terrain_work_item.brush_flags = BRUSH_FLAGS_USER_BRUSH_B;
            try_elect(voxel_globals, terrain_work_item, update_index);
        }
    }

    CHUNKS(chunk_index).update_index = update_index;
}
void VoxelWorld::edit() {
}
void VoxelWorld::edit_post_process() {
}
void VoxelWorld::opt() {
}
void VoxelWorld::alloc() {
}
#endif

void VoxelWorld::record_startup(RecordContext &record_ctx) {
#if CPU_VOXEL_GEN
    startup();
#else
    use_buffers(record_ctx);

    record_ctx.task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_globals.task_resource),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_chunks.task_resource),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_malloc.task_element_buffer),
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
        },
        .name = "clear chunk editor",
    });

    record_ctx.task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_malloc.task_allocator_buffer),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            buffers.voxel_malloc.init(ti.device, ti.recorder);
        },
        .name = "Initialize",
    });

    record_ctx.add(ComputeTask<VoxelWorldStartupCompute, VoxelWorldStartupComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/startup.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{VoxelWorldStartupCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
            daxa::TaskViewVariant{std::pair{VoxelWorldStartupCompute::globals, record_ctx.gpu_context->task_globals_buffer}},
            VOXELS_BUFFER_USES_ASSIGN(VoxelWorldStartupCompute, buffers),
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, VoxelWorldStartupComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({1, 1, 1});
        },
    });
#endif
}

void VoxelWorld::begin_frame(daxa::Device &device, VoxelWorldOutput const &gpu_output) {
#if CPU_VOXEL_GEN
    per_frame();
    for (uint32_t zi = 0; zi < (1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS); ++zi) {
        for (uint32_t yi = 0; yi < (1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS); ++yi) {
            for (uint32_t xi = 0; xi < (1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS); ++xi) {
                per_chunk(uvec3(xi, yi, zi));
            }
        }
    }
#else
    buffers.voxel_malloc.check_for_realloc(device, gpu_output.voxel_malloc_output.current_element_count);

    bool needs_realloc = false;
    needs_realloc = needs_realloc || buffers.voxel_malloc.needs_realloc();

    debug_gpu_heap_usage = gpu_output.voxel_malloc_output.current_element_count * VOXEL_MALLOC_PAGE_SIZE_BYTES;
    debug_page_count = buffers.voxel_malloc.current_element_count;
    debug_utils::DebugDisplay::set_debug_string("VoxelWorld: Page count", fmt::format("{} pages ({:.2f} MB)", debug_page_count, static_cast<double>(debug_page_count) * VOXEL_MALLOC_PAGE_SIZE_BYTES / 1'000'000.0));
    debug_utils::DebugDisplay::set_debug_string("VoxelWorld: GPU heap usage", fmt::format("{:.2f} MB", static_cast<double>(debug_gpu_heap_usage) / 1'000'000));

    if (needs_realloc) {
        auto temp_task_graph = daxa::TaskGraph({
            .device = device,
            .name = "temp_task_graph",
        });

        buffers.voxel_malloc.for_each_task_buffer([&temp_task_graph](auto &task_buffer) { temp_task_graph.use_persistent_buffer(task_buffer); });
        temp_task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_READ, buffers.voxel_malloc.task_old_element_buffer),
                daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_malloc.task_element_buffer),
            },
            .task = [this](daxa::TaskInterface const &ti) {
                if (buffers.voxel_malloc.needs_realloc()) {
                    buffers.voxel_malloc.realloc(ti.device, ti.recorder);
                }
            },
            .name = "Transfer Task",
        });

        temp_task_graph.submit({});
        temp_task_graph.complete({});
        temp_task_graph.execute({});
    }
#endif
}

void VoxelWorld::use_buffers(RecordContext &record_ctx) {
    buffers.voxel_globals = record_ctx.gpu_context->find_or_add_temporal_buffer({
        .size = sizeof(VoxelWorldGlobals),
        .name = "voxel_globals",
    });

    auto chunk_n = (1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);
    chunk_n = chunk_n * chunk_n * chunk_n * CHUNK_LOD_LEVELS;
    buffers.voxel_chunks = record_ctx.gpu_context->find_or_add_temporal_buffer({
        .size = sizeof(VoxelLeafChunk) * chunk_n,
        .name = "voxel_chunks",
    });

    if (!gpu_malloc_initialized) {
        gpu_malloc_initialized = true;
        buffers.voxel_malloc.create(*record_ctx.gpu_context);
    }

    record_ctx.task_graph.use_persistent_buffer(buffers.voxel_globals.task_resource);
    record_ctx.task_graph.use_persistent_buffer(buffers.voxel_chunks.task_resource);
    buffers.voxel_malloc.for_each_task_buffer([&record_ctx](auto &task_buffer) { record_ctx.task_graph.use_persistent_buffer(task_buffer); });

#if CPU_VOXEL_GEN
    temp_voxel_chunks.resize(MAX_CHUNK_UPDATES_PER_FRAME);
    voxel_chunks.resize(chunk_n);
#endif
}

void VoxelWorld::record_frame(RecordContext &record_ctx, daxa::TaskBufferView task_gvox_model_buffer, daxa::TaskImageView task_value_noise_image) {
    use_buffers(record_ctx);

#if CPU_VOXEL_GEN
    // upload data
#else
    record_ctx.add(ComputeTask<VoxelWorldPerframeCompute, VoxelWorldPerframeComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"voxels/impl/perframe.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{VoxelWorldPerframeCompute::gpu_input, record_ctx.gpu_context->task_input_buffer}},
            daxa::TaskViewVariant{std::pair{VoxelWorldPerframeCompute::gpu_output, record_ctx.gpu_context->task_output_buffer}},
            daxa::TaskViewVariant{std::pair{VoxelWorldPerframeCompute::globals, record_ctx.gpu_context->task_globals_buffer}},
            VOXELS_BUFFER_USES_ASSIGN(VoxelWorldPerframeCompute, buffers),
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, VoxelWorldPerframeComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({1, 1, 1});
        },
    });

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
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::value_noise_texture, task_value_noise_image.view({.layer_count = 256})}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::test_texture, record_ctx.gpu_context->task_test_texture}},
            daxa::TaskViewVariant{std::pair{ChunkEditCompute::test_texture2, record_ctx.gpu_context->task_test_texture2}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ChunkEditComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.get(ChunkEditCompute::voxel_globals).ids[0],
                .offset = offsetof(VoxelWorldGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, chunk_edit_dispatch),
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
            daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::value_noise_texture, task_value_noise_image.view({.layer_count = 256})}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ChunkEditPostProcessComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.get(ChunkEditPostProcessCompute::voxel_globals).ids[0],
                .offset = offsetof(VoxelWorldGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, chunk_edit_dispatch),
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
                .indirect_buffer = ti.get(ChunkOptCompute::voxel_globals).ids[0],
                .offset = offsetof(VoxelWorldGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, subchunk_x2x4_dispatch),
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
                .indirect_buffer = ti.get(ChunkOptCompute::voxel_globals).ids[0],
                .offset = offsetof(VoxelWorldGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, subchunk_x8up_dispatch),
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
                .indirect_buffer = ti.get(ChunkAllocCompute::voxel_globals).ids[0],
                // NOTE: This should always have the same value as the chunk edit dispatch, so we're re-using it here
                .offset = offsetof(VoxelWorldGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, chunk_edit_dispatch),
            });
        },
    });
#endif
}
