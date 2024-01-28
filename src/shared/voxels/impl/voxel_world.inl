#pragma once

#include <shared/voxels/impl/voxels.inl>
#include <shared/globals.inl>

#if PerChunkComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(PerChunkCompute, 6)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGvoxModel), gvox_model)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_DECL_TASK_HEAD_END
struct PerChunkComputePush {
    DAXA_TH_BLOB(PerChunkCompute, uses)
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(PerChunkComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(GpuGvoxModel) gvox_model = push.uses.gvox_model;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_RWBufferPtr(VoxelWorldGlobals) voxel_globals = push.uses.voxel_globals;
daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunks = push.uses.voxel_chunks;
daxa_ImageViewId value_noise_texture = push.uses.value_noise_texture;
#endif
#endif

#if ChunkEditComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(ChunkEditCompute, 8)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGvoxModel), gvox_model)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelMallocPageAllocator), voxel_malloc_page_allocator)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(TempVoxelChunk), temp_voxel_chunks)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D_ARRAY, value_noise_texture)
// DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(SimulatedVoxelParticle), simulated_voxel_particles)
// DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), placed_voxel_particles)
DAXA_DECL_TASK_HEAD_END
struct ChunkEditComputePush {
    DAXA_TH_BLOB(ChunkEditCompute, uses)
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(ChunkEditComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_BufferPtr(GpuGvoxModel) gvox_model = push.uses.gvox_model;
daxa_BufferPtr(VoxelWorldGlobals) voxel_globals = push.uses.voxel_globals;
daxa_BufferPtr(VoxelLeafChunk) voxel_chunks = push.uses.voxel_chunks;
daxa_BufferPtr(VoxelMallocPageAllocator) voxel_malloc_page_allocator = push.uses.voxel_malloc_page_allocator;
daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunks = push.uses.temp_voxel_chunks;
daxa_ImageViewId value_noise_texture = push.uses.value_noise_texture;
#endif
#endif

#if ChunkEditPostProcessComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(ChunkEditPostProcessCompute, 8)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGvoxModel), gvox_model)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelMallocPageAllocator), voxel_malloc_page_allocator)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(TempVoxelChunk), temp_voxel_chunks)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D_ARRAY, value_noise_texture)
// DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(SimulatedVoxelParticle), simulated_voxel_particles)
// DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), placed_voxel_particles)
DAXA_DECL_TASK_HEAD_END
struct ChunkEditPostProcessComputePush {
    DAXA_TH_BLOB(ChunkEditPostProcessCompute, uses)
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(ChunkEditPostProcessComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_BufferPtr(GpuGvoxModel) gvox_model = push.uses.gvox_model;
daxa_BufferPtr(VoxelWorldGlobals) voxel_globals = push.uses.voxel_globals;
daxa_BufferPtr(VoxelLeafChunk) voxel_chunks = push.uses.voxel_chunks;
daxa_BufferPtr(VoxelMallocPageAllocator) voxel_malloc_page_allocator = push.uses.voxel_malloc_page_allocator;
daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunks = push.uses.temp_voxel_chunks;
daxa_ImageViewId value_noise_texture = push.uses.value_noise_texture;
#endif
#endif

#if ChunkOptComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(ChunkOptCompute, 5)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(TempVoxelChunk), temp_voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_DECL_TASK_HEAD_END
struct ChunkOptComputePush {
    DAXA_TH_BLOB(ChunkOptCompute, uses)
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(ChunkOptComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_BufferPtr(VoxelWorldGlobals) voxel_globals = push.uses.voxel_globals;
daxa_RWBufferPtr(TempVoxelChunk) temp_voxel_chunks = push.uses.temp_voxel_chunks;
daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunks = push.uses.voxel_chunks;
#endif
#endif

#if ChunkAllocComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(ChunkAllocCompute, 6)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(TempVoxelChunk), temp_voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelMallocPageAllocator), voxel_malloc_page_allocator)
DAXA_DECL_TASK_HEAD_END
struct ChunkAllocComputePush {
    DAXA_TH_BLOB(ChunkAllocCompute, uses)
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(ChunkAllocComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_BufferPtr(VoxelWorldGlobals) voxel_globals = push.uses.voxel_globals;
daxa_BufferPtr(TempVoxelChunk) temp_voxel_chunks = push.uses.temp_voxel_chunks;
daxa_RWBufferPtr(VoxelLeafChunk) voxel_chunks = push.uses.voxel_chunks;
daxa_RWBufferPtr(VoxelMallocPageAllocator) voxel_malloc_page_allocator = push.uses.voxel_malloc_page_allocator;
#endif
#endif

#if defined(__cplusplus)

struct VoxelWorld : AppUi::DebugDisplayProvider {
    struct Buffers {
        daxa::BufferId voxel_globals_buffer;
        daxa::TaskBuffer task_voxel_globals_buffer{{.name = "task_voxel_globals_buffer"}};
        daxa::BufferId voxel_chunks_buffer;
        daxa::TaskBuffer task_voxel_chunks_buffer{{.name = "task_voxel_chunks_buffer"}};
        AllocatorBufferState<VoxelMallocPageAllocator> voxel_malloc;
        // AllocatorBufferState<VoxelLeafChunkAllocator> voxel_leaf_chunk_malloc;
        // AllocatorBufferState<VoxelParentChunkAllocator> voxel_parent_chunk_malloc;
    };

    Buffers buffers;
    daxa_u32 debug_page_count{};
    daxa_u32 debug_gpu_heap_usage{};

    virtual ~VoxelWorld() override = default;

    virtual void add_ui() override {
        if (ImGui::TreeNode("Voxel World")) {
            ImGui::Text("Page count: %u pages (%.2f MB)", debug_page_count, static_cast<double>(debug_page_count) * VOXEL_MALLOC_PAGE_SIZE_BYTES / 1'000'000.0);
            ImGui::Text("GPU heap usage: %.2f MB", static_cast<double>(debug_gpu_heap_usage) / 1'000'000);
            ImGui::TreePop();
        }
    }

    void create(daxa::Device &device) {
        buffers.voxel_globals_buffer = device.create_buffer({
            .size = static_cast<daxa_u32>(sizeof(VoxelWorldGlobals)),
            .name = "voxel_globals_buffer",
        });
        buffers.task_voxel_globals_buffer.set_buffers({.buffers = std::array{buffers.voxel_globals_buffer}});

        auto chunk_n = (1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);
        chunk_n = chunk_n * chunk_n * chunk_n * CHUNK_LOD_LEVELS;
        buffers.voxel_chunks_buffer = device.create_buffer({
            .size = static_cast<daxa_u32>(sizeof(VoxelLeafChunk)) * chunk_n,
            .name = "voxel_chunks_buffer",
        });
        buffers.task_voxel_chunks_buffer.set_buffers({.buffers = std::array{buffers.voxel_chunks_buffer}});

        buffers.voxel_malloc.create(device);
        // buffers.voxel_leaf_chunk_malloc.create(device);
        // buffers.voxel_parent_chunk_malloc.create(device);
    }
    void destroy(daxa::Device &device) const {
        device.destroy_buffer(buffers.voxel_globals_buffer);
        device.destroy_buffer(buffers.voxel_chunks_buffer);
        buffers.voxel_malloc.destroy(device);
        // buffers.voxel_leaf_chunk_malloc.destroy(device);
        // buffers.voxel_parent_chunk_malloc.destroy(device);
    }

    void for_each_buffer(auto func) {
        func(buffers.voxel_globals_buffer);
        func(buffers.voxel_chunks_buffer);
        buffers.voxel_malloc.for_each_buffer(func);
        // buffers.voxel_leaf_chunk_malloc.for_each_buffer(func);
        // buffers.voxel_parent_chunk_malloc.for_each_buffer(func);
    }

    void record_startup(RecordContext &record_ctx) {
        record_ctx.task_graph.add_task({
            .attachments = {
                daxa::inl_atch(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.task_voxel_globals_buffer),
                daxa::inl_atch(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.task_voxel_chunks_buffer),
                daxa::inl_atch(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_malloc.task_element_buffer),
                // daxa::inl_atch(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_leaf_chunk_malloc.task_element_buffer),
                // daxa::inl_atch(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_parent_chunk_malloc.task_element_buffer),
            },
            .task = [this](daxa::TaskInterface const &ti) {
                ti.recorder.clear_buffer({
                    .buffer = buffers.task_voxel_globals_buffer.get_state().buffers[0],
                    .offset = 0,
                    .size = sizeof(VoxelWorldGlobals),
                    .clear_value = 0,
                });

                auto chunk_n = (1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);
                chunk_n = chunk_n * chunk_n * chunk_n * CHUNK_LOD_LEVELS;
                ti.recorder.clear_buffer({
                    .buffer = buffers.task_voxel_chunks_buffer.get_state().buffers[0],
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
                daxa::inl_atch(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_malloc.task_allocator_buffer),
                // daxa::inl_atch(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_leaf_chunk_malloc.task_allocator_buffer),
                // daxa::inl_atch(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_parent_chunk_malloc.task_allocator_buffer),
            },
            .task = [this](daxa::TaskInterface const &ti) {
                buffers.voxel_malloc.init(ti.device, ti.recorder);
                // buffers.voxel_leaf_chunk_malloc.init(ti.device, ti.recorder);
                // buffers.voxel_parent_chunk_malloc.init(ti.device, ti.recorder);
            },
            .name = "Initialize",
        });
    }

    auto check_for_realloc(daxa::Device &device, VoxelWorldOutput const &gpu_output) -> bool {
        buffers.voxel_malloc.check_for_realloc(device, gpu_output.voxel_malloc_output.current_element_count);
        // buffers.voxel_leaf_chunk_malloc.check_for_realloc(device, gpu_output.voxel_leaf_chunk_output.current_element_count);
        // buffers.voxel_parent_chunk_malloc.check_for_realloc(device, gpu_output.voxel_parent_chunk_output.current_element_count);

        bool needs_realloc = false;
        needs_realloc = needs_realloc || buffers.voxel_malloc.needs_realloc();
        // needs_realloc = needs_realloc || buffers.voxel_leaf_chunk_malloc.needs_realloc();
        // needs_realloc = needs_realloc || buffers.voxel_parent_chunk_malloc.needs_realloc();

        debug_gpu_heap_usage = gpu_output.voxel_malloc_output.current_element_count * VOXEL_MALLOC_PAGE_SIZE_BYTES;
        debug_page_count = buffers.voxel_malloc.current_element_count;

        return needs_realloc;
    }

    void dynamic_buffers_realloc(daxa::TaskGraph &temp_task_graph, bool &needs_vram_calc) {
        buffers.voxel_malloc.for_each_task_buffer([&temp_task_graph](auto &task_buffer) { temp_task_graph.use_persistent_buffer(task_buffer); });
        // buffers.voxel_leaf_chunk_malloc.for_each_task_buffer([&temp_task_graph](auto &task_buffer) { temp_task_graph.use_persistent_buffer(task_buffer); });
        // buffers.voxel_parent_chunk_malloc.for_each_task_buffer([&temp_task_graph](auto &task_buffer) { temp_task_graph.use_persistent_buffer(task_buffer); });
        temp_task_graph.add_task({
            .attachments = {
                daxa::inl_atch(daxa::TaskBufferAccess::TRANSFER_READ, buffers.voxel_malloc.task_old_element_buffer),
                daxa::inl_atch(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_malloc.task_element_buffer),
                // daxa::inl_atch(daxa::TaskBufferAccess::TRANSFER_READ, buffers.voxel_leaf_chunk_malloc.task_old_element_buffer),
                // daxa::inl_atch(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_leaf_chunk_malloc.task_element_buffer),
                // daxa::inl_atch(daxa::TaskBufferAccess::TRANSFER_READ, buffers.voxel_parent_chunk_malloc.task_old_element_buffer),
                // daxa::inl_atch(daxa::TaskBufferAccess::TRANSFER_WRITE, buffers.voxel_parent_chunk_malloc.task_element_buffer),
            },
            .task = [this, &needs_vram_calc](daxa::TaskInterface const &ti) {
                if (buffers.voxel_malloc.needs_realloc()) {
                    buffers.voxel_malloc.realloc(ti.device, ti.recorder);
                    needs_vram_calc = true;
                }
                // if (buffers.voxel_leaf_chunk_malloc.needs_realloc()) {
                //     buffers.voxel_leaf_chunk_malloc.realloc(ti.device, ti.recorder);
                //     needs_vram_calc = true;
                // }
                // if (buffers.voxel_parent_chunk_malloc.needs_realloc()) {
                //     buffers.voxel_parent_chunk_malloc.realloc(ti.device, ti.recorder);
                //     needs_vram_calc = true;
                // }
            },
            .name = "Transfer Task",
        });
    }

    void use_buffers(RecordContext &record_ctx) {
        record_ctx.task_graph.use_persistent_buffer(buffers.task_voxel_globals_buffer);
        record_ctx.task_graph.use_persistent_buffer(buffers.task_voxel_chunks_buffer);
        buffers.voxel_malloc.for_each_task_buffer([&record_ctx](auto &task_buffer) { record_ctx.task_graph.use_persistent_buffer(task_buffer); });
        // buffers.voxel_leaf_chunk_malloc.for_each_task_buffer([&record_ctx](auto &task_buffer) { record_ctx.task_graph.use_persistent_buffer(task_buffer); });
        // buffers.voxel_parent_chunk_malloc.for_each_task_buffer([&record_ctx](auto &task_buffer) { record_ctx.task_graph.use_persistent_buffer(task_buffer); });
    }

    void record_frame(RecordContext &record_ctx, daxa::TaskBufferView task_gvox_model_buffer, daxa::TaskImageView task_value_noise_image) {
        record_ctx.add(ComputeTask<PerChunkCompute, PerChunkComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{PerChunkCompute::gpu_input, record_ctx.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{PerChunkCompute::gvox_model, task_gvox_model_buffer}},
                daxa::TaskViewVariant{std::pair{PerChunkCompute::globals, record_ctx.task_globals_buffer}},
                daxa::TaskViewVariant{std::pair{PerChunkCompute::voxel_globals, buffers.task_voxel_globals_buffer}},
                daxa::TaskViewVariant{std::pair{PerChunkCompute::voxel_chunks, buffers.task_voxel_chunks_buffer}},
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
                daxa::TaskViewVariant{std::pair{ChunkEditCompute::gpu_input, record_ctx.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkEditCompute::globals, record_ctx.task_globals_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkEditCompute::gvox_model, task_gvox_model_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkEditCompute::voxel_globals, buffers.task_voxel_globals_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkEditCompute::voxel_chunks, buffers.task_voxel_chunks_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkEditCompute::voxel_malloc_page_allocator, buffers.voxel_malloc.task_allocator_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkEditCompute::temp_voxel_chunks, task_temp_voxel_chunks_buffer}},
                // daxa::TaskViewVariant{std::pair{ChunkEditCompute::simulated_voxel_particles, task_simulated_voxel_particles_buffer}},
                // daxa::TaskViewVariant{std::pair{ChunkEditCompute::placed_voxel_particles, task_placed_voxel_particles_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkEditCompute::value_noise_texture, task_value_noise_image.view({.layer_count = 256})}},
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
                daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::gpu_input, record_ctx.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::globals, record_ctx.task_globals_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::gvox_model, task_gvox_model_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::voxel_globals, buffers.task_voxel_globals_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkEditPostProcessCompute::voxel_chunks, buffers.task_voxel_chunks_buffer}},
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
                daxa::TaskViewVariant{std::pair{ChunkOptCompute::gpu_input, record_ctx.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkOptCompute::globals, record_ctx.task_globals_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkOptCompute::voxel_globals, buffers.task_voxel_globals_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkOptCompute::temp_voxel_chunks, task_temp_voxel_chunks_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkOptCompute::voxel_chunks, buffers.task_voxel_chunks_buffer}},
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
                daxa::TaskViewVariant{std::pair{ChunkOptCompute::gpu_input, record_ctx.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkOptCompute::globals, record_ctx.task_globals_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkOptCompute::voxel_globals, buffers.task_voxel_globals_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkOptCompute::temp_voxel_chunks, task_temp_voxel_chunks_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkOptCompute::voxel_chunks, buffers.task_voxel_chunks_buffer}},
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
                daxa::TaskViewVariant{std::pair{ChunkAllocCompute::gpu_input, record_ctx.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkAllocCompute::globals, record_ctx.task_globals_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkAllocCompute::voxel_globals, buffers.task_voxel_globals_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkAllocCompute::temp_voxel_chunks, task_temp_voxel_chunks_buffer}},
                daxa::TaskViewVariant{std::pair{ChunkAllocCompute::voxel_chunks, buffers.task_voxel_chunks_buffer}},
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
};

#endif
