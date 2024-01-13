#pragma once

#include <shared/voxels/impl/voxels.inl>
#include <shared/globals.inl>

#if PER_CHUNK_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(PerChunkCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGvoxModel), gvox_model)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D_ARRAY, value_noise_texture)
DAXA_DECL_TASK_HEAD_END
struct PerChunkComputePush {
    PerChunkCompute uses;
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

#if CHUNK_EDIT_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(ChunkEditCompute)
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
    ChunkEditCompute uses;
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

#if CHUNK_OPT_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(ChunkOptCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(TempVoxelChunk), temp_voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_DECL_TASK_HEAD_END
struct ChunkOptComputePush {
    ChunkOptCompute uses;
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

#if CHUNK_ALLOC_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(ChunkAllocCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VoxelWorldGlobals), voxel_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(TempVoxelChunk), temp_voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelLeafChunk), voxel_chunks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VoxelMallocPageAllocator), voxel_malloc_page_allocator)
DAXA_DECL_TASK_HEAD_END
struct ChunkAllocComputePush {
    ChunkAllocCompute uses;
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

struct PerChunkComputeTaskState {
    AsyncManagedComputePipeline pipeline;

    PerChunkComputeTaskState(AsyncPipelineManager &pipeline_manager) {
        pipeline = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
                .compile_options = {.defines = {{"PER_CHUNK_COMPUTE", "1"}}},
            },
            .push_constant_size = sizeof(PerChunkComputePush),
            .name = "per_chunk",
        });
    }

    void record_commands(PerChunkComputePush const &push, daxa::CommandRecorder &recorder) {
        if (!pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(pipeline.get());
        recorder.push_constant(push);
        auto const dispatch_size = 1 << LOG2_CHUNKS_DISPATCH_SIZE;
        recorder.dispatch({dispatch_size, dispatch_size, dispatch_size * CHUNK_LOD_LEVELS});
    }
};

struct ChunkEditComputeTaskState {
    AsyncManagedComputePipeline pipeline;

    ChunkEditComputeTaskState(AsyncPipelineManager &pipeline_manager) {
        pipeline = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
                .compile_options = {.defines = {{"CHUNK_EDIT_COMPUTE", "1"}}},
            },
            .push_constant_size = sizeof(ChunkEditComputePush),
            .name = "voxel_world",
        });
    }

    void record_commands(ChunkEditComputePush const &push, daxa::CommandRecorder &recorder, daxa::BufferId globals_buffer_id) {
        if (!pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(pipeline.get());
        recorder.push_constant(push);
        recorder.dispatch_indirect({
            .indirect_buffer = globals_buffer_id,
            .offset = offsetof(GpuGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, chunk_edit_dispatch),
        });
    }
};

template <int PASS_INDEX>
struct ChunkOptComputeTaskState {
    AsyncManagedComputePipeline pipeline;

    ChunkOptComputeTaskState(AsyncPipelineManager &pipeline_manager) {
        char const define_str[2] = {'0' + PASS_INDEX, '\0'};
        pipeline = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
                .compile_options = {
                    .defines = {
                        {"CHUNK_OPT_COMPUTE", "1"},
                        {"CHUNK_OPT_STAGE", define_str},
                    },
                },
            },
            .push_constant_size = sizeof(ChunkOptComputePush),
            .name = "chunk_op",
        });
    }

    auto get_pass_indirect_offset() {
        if constexpr (PASS_INDEX == 0) {
            return offsetof(GpuGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, subchunk_x2x4_dispatch);
        } else {
            return offsetof(GpuGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, subchunk_x8up_dispatch);
        }
    }

    void record_commands(ChunkOptComputePush const &push, daxa::CommandRecorder &recorder, daxa::BufferId globals_buffer_id) {
        if (!pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(pipeline.get());
        recorder.push_constant(push);
        recorder.dispatch_indirect({
            .indirect_buffer = globals_buffer_id,
            .offset = get_pass_indirect_offset(),
        });
    }
};

struct ChunkAllocComputeTaskState {
    AsyncManagedComputePipeline pipeline;

    ChunkAllocComputeTaskState(AsyncPipelineManager &pipeline_manager) {
        pipeline = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"voxels/impl/voxel_world.comp.glsl"},
                .compile_options = {.defines = {{"CHUNK_ALLOC_COMPUTE", "1"}}},
            },
            .push_constant_size = sizeof(ChunkAllocComputePush),
            .name = "chunk_alloc",
        });
    }

    void record_commands(ChunkAllocComputePush const &push, daxa::CommandRecorder &recorder, daxa::BufferId globals_buffer_id) {
        if (!pipeline.is_valid()) {
            return;
        }
        recorder.set_pipeline(pipeline.get());
        recorder.push_constant(push);
        recorder.dispatch_indirect({
            .indirect_buffer = globals_buffer_id,
            // NOTE: This should always have the same value as the chunk edit dispatch, so we're re-using it here
            .offset = offsetof(GpuGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, chunk_edit_dispatch),
        });
    }
};

struct PerChunkComputeTask {
    PerChunkCompute::Uses uses;
    PerChunkComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto push = PerChunkComputePush{};
        ti.copy_task_head_to(&push.uses);
        state->record_commands(push, recorder);
    }
};

struct ChunkEditComputeTask {
    ChunkEditCompute::Uses uses;
    ChunkEditComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto push = ChunkEditComputePush{};
        ti.copy_task_head_to(&push.uses);
        state->record_commands(push, recorder, uses.globals.buffer());
    }
};

template <int PASS_INDEX>
struct ChunkOptComputeTask {
    ChunkOptCompute::Uses uses;
    ChunkOptComputeTaskState<PASS_INDEX> *state;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto push = ChunkOptComputePush{};
        ti.copy_task_head_to(&push.uses);
        state->record_commands(push, recorder, uses.globals.buffer());
    }
};

using ChunkOpt_x2x4_ComputeTaskState = ChunkOptComputeTaskState<0>;
using ChunkOpt_x8up_ComputeTaskState = ChunkOptComputeTaskState<1>;
using ChunkOpt_x2x4_ComputeTask = ChunkOptComputeTask<0>;
using ChunkOpt_x8up_ComputeTask = ChunkOptComputeTask<1>;

struct ChunkAllocComputeTask {
    ChunkAllocCompute::Uses uses;
    ChunkAllocComputeTaskState *state;
    void callback(daxa::TaskInterface const &ti) {
        auto &recorder = ti.get_recorder();
        auto push = ChunkAllocComputePush{};
        ti.copy_task_head_to(&push.uses);
        state->record_commands(push, recorder, uses.globals.buffer());
    }
};

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

    PerChunkComputeTaskState per_chunk_task_state;
    ChunkEditComputeTaskState chunk_edit_task_state;
    ChunkOpt_x2x4_ComputeTaskState chunk_opt_x2x4_task_state;
    ChunkOpt_x8up_ComputeTaskState chunk_opt_x8up_task_state;
    ChunkAllocComputeTaskState chunk_alloc_task_state;

    VoxelWorld(AsyncPipelineManager &pipeline_manager)
        : per_chunk_task_state{pipeline_manager},
          chunk_edit_task_state{pipeline_manager},
          chunk_opt_x2x4_task_state{pipeline_manager},
          chunk_opt_x8up_task_state{pipeline_manager},
          chunk_alloc_task_state{pipeline_manager} {
    }
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
            .uses = {
                daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{buffers.task_voxel_globals_buffer},
                daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{buffers.task_voxel_chunks_buffer},
                daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{buffers.voxel_malloc.task_element_buffer},
                // daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{buffers.voxel_leaf_chunk_malloc.task_element_buffer},
                // daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{buffers.voxel_parent_chunk_malloc.task_element_buffer},
            },
            .task = [this](daxa::TaskInterface ti) {
                auto &recorder = ti.get_recorder();

                recorder.clear_buffer({
                    .buffer = buffers.task_voxel_globals_buffer.get_state().buffers[0],
                    .offset = 0,
                    .size = sizeof(VoxelWorldGlobals),
                    .clear_value = 0,
                });

                auto chunk_n = (1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);
                chunk_n = chunk_n * chunk_n * chunk_n * CHUNK_LOD_LEVELS;
                recorder.clear_buffer({
                    .buffer = buffers.task_voxel_chunks_buffer.get_state().buffers[0],
                    .offset = 0,
                    .size = sizeof(VoxelLeafChunk) * chunk_n,
                    .clear_value = 0,
                });

                buffers.voxel_malloc.clear_buffers(recorder);
                // buffers.voxel_leaf_chunk_malloc.clear_buffers(recorder);
                // buffers.voxel_parent_chunk_malloc.clear_buffers(recorder);
            },
            .name = "clear chunk editor",
        });

        record_ctx.task_graph.add_task({
            .uses = {
                daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{buffers.voxel_malloc.task_allocator_buffer},
                // daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{buffers.voxel_leaf_chunk_malloc.task_allocator_buffer},
                // daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{buffers.voxel_parent_chunk_malloc.task_allocator_buffer},
            },
            .task = [this](daxa::TaskInterface ti) {
                auto &recorder = ti.get_recorder();
                buffers.voxel_malloc.init(ti.get_device(), recorder);
                // buffers.voxel_leaf_chunk_malloc.init(ti.get_device(), recorder);
                // buffers.voxel_parent_chunk_malloc.init(ti.get_device(), recorder);
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
            .uses = {
                daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_READ>{buffers.voxel_malloc.task_old_element_buffer},
                daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{buffers.voxel_malloc.task_element_buffer},
                // daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_READ>{buffers.voxel_leaf_chunk_malloc.task_old_element_buffer},
                // daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{buffers.voxel_leaf_chunk_malloc.task_element_buffer},
                // daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_READ>{buffers.voxel_parent_chunk_malloc.task_old_element_buffer},
                // daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{buffers.voxel_parent_chunk_malloc.task_element_buffer},
            },
            .task = [this, &needs_vram_calc](daxa::TaskInterface ti) {
                auto &recorder = ti.get_recorder();
                if (buffers.voxel_malloc.needs_realloc()) {
                    buffers.voxel_malloc.realloc(ti.get_device(), recorder);
                    needs_vram_calc = true;
                }
                // if (buffers.voxel_leaf_chunk_malloc.needs_realloc()) {
                //     buffers.voxel_leaf_chunk_malloc.realloc(ti.get_device(), recorder);
                //     needs_vram_calc = true;
                // }
                // if (buffers.voxel_parent_chunk_malloc.needs_realloc()) {
                //     buffers.voxel_parent_chunk_malloc.realloc(ti.get_device(), recorder);
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
        record_ctx.task_graph.add_task(PerChunkComputeTask{
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .gvox_model = task_gvox_model_buffer,
                .globals = record_ctx.task_globals_buffer,
                .voxel_globals = buffers.task_voxel_globals_buffer,
                .voxel_chunks = buffers.task_voxel_chunks_buffer,
                .value_noise_texture = task_value_noise_image.view({.layer_count = 256}),
            },
            .state = &per_chunk_task_state,
        });

        auto task_temp_voxel_chunks_buffer = record_ctx.task_graph.create_transient_buffer({
            .size = sizeof(TempVoxelChunk) * MAX_CHUNK_UPDATES_PER_FRAME,
            .name = "temp_voxel_chunks_buffer",
        });

        record_ctx.task_graph.add_task(ChunkEditComputeTask{
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                .gvox_model = task_gvox_model_buffer,
                .voxel_globals = buffers.task_voxel_globals_buffer,
                .voxel_chunks = buffers.task_voxel_chunks_buffer,
                .voxel_malloc_page_allocator = buffers.voxel_malloc.task_allocator_buffer,
                .temp_voxel_chunks = task_temp_voxel_chunks_buffer,
                // .simulated_voxel_particles = task_simulated_voxel_particles_buffer,
                // .placed_voxel_particles = task_placed_voxel_particles_buffer,
                .value_noise_texture = task_value_noise_image.view({.layer_count = 256}),
            },
            .state = &chunk_edit_task_state,
        });

        record_ctx.task_graph.add_task(ChunkOpt_x2x4_ComputeTask{
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                .voxel_globals = buffers.task_voxel_globals_buffer,
                .temp_voxel_chunks = task_temp_voxel_chunks_buffer,
                .voxel_chunks = buffers.task_voxel_chunks_buffer,
            },
            .state = &chunk_opt_x2x4_task_state,
        });

        record_ctx.task_graph.add_task(ChunkOpt_x8up_ComputeTask{
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                .voxel_globals = buffers.task_voxel_globals_buffer,
                .temp_voxel_chunks = task_temp_voxel_chunks_buffer,
                .voxel_chunks = buffers.task_voxel_chunks_buffer,
            },
            .state = &chunk_opt_x8up_task_state,
        });

        record_ctx.task_graph.add_task(ChunkAllocComputeTask{
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .globals = record_ctx.task_globals_buffer,
                .voxel_globals = buffers.task_voxel_globals_buffer,
                .temp_voxel_chunks = task_temp_voxel_chunks_buffer,
                .voxel_chunks = buffers.task_voxel_chunks_buffer,
                .voxel_malloc_page_allocator = buffers.voxel_malloc.task_allocator_buffer,
            },
            .state = &chunk_alloc_task_state,
        });
    }
};

#endif
