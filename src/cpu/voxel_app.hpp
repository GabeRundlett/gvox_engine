#pragma once

#include "app_window.hpp"
#include "app_ui.hpp"

#include <daxa/utils/pipeline_manager.hpp>
#include <daxa/utils/imgui.hpp>
#include <daxa/utils/task_list.hpp>
#include <daxa/utils/math_operators.hpp>
using namespace daxa::math_operators;

#include <shared/shared.inl>

#include <chrono>
#include <future>

using BDA = daxa::BufferDeviceAddress;

static constexpr usize FRAMES_IN_FLIGHT = 1;

struct GpuInputUploadTransferTask {
    static void record(daxa::Device &device, daxa::CommandList &cmd_list, daxa::BufferId input_buffer, GpuInput &gpu_input);
};

struct StartupTask {
    std::shared_ptr<daxa::ComputePipeline> pipeline;
    void record(
        daxa::CommandList &cmd_list,
        BDA globals_buffer_ptr,
        BDA voxel_chunks_buffer_ptr) const;
};

struct PerframeTask {
    std::shared_ptr<daxa::ComputePipeline> pipeline;
    void record(
        daxa::CommandList &cmd_list,
        BDA settings_buffer_ptr,
        BDA input_buffer_ptr,
        BDA output_buffer_ptr,
        BDA globals_buffer_ptr,
        BDA voxel_malloc_global_allocator_buffer_ptr,
        BDA voxel_chunks_buffer_ptr) const;
};

struct PerChunkTask {
    std::shared_ptr<daxa::ComputePipeline> pipeline;
    void record(
        daxa::CommandList &cmd_list,
        BDA settings_buffer_ptr,
        BDA input_buffer_ptr,
        BDA globals_buffer_ptr,
        BDA voxel_chunks_buffer_ptr,
        u32vec3 chunk_n) const;
};

struct ChunkEdit {
    std::shared_ptr<daxa::ComputePipeline> pipeline;
    void record(
        daxa::CommandList &cmd_list,
        BDA settings_buffer_ptr,
        BDA input_buffer_ptr,
        BDA globals_buffer_ptr,
        BDA temp_voxel_chunks_ptr,
        daxa::BufferId globals_buffer_id) const;
};

struct ChunkOpt_x2x4 {
    std::shared_ptr<daxa::ComputePipeline> pipeline;
    void record(
        daxa::CommandList &cmd_list,
        BDA settings_buffer_ptr,
        BDA input_buffer_ptr,
        BDA globals_buffer_ptr,
        BDA temp_voxel_chunks_ptr,
        BDA voxel_chunks_buffer_ptr,
        daxa::BufferId globals_buffer_id) const;
};

struct ChunkOpt_x8up {
    std::shared_ptr<daxa::ComputePipeline> pipeline;
    void record(
        daxa::CommandList &cmd_list,
        BDA settings_buffer_ptr,
        BDA input_buffer_ptr,
        BDA globals_buffer_ptr,
        BDA temp_voxel_chunks_ptr,
        BDA voxel_chunks_buffer_ptr,
        daxa::BufferId globals_buffer_id) const;
};

struct ChunkAlloc {
    std::shared_ptr<daxa::ComputePipeline> pipeline;
    void record(
        daxa::CommandList &cmd_list,
        BDA settings_buffer_ptr,
        BDA globals_buffer_ptr,
        BDA voxel_malloc_global_allocator_buffer_ptr,
        BDA temp_voxel_chunks_ptr,
        BDA voxel_chunks_buffer_ptr,
        daxa::BufferId globals_buffer_id) const;
};

struct TracePrimaryTask {
    std::shared_ptr<daxa::ComputePipeline> pipeline;
    void record(
        daxa::CommandList &cmd_list,
        BDA settings_buffer_ptr,
        BDA input_buffer_ptr,
        BDA globals_buffer_ptr,
        BDA voxel_malloc_global_allocator_buffer_ptr,
        BDA voxel_chunks_buffer_ptr,
        daxa::ImageId render_image,
        u32vec2 render_size) const;
};

struct ColorSceneTask {
    std::shared_ptr<daxa::ComputePipeline> pipeline;
    void record(
        daxa::CommandList &cmd_list,
        BDA settings_buffer_ptr,
        BDA input_buffer_ptr,
        BDA globals_buffer_ptr,
        BDA voxel_malloc_global_allocator_buffer_ptr,
        BDA voxel_chunks_buffer_ptr,
        daxa::ImageId render_pos_image,
        daxa::ImageId render_col_image,
        u32vec2 render_size) const;
};

struct PostprocessingTask {
    std::shared_ptr<daxa::ComputePipeline> pipeline;
    void record(
        daxa::CommandList &cmd_list,
        BDA settings_buffer_ptr,
        BDA input_buffer_ptr,
        BDA globals_buffer_ptr,
        daxa::ImageId render_col_image,
        daxa::ImageId final_image,
        u32vec2 render_size) const;
};

struct GpuOutputDownloadTransferTask {
    static void record(daxa::Device &device, daxa::CommandList &cmd_list, daxa::BufferId output_buffer, daxa::BufferId staging_output_buffer, GpuOutput &gpu_output, u32 frame_index);
};

struct RenderImages {
    u32vec2 size;
    daxa::ImageId pos_image;
    daxa::ImageId col_image;
    daxa::ImageId final_image;

    void create(daxa::Device &device);
    void destroy(daxa::Device &device) const;
};

struct VoxelChunks {
    daxa::BufferId buffer;

    void create(daxa::Device &device, u32 log2_chunks_per_axis);
    void destroy(daxa::Device &device) const;
};

struct VoxelMalloc {
    daxa::BufferId global_allocator_buffer;
    daxa::BufferId pages_buffer;
    daxa::BufferId available_pages_stack_buffer;
    daxa::BufferId released_pages_stack_buffer;
    u32 current_page_count = 0;
    u32 next_page_count = 0;

    void create(daxa::Device &device, u32 page_count);
    void destroy(daxa::Device &device) const;
};

struct GpuHeap {
    daxa::BufferId buffer;

    void create(daxa::Device &device, u32 size);
    void destroy(daxa::Device &device) const;
};

struct GpuResources {
    RenderImages render_images;
    daxa::BufferId settings_buffer;
    daxa::BufferId input_buffer;
    daxa::BufferId output_buffer;
    daxa::BufferId staging_output_buffer;
    daxa::BufferId globals_buffer;
    daxa::BufferId temp_voxel_chunks_buffer;
    VoxelChunks voxel_chunks;
    VoxelMalloc voxel_malloc;
    GpuHeap gpu_heap;
    daxa::BufferId gvox_model_buffer;

    void create(daxa::Device &device);
    void destroy(daxa::Device &device) const;
};

struct GvoxModelData {
    size_t size = 0;
    uint8_t *ptr = nullptr;
};

struct VoxelApp : AppWindow<VoxelApp> {
    using Clock = std::chrono::high_resolution_clock;

    daxa::Context daxa_ctx;
    daxa::Device device;

    daxa::Swapchain swapchain;
    daxa::ImageId swapchain_image{};
    daxa::TaskImageId task_swapchain_image;

    daxa::PipelineManager main_pipeline_manager;

    AppUi ui;
    daxa::ImGuiRenderer imgui_renderer;

    GpuResources gpu_resources;

    daxa::TaskImageId task_render_pos_image;
    daxa::TaskImageId task_render_col_image;
    daxa::TaskImageId task_render_final_image;

    daxa::TaskBufferId task_settings_buffer;
    daxa::TaskBufferId task_input_buffer;
    daxa::TaskBufferId task_output_buffer;
    daxa::TaskBufferId task_staging_output_buffer;
    daxa::TaskBufferId task_globals_buffer;
    daxa::TaskBufferId task_temp_voxel_chunks_buffer;
    daxa::TaskBufferId task_voxel_malloc_global_allocator_buffer;
    daxa::TaskBufferId task_voxel_malloc_pages_buffer;
    daxa::TaskBufferId task_voxel_malloc_new_pages_buffer;
    daxa::TaskBufferId task_voxel_chunks_buffer;
    daxa::TaskBufferId task_gpu_heap_buffer;
    daxa::TaskBufferId task_gvox_model_buffer;

    GpuInputUploadTransferTask gpu_input_upload_transfer_task;
    StartupTask startup_task;
    PerframeTask perframe_task;
    PerChunkTask per_chunk_task;
    ChunkEdit chunk_edit_task;
    ChunkOpt_x2x4 chunk_opt_x2x4_task;
    ChunkOpt_x8up chunk_opt_x8up_task;
    ChunkAlloc chunk_alloc_task;
    TracePrimaryTask trace_primary_task;
    ColorSceneTask color_scene_task;
    PostprocessingTask postprocessing_task;
    GpuOutputDownloadTransferTask gpu_output_download_transfer_task;

    enum class Conditions {
        STARTUP,
        UPLOAD_SETTINGS,
        UPLOAD_GVOX_MODEL,
        VOXEL_MALLOC_REALLOC,
        LAST,
    };
    daxa::TaskList main_task_list;
    daxa::CommandSubmitInfo submit_info;
    std::array<bool, static_cast<usize>(Conditions::LAST)> condition_values{};

    GpuInput gpu_input{};
    GpuOutput gpu_output{};
    Clock::time_point start;
    Clock::time_point prev_time;
    f32 render_res_scl{1.0f};
    GvoxContext *gvox_ctx;
    bool needs_vram_calc = true;

    bool has_model = false;
    std::future<GvoxModelData> gvox_model_data_future;
    bool model_is_loading = false;

    VoxelApp();
    ~VoxelApp();

    void run();

    void calc_vram_usage();
    auto load_gvox_data() -> GvoxModelData;

    void on_update();
    void on_mouse_move(f32 x, f32 y);
    void on_mouse_scroll(f32 dx, f32 dy);
    void on_mouse_button(i32 button_id, i32 action);
    void on_key(i32 key_id, i32 action);
    void on_resize(u32 sx, u32 sy);

    void recreate_render_images();
    void recreate_voxel_chunks();

    void run_startup(daxa::TaskList &temp_task_list);
    void upload_settings(daxa::TaskList &temp_task_list);
    void upload_model(daxa::TaskList &temp_task_list);
    void voxel_malloc_realloc(daxa::TaskList &temp_task_list);

    auto record_main_task_list() -> daxa::TaskList;
};
