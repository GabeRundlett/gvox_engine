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

using BDA = daxa::BufferDeviceAddress;

struct GpuInputUploadTransferTask {
    static void record(daxa::Device &device, daxa::CommandList &cmd_list, daxa::BufferId input_buffer, GpuInput &gpu_input);
};

struct StartupTask {
    std::shared_ptr<daxa::ComputePipeline> pipeline;
    void record(daxa::CommandList &cmd_list, BDA globals_buffer_ptr) const;
};

struct PerframeTask {
    std::shared_ptr<daxa::ComputePipeline> pipeline;
    void record(daxa::CommandList &cmd_list, BDA settings_buffer_ptr, BDA input_buffer_ptr, BDA globals_buffer_ptr) const;
};

struct TracePrimaryTask {
    std::shared_ptr<daxa::ComputePipeline> pipeline;
    void record(daxa::CommandList &cmd_list, BDA settings_buffer_ptr, BDA input_buffer_ptr, BDA globals_buffer_ptr, BDA gvox_model_buffer_ptr, daxa::ImageId render_image, u32vec2 render_size) const;
};

struct ColorSceneTask {
    std::shared_ptr<daxa::ComputePipeline> pipeline;
    void record(daxa::CommandList &cmd_list, BDA settings_buffer_ptr, BDA input_buffer_ptr, BDA globals_buffer_ptr, BDA gvox_model_buffer_ptr, daxa::ImageId render_pos_image, daxa::ImageId render_col_image, u32vec2 render_size) const;
};

struct PostprocessingTask {
    std::shared_ptr<daxa::ComputePipeline> pipeline;
    void record(daxa::CommandList &cmd_list, BDA settings_buffer_ptr, BDA input_buffer_ptr, BDA globals_buffer_ptr, BDA gvox_model_buffer_ptr, daxa::ImageId render_col_image, daxa::ImageId final_image, u32vec2 render_size) const;
};

struct RenderImages {
    u32vec2 size;
    daxa::ImageId pos_image;
    daxa::ImageId col_image;
    daxa::ImageId final_image;

    void create(daxa::Device &device);
    void destroy(daxa::Device &device) const;
};

struct GpuResources {
    RenderImages render_images;
    daxa::BufferId settings_buffer;
    daxa::BufferId input_buffer;
    daxa::BufferId globals_buffer;
    daxa::BufferId gvox_model_buffer;

    void create(daxa::Device &device);
    void destroy(daxa::Device &device) const;
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
    daxa::TaskBufferId task_globals_buffer;
    daxa::TaskBufferId task_gvox_model_buffer;

    GpuInputUploadTransferTask gpu_input_upload_transfer_task;
    StartupTask startup_task;
    PerframeTask perframe_task;
    TracePrimaryTask trace_primary_task;
    ColorSceneTask color_scene_task;
    PostprocessingTask postprocessing_task;

    daxa::TaskList main_task_list;
    daxa::CommandSubmitInfo submit_info;

    GpuInput gpu_input{};
    Clock::time_point start;
    Clock::time_point prev_time;
    f32 render_res_scl{1.0f};
    GvoxContext *gvox_ctx;
    bool has_model = false;
    bool needs_vram_calc = true;

    VoxelApp();
    ~VoxelApp();

    void recreate_render_images();

    auto update() -> bool;
    void on_update();
    void on_mouse_move(f32 x, f32 y);
    void on_mouse_scroll(f32 dx, f32 dy);
    void on_mouse_button(i32 button_id, i32 action);
    void on_key(i32 key_id, i32 action);
    void on_resize(u32 sx, u32 sy);

    void calc_vram_usage();

    void run_startup();
    void upload_settings();
    void upload_model();

    auto record_main_task_list() -> daxa::TaskList;
};
