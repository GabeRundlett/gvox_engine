#pragma once

#include "base_app.hpp"
#include <unordered_map>
#include <queue>

#include <gvox/gvox.h>

struct BrushSettings {
    bool limit_edit_rate;
    f32 edit_rate;
    f32vec3 color;
};

struct Brush {
    std::filesystem::path key;
    std::string display_name;
    std::filesystem::path thumbnail_image_path; // TODO: auto generated images don't have paths
    bool thumbnail_needs_updating;
    daxa::ImageId preview_thumbnail;
    daxa::TaskImageId task_preview_thumbnail;
    daxa::ComputePipeline perframe_comp_pipeline;
    daxa::ComputePipeline chunk_edit_comp_pipeline;
    BrushSettings settings;
};

struct ThreadPool {
    void start();
    void enqueue(std::function<void()> const &job);
    void stop();
    auto busy() -> bool;

  private:
    void thread_loop();

    bool should_terminate = false;
    std::mutex queue_mutex;
    std::condition_variable mutex_condition;
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> jobs;
};

struct App : BaseApp<App> {
    ThreadPool thread_pool;

    daxa::ComputePipeline startup_comp_pipeline;
    daxa::ComputePipeline optical_depth_comp_pipeline;
    daxa::ComputePipeline chunkgen_comp_pipeline;
    daxa::ComputePipeline subchunk_x2x4_comp_pipeline;
    daxa::ComputePipeline subchunk_x8up_comp_pipeline;
    daxa::ComputePipeline draw_comp_pipeline;

    GpuInput gpu_input;
    daxa::BufferId gpu_input_buffer;
    daxa::TaskBufferId task_gpu_input_buffer;

    f32 render_resolution_scl = 1.0f;
    u32 render_size_x = size_x, render_size_y = size_y;
    daxa::ImageId render_image;
    daxa::TaskImageId task_render_image;

    daxa::BufferId gpu_globals_buffer;
    daxa::TaskBufferId task_gpu_globals_buffer;

    BufferId gpu_indirect_dispatch_buffer;
    daxa::TaskBufferId task_gpu_indirect_dispatch_buffer;

    daxa::ImageId optical_depth_image;
    daxa::TaskImageId task_optical_depth_image;
    daxa::SamplerId optical_depth_sampler;

    std::filesystem::path data_directory;
    std::unordered_map<std::filesystem::path, Brush> brushes;
    std::filesystem::path current_brush_key;
    std::chrono::file_clock::time_point last_seen_brushes_folder_update;

    std::array<i32, GAME_KEY_LAST + 1> keys;
    std::array<i32, GAME_MOUSE_BUTTON_LAST + 1> mouse_buttons;

    i32 new_key_id, prev_key_id;
    usize new_key_index = GAME_KEY_LAST + 1;
    usize old_key_index = GAME_KEY_LAST + 1;

    bool controls_popup_is_open = false;
    bool paused = false;
    bool battery_saving_mode = false;
    bool should_run_startup = true;
    bool should_regenerate = false;
    bool should_regen_optical_depth = true;
    bool use_vsync = false;
    bool use_custom_resolution = false;
    bool show_menus = true;
    bool show_debug_menu = false;
    bool show_help_menu = false;
    bool show_generation_menu = false;
    bool show_tool_menu = true;
    bool show_tool_settings_menu = true;

    GVoxContext *gvox_ctx;
    GVoxScene gvox_model = {};
    daxa::BufferId gvox_model_buffer = {};
    daxa::TaskBufferId task_gvox_model_buffer = {};
    std::string gvox_model_path, gvox_model_type;
    u32 gvox_model_size;
    bool should_upload_gvox_model = true;

    std::array<float, 40> frametimes = {};
    u64 frametime_rotation_index = 0;
    std::string fmt_str;
    daxa::TaskList loop_task_list;

    App();
    ~App();

    void load_settings();
    void save_settings();
    void reset_settings();

    auto load_brushes() -> std::unordered_map<std::filesystem::path, Brush>;
    void reload_brushes();

    auto get_flag(u32 index) -> bool;
    void set_flag(u32 index, bool value);

    void ui_update();
    void on_update();

    void on_mouse_move(f32 x, f32 y);
    void on_mouse_scroll(f32 dx, f32 dy);
    void on_mouse_button(i32 button_id, i32 action);
    void on_key(i32 key_id, i32 action);
    void on_resize(u32 sx, u32 sy);

    void recreate_render_images();
    void toggle_menus();

    void submit_task_list();
    void record_tasks(daxa::TaskList &new_task_list);
};
