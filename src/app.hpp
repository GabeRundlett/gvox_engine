#pragma once

#include "base_app.hpp"
#include "custom_ui_components.hpp"

#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>

struct BrushSettings {
    bool limit_edit_rate;
    f32 edit_rate;
    f32vec3 color;
};

struct BrushPipelines {
    daxa::PipelineManager pipeline_manager;

    bool compiled = false;
    bool valid = false;

    daxa::ComputePipelineCompileInfo perframe_comp_info;
    std::shared_ptr<daxa::ComputePipeline> perframe_comp;

    daxa::ComputePipelineCompileInfo chunk_edit_comp_info;
    std::shared_ptr<daxa::ComputePipeline> chunk_edit_comp;

    daxa::ComputePipelineCompileInfo chunkgen_comp_info;
    std::shared_ptr<daxa::ComputePipeline> chunkgen_comp;

    daxa::ComputePipelineCompileInfo brush_chunkgen_comp_info;
    std::shared_ptr<daxa::ComputePipeline> brush_chunkgen_comp;

    void compile() {
        if (!compiled) {
            auto perframe_comp_result = pipeline_manager.add_compute_pipeline(perframe_comp_info);
            auto chunk_edit_comp_result = pipeline_manager.add_compute_pipeline(chunk_edit_comp_info);
            auto chunkgen_comp_result = pipeline_manager.add_compute_pipeline(chunkgen_comp_info);
            auto brush_chunkgen_comp_result = pipeline_manager.add_compute_pipeline(brush_chunkgen_comp_info);
            if (perframe_comp_result.is_ok() &&
                chunk_edit_comp_result.is_ok() &&
                chunkgen_comp_result.is_ok() &&
                brush_chunkgen_comp_result.is_ok()) {

                perframe_comp = perframe_comp_result.value();
                chunk_edit_comp = chunk_edit_comp_result.value();
                chunkgen_comp = chunkgen_comp_result.value();
                brush_chunkgen_comp = brush_chunkgen_comp_result.value();

                compiled = true;
                valid = true;
            } else if (perframe_comp_result.is_err()) {
                imgui_console.add_log("[error] %s", perframe_comp_result.message().c_str());
            } else if (chunk_edit_comp_result.is_err()) {
                imgui_console.add_log("[error] %s", chunk_edit_comp_result.message().c_str());
            } else if (chunkgen_comp_result.is_err()) {
                imgui_console.add_log("[error] %s", chunkgen_comp_result.message().c_str());
            } else if (brush_chunkgen_comp_result.is_err()) {
                imgui_console.add_log("[error] %s", brush_chunkgen_comp_result.message().c_str());
            }
        }
    }

    auto &get_perframe_comp() {
        compile();
        return perframe_comp;
    }
    auto &get_chunk_edit_comp() {
        compile();
        return chunk_edit_comp;
    }
    auto &get_chunkgen_comp() {
        compile();
        return chunkgen_comp;
    }
    auto &get_brush_chunkgen_comp() {
        compile();
        return brush_chunkgen_comp;
    }
};

struct Brush {
    std::filesystem::path key;
    std::string display_name;
    std::filesystem::path thumbnail_image_path; // TODO: auto generated images don't have paths
    bool thumbnail_needs_updating;
    daxa::ImageId preview_thumbnail;
    daxa::TaskImageId task_preview_thumbnail;
    BrushPipelines pipelines;
    BrushSettings settings;

    std::vector<CustomUIParameter> custom_brush_settings;
    daxa::BufferId custom_brush_settings_buffer;
    usize custom_buffer_size;
    u8 *custom_brush_settings_data;

    void cleanup(daxa::Device &device);
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

    std::shared_ptr<daxa::ComputePipeline> startup_comp_pipeline;
    std::shared_ptr<daxa::ComputePipeline> optical_depth_comp_pipeline;
    std::shared_ptr<daxa::ComputePipeline> subchunk_x2x4_comp_pipeline;
    std::shared_ptr<daxa::ComputePipeline> subchunk_x8up_comp_pipeline;
    std::shared_ptr<daxa::ComputePipeline> subchunk_brush_x2x4_comp_pipeline;
    std::shared_ptr<daxa::ComputePipeline> subchunk_brush_x8up_comp_pipeline;
    std::shared_ptr<daxa::ComputePipeline> draw_comp_pipeline;

    GpuInput gpu_input;
    daxa::BufferId gpu_input_buffer;
    daxa::TaskBufferId task_gpu_input_buffer;

    daxa::TaskBufferId task_gpu_brush_settings_buffer;

    f32 render_resolution_scl = 1.0f;
#if RENDER_PERF_TESTING
    u32 render_size_x = 100, render_size_y = 100;
    u64 chunk_render_frame_index = 0;
#else
    u32 render_size_x = size_x, render_size_y = size_y;
#endif
    daxa::ImageId render_image;
    daxa::TaskImageId task_render_image;

    std::shared_ptr<daxa::ComputePipeline> raytrace_comp_pipeline;
    daxa::ImageId raytrace_output_image;
    daxa::TaskImageId task_raytrace_output_image;

    daxa::BufferId gpu_globals_buffer;
    daxa::TaskBufferId task_gpu_globals_buffer;

    daxa::BufferId gpu_voxel_world_buffer;
    daxa::TaskBufferId task_gpu_voxel_world_buffer;

    daxa::BufferId gpu_voxel_brush_buffer;
    daxa::TaskBufferId task_gpu_voxel_brush_buffer;

    BufferId gpu_indirect_dispatch_buffer;
    daxa::TaskBufferId task_gpu_indirect_dispatch_buffer;

    daxa::ImageId optical_depth_image;
    daxa::TaskImageId task_optical_depth_image;
    daxa::SamplerId optical_depth_sampler;

    std::filesystem::path data_directory;
    std::unordered_map<std::string, Brush> brushes;
    std::string chunkgen_brush_key;
    std::string current_brush_key;
    std::chrono::file_clock::time_point last_seen_brushes_folder_update;

    u32 current_tool = GAME_TOOL_BRUSH;

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
#if RENDER_PERF_TESTING
    bool use_custom_resolution = true;
#else
    bool use_custom_resolution = false;
#endif
    bool show_menus = true;
    bool show_console = true;
    bool show_debug_menu = false;
    bool show_help_menu = false;
    bool show_tool_menu = true;
    bool show_tool_settings_menu = true;

    daxa::BufferId gvox_model_buffer = {};
    daxa::TaskBufferId task_gvox_model_buffer = {};
    std::string gvox_model_path, gvox_model_type;
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

    auto load_brushes() -> std::unordered_map<std::string, Brush>;
    void reload_brushes();

    auto get_flag(u32 index) -> bool;
    void set_flag(u32 index, bool value);
    void imgui_gpu_input_flag_checkbox(char const *const str, u32 flag_index);

    void brush_tool_ui();
    void settings_ui();

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
