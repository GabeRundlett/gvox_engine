#pragma once

struct GLFWwindow;
struct ImFont;

#include "settings.hpp"
#include <imgui.h>
#include <chrono>
#include <filesystem>
#include <thread>
#include <mutex>
#include <fmt/format.h>

#define INVALID_GAME_ACTION (-1)

struct AppUi {
    using Clock = std::chrono::high_resolution_clock;

    AppUi(GLFWwindow *glfw_window_ptr);
    ~AppUi();

    AppSettings settings;

    GLFWwindow *glfw_window_ptr;
    ImFont *mono_font = nullptr;
    ImFont *menu_font = nullptr;

    std::array<float, 200> full_frametimes = {};
    std::array<float, 200> cpu_frametimes = {};
    daxa_u64 frametime_rotation_index = 0;

    daxa_f32 debug_menu_size{};
    char const *debug_gpu_name{};

    bool needs_saving = false;
    Clock::time_point last_save_time{};

    daxa_u32 conflict_resolution_mode = 0;
    daxa_i32 new_key_id{};
    daxa_i32 limbo_action_index = INVALID_GAME_ACTION;
    daxa_i32 limbo_key_index = GLFW_KEY_LAST + 1;
    daxa_f32 ui_scale = 1.0f;
    bool limbo_is_button = false;

    bool paused = true;
    bool show_settings = false;
    bool show_imgui_demo_window = false;
    bool should_run_startup = true;
    bool should_recreate_voxel_buffers = true;
    bool autosave_override = false;
    bool should_upload_seed_data = true;
    bool should_hotload_shaders = false;
    bool should_regenerate_sky = true;

    bool should_record_task_graph = false;

    bool should_upload_gvox_model = false;
    std::filesystem::path gvox_model_path;
    std::filesystem::path data_directory;

    void rescale_ui();
    void update(daxa_f32 delta_time, daxa_f32 cpu_delta_time);

    void toggle_pause();
    void toggle_debug();
    void toggle_console();

  private:
    void settings_ui();
    void settings_controls_ui();
    void settings_passes_ui();
};
