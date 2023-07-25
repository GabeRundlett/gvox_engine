#pragma once

struct GLFWwindow;
struct ImFont;

#include "app_settings.hpp"
#include <imgui.h>
#include <imgui_stdlib.h>
#include <chrono>
#include <filesystem>
#include <fmt/format.h>

#include <gvox/gvox.h>

#include <shared/input.inl>

struct ChunkHierarchyJobCounters {
    u32 available_threads_queue_top;
    u32 available_threads_queue_bottom;
};

struct AppUi {
    struct Console {
        char input_buffer[256]{};
        std::vector<std::string> items;
        std::vector<const char *> commands;
        std::vector<char *> history;
        int history_pos{-1};
        ImGuiTextFilter filter;
        bool auto_scroll{true};
        bool scroll_to_bottom{false};

        Console();
        ~Console();

        void clear_log();
        void add_log(std::string const &str);
        template <typename... Args>
        void add_log(fmt::format_string<Args...> format_string, Args &&...args) {
            add_log(fmt::vformat(format_string, fmt::make_format_args(std::forward<Args>(args)...)));
        }
        void draw(const char *title, bool *p_open);
        void exec_command(const char *command_line);
        int on_text_edit(ImGuiInputTextCallbackData *data);
    };

    using Clock = std::chrono::high_resolution_clock;

    AppUi(GLFWwindow *glfw_window_ptr);
    ~AppUi();

    AppSettings settings;

    GLFWwindow *glfw_window_ptr;
    ImFont *mono_font = nullptr;
    ImFont *menu_font = nullptr;

    std::array<float, 200> frametimes = {};
    u64 frametime_rotation_index = 0;
    std::string fmt_str;

    f32 debug_menu_size{};
    char const *debug_gpu_name{};
    usize debug_vram_usage{};
    u32 debug_page_count{};
    u32 debug_gpu_heap_usage{};
    f32vec3 debug_player_pos{};
    f32vec3 debug_player_rot{};
    f32vec3 debug_chunk_offset{};
    ChunkHierarchyJobCounters debug_job_counters{};
    u32 debug_total_jobs_ran{};

    struct GpuResourceInfo {
        std::string type;
        std::string name;
        usize size;
    };
    std::vector<GpuResourceInfo> debug_gpu_resource_infos;

    bool needs_saving = false;
    Clock::time_point last_save_time{};
    Console console{};

    u32 conflict_resolution_mode = 0;
    i32 new_key_id{};
    i32 limbo_action_index = GAME_ACTION_LAST + 1;
    i32 limbo_key_index = GLFW_KEY_LAST + 1;
    bool limbo_is_button = false;

    bool paused = true;
    bool show_settings = false;
    bool should_run_startup = true;
    bool should_recreate_voxel_buffers = true;
    bool autosave_override = false;
    bool should_upload_seed_data = true;

    bool should_upload_settings = true;

    bool should_upload_gvox_model = false;
    std::filesystem::path gvox_model_path;
    GvoxRegionRange gvox_region_range{
        .offset = {0, 0, 0},
        .extent = {256, 256, 256},
        // .offset = {-932, -663, -72},
        // .extent = {1932, 1167, 635},
        // .offset = {-70, 108, 150},
        // .extent = {32, 32, 16},
    };

    std::filesystem::path data_directory;

    void rescale_ui();
    void update(f32 delta_time);

    void toggle_pause();
    void toggle_debug();
    void toggle_help();
    void toggle_console();

  private:
    void settings_ui();
    void settings_controls_ui();
};
