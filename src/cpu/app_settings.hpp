#pragma once

#include <map>
#include <filesystem>

#include <GLFW/glfw3.h>
#include <shared/settings.inl>

struct AppSettings {
    std::map<daxa_i32, daxa_i32> keybinds;
    std::map<daxa_i32, daxa_i32> mouse_button_binds;
    daxa_f32 ui_scl;
    daxa_f32 camera_fov;
    daxa_f32 mouse_sensitivity;
    daxa_f32 render_res_scl;
    std::string world_seed_str;

    SkySettings sky;
    daxa_f32vec2 sun_angle;
    daxa_f32 sun_angular_radius;

    bool show_debug_info;
    bool show_console;
    bool show_help;
    bool autosave;
    bool battery_saving_mode;

    void save(std::filesystem::path const &filepath);
    void load(std::filesystem::path const &filepath);
    void clear();
    void reset_default();

    void recompute_sun_direction();
};
