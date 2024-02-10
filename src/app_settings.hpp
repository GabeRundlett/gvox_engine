#pragma once

#include <map>
#include <filesystem>

#include <GLFW/glfw3.h>
#include <shared/settings.inl>

enum struct RenderResScl {
    SCL_33_PCT,
    SCL_50_PCT,
    SCL_67_PCT,
    SCL_75_PCT,
    SCL_100_PCT,
};

struct AppSettings {
    std::map<daxa_i32, daxa_i32> keybinds;
    std::map<daxa_i32, daxa_i32> mouse_button_binds;
    daxa_f32 ui_scl;
    daxa_f32 camera_fov;
    daxa_f32 mouse_sensitivity;
    RenderResScl render_res_scl_id;
    std::string world_seed_str;

    SkySettings sky;
    daxa_f32vec2 sun_angle;
    daxa_f32 sun_angular_radius;

    RendererSettings renderer;

    BrushSettings world_brush_settings;
    BrushSettings brush_a_settings;
    BrushSettings brush_b_settings;

    bool show_debug_info;
    bool show_console;
    bool show_help;
    bool autosave;
    bool battery_saving_mode;
    bool global_illumination;

    void save(std::filesystem::path const &filepath);
    void load(std::filesystem::path const &filepath);
    void clear();
    void reset_default();

    void recompute_sun_direction();
};
