#pragma once

#include <daxa/types.hpp>
using namespace daxa::types;

#include <map>
#include <filesystem>

#include <shared/shared.inl>
#include <GLFW/glfw3.h>

struct AppSettings {
    std::map<i32, i32> keybinds;
    std::map<i32, i32> mouse_button_binds;
    f32 ui_scl;
    f32 camera_fov;
    f32 mouse_sensitivity;
    f32 render_res_scl;
    u32 log2_chunks_per_axis;

    bool show_debug_info;
    bool show_console;
    bool show_help;
    bool autosave;

    void save(std::filesystem::path const &filepath);
    void load(std::filesystem::path const &filepath);
    void clear();
    void reset_default();
};
