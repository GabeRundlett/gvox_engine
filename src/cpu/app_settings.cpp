#include "app_ui.hpp"
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <fstream>

#include <shared/input.inl>

void AppSettings::save(std::filesystem::path const &filepath) {
    auto json = nlohmann::json{};

    json["_version"] = 1;
    json["ui_scl"] = ui_scl;
    json["camera_fov"] = camera_fov;
    json["mouse_sensitivity"] = mouse_sensitivity;
    json["render_res_scl"] = render_res_scl;
    // json["log2_chunks_per_axis"] = log2_chunks_per_axis;
    // json["gpu_heap_size"] = gpu_heap_size;
    json["world_seed_str"] = world_seed_str;

    json["show_debug_info"] = show_debug_info;
    json["show_console"] = show_console;
    json["show_help"] = show_help;
    json["autosave"] = autosave;
    json["battery_saving_mode"] = battery_saving_mode;

    for (auto [key_i, action_i] : keybinds) {
        auto str = fmt::format("key_{}", key_i);
        json[str] = action_i;
    }
    for (auto [mouse_button_i, action_i] : mouse_button_binds) {
        auto str = fmt::format("mouse_button_{}", mouse_button_i);
        json[str] = action_i;
    }

    auto f = std::ofstream(filepath);
    f << std::setw(4) << json;
}

void AppSettings::load(std::filesystem::path const &filepath) {
    clear();

    auto json = nlohmann::json::parse(std::ifstream(filepath));

    auto grab_value = [&json](auto str, auto &val) {
        if (json.contains(str)) {
            val = json[str];
        }
    };

    grab_value("ui_scl", ui_scl);
    grab_value("camera_fov", camera_fov);
    grab_value("mouse_sensitivity", mouse_sensitivity);
    grab_value("render_res_scl", render_res_scl);
    // grab_value("log2_chunks_per_axis", log2_chunks_per_axis);
    // grab_value("gpu_heap_size", gpu_heap_size);
    grab_value("world_seed_str", world_seed_str);

    grab_value("show_debug_info", show_debug_info);
    grab_value("show_console", show_console);
    grab_value("show_help", show_help);
    grab_value("autosave", autosave);
    grab_value("battery_saving_mode", battery_saving_mode);

    for (i32 key_i = 0; key_i < GLFW_KEY_LAST + 1; ++key_i) {
        auto str = fmt::format("key_{}", key_i);
        if (json.contains(str)) {
            keybinds[key_i] = json[str];
        }
    }
    for (i32 mouse_button_i = 0; mouse_button_i < GLFW_MOUSE_BUTTON_LAST + 1; ++mouse_button_i) {
        auto str = fmt::format("mouse_button_{}", mouse_button_i);
        if (json.contains(str)) {
            mouse_button_binds[mouse_button_i] = json[str];
        }
    }
}

void AppSettings::clear() {
    ui_scl = 1.0f;

    camera_fov = 90.0f;
    mouse_sensitivity = 1.0f;
    render_res_scl = 1.0f;
    log2_chunks_per_axis = 5;
    gpu_heap_size = 1u << 30;
    world_seed_str = "gvox";

    show_debug_info = false;
    show_console = false;
    show_help = false;
    autosave = true;
    battery_saving_mode = false;

    keybinds.clear();
    mouse_button_binds.clear();
}

void AppSettings::reset_default() {
    clear();

    // clang-format off
    keybinds[GLFW_KEY_W]             = GAME_ACTION_MOVE_FORWARD;
    keybinds[GLFW_KEY_A]             = GAME_ACTION_MOVE_LEFT;
    keybinds[GLFW_KEY_S]             = GAME_ACTION_MOVE_BACKWARD;
    keybinds[GLFW_KEY_D]             = GAME_ACTION_MOVE_RIGHT;
    keybinds[GLFW_KEY_R]             = GAME_ACTION_RELOAD;
    keybinds[GLFW_KEY_F]             = GAME_ACTION_TOGGLE_FLY;
    keybinds[GLFW_KEY_E]             = GAME_ACTION_INTERACT0;
    keybinds[GLFW_KEY_Q]             = GAME_ACTION_INTERACT1;
    keybinds[GLFW_KEY_SPACE]         = GAME_ACTION_JUMP;
    keybinds[GLFW_KEY_LEFT_CONTROL]  = GAME_ACTION_CROUCH;
    keybinds[GLFW_KEY_LEFT_SHIFT]    = GAME_ACTION_SPRINT;
    keybinds[GLFW_KEY_LEFT_ALT]      = GAME_ACTION_WALK;
    keybinds[GLFW_KEY_F5]            = GAME_ACTION_CYCLE_VIEW;
    keybinds[GLFW_KEY_B]             = GAME_ACTION_TOGGLE_BRUSH;

    mouse_button_binds[GLFW_MOUSE_BUTTON_1] = GAME_ACTION_BRUSH_A;
    mouse_button_binds[GLFW_MOUSE_BUTTON_2] = GAME_ACTION_BRUSH_B;
    // clang-format on
}
