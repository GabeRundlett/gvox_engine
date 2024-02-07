#include "app_ui.hpp"
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <fstream>
#include <numbers>

#include <shared/input.inl>

// sky.absorption_density[0].const_term
// sky.absorption_density[0].exp_scale
// sky.absorption_density[0].exp_term
// sky.absorption_density[0].layer_width
// sky.absorption_density[0].lin_term
// sky.absorption_density[1].const_term
// sky.absorption_density[1].exp_scale
// sky.absorption_density[1].exp_term
// sky.absorption_density[1].layer_width
// sky.absorption_density[1].lin_term
// sky.absorption_extinction.x
// sky.absorption_extinction.y
// sky.absorption_extinction.z
// sky.atmosphere_bottom
// sky.atmosphere_top
// sky.mie_density[0].const_term
// sky.mie_density[0].exp_scale
// sky.mie_density[0].exp_term
// sky.mie_density[0].layer_width
// sky.mie_density[0].lin_term
// sky.mie_density[1].const_term
// sky.mie_density[1].exp_scale
// sky.mie_density[1].exp_term
// sky.mie_density[1].layer_width
// sky.mie_density[1].lin_term
// sky.mie_extinction.x
// sky.mie_extinction.y
// sky.mie_extinction.z
// sky.mie_phase_function_g
// sky.mie_scale_height
// sky.mie_scattering.x
// sky.mie_scattering.y
// sky.mie_scattering.z
// sky.rayleigh_density[0].const_term
// sky.rayleigh_density[0].exp_scale
// sky.rayleigh_density[0].exp_term
// sky.rayleigh_density[0].layer_width
// sky.rayleigh_density[0].lin_term
// sky.rayleigh_density[1].const_term
// sky.rayleigh_density[1].exp_scale
// sky.rayleigh_density[1].exp_term
// sky.rayleigh_density[1].layer_width
// sky.rayleigh_density[1].lin_term
// sky.rayleigh_scale_height
// sky.rayleigh_scattering.x
// sky.rayleigh_scattering.y
// sky.rayleigh_scattering.z

void AppSettings::save(std::filesystem::path const &filepath) {
    auto json = nlohmann::json{};

    json["_version"] = 1;
    json["ui_scl"] = ui_scl;
    json["camera_fov"] = camera_fov;
    json["mouse_sensitivity"] = mouse_sensitivity;
    json["render_res_scl_id"] = render_res_scl_id;
    json["world_seed_str"] = world_seed_str;

    json["show_debug_info"] = show_debug_info;
    json["show_console"] = show_console;
    json["show_help"] = show_help;
    json["autosave"] = autosave;
    json["battery_saving_mode"] = battery_saving_mode;
    json["global_illumination"] = global_illumination;

    json["sky_absorption_density_0_const_term"] = sky.absorption_density[0].const_term;
    json["sky_absorption_density_0_exp_scale"] = sky.absorption_density[0].exp_scale;
    json["sky_absorption_density_0_exp_term"] = sky.absorption_density[0].exp_term;
    json["sky_absorption_density_0_layer_width"] = sky.absorption_density[0].layer_width;
    json["sky_absorption_density_0_lin_term"] = sky.absorption_density[0].lin_term;
    json["sky_absorption_density_1_const_term"] = sky.absorption_density[1].const_term;
    json["sky_absorption_density_1_exp_scale"] = sky.absorption_density[1].exp_scale;
    json["sky_absorption_density_1_exp_term"] = sky.absorption_density[1].exp_term;
    json["sky_absorption_density_1_layer_width"] = sky.absorption_density[1].layer_width;
    json["sky_absorption_density_1_lin_term"] = sky.absorption_density[1].lin_term;
    json["sky_absorption_extinction_x"] = sky.absorption_extinction.x;
    json["sky_absorption_extinction_y"] = sky.absorption_extinction.y;
    json["sky_absorption_extinction_z"] = sky.absorption_extinction.z;
    json["sky_atmosphere_bottom"] = sky.atmosphere_bottom;
    json["sky_atmosphere_top"] = sky.atmosphere_top;
    json["sky_mie_density_0_const_term"] = sky.mie_density[0].const_term;
    json["sky_mie_density_0_exp_scale"] = sky.mie_density[0].exp_scale;
    json["sky_mie_density_0_exp_term"] = sky.mie_density[0].exp_term;
    json["sky_mie_density_0_layer_width"] = sky.mie_density[0].layer_width;
    json["sky_mie_density_0_lin_term"] = sky.mie_density[0].lin_term;
    json["sky_mie_density_1_const_term"] = sky.mie_density[1].const_term;
    json["sky_mie_density_1_exp_scale"] = sky.mie_density[1].exp_scale;
    json["sky_mie_density_1_exp_term"] = sky.mie_density[1].exp_term;
    json["sky_mie_density_1_layer_width"] = sky.mie_density[1].layer_width;
    json["sky_mie_density_1_lin_term"] = sky.mie_density[1].lin_term;
    json["sky_mie_extinction_x"] = sky.mie_extinction.x;
    json["sky_mie_extinction_y"] = sky.mie_extinction.y;
    json["sky_mie_extinction_z"] = sky.mie_extinction.z;
    json["sky_mie_phase_function_g"] = sky.mie_phase_function_g;
    json["sky_mie_scale_height"] = sky.mie_scale_height;
    json["sky_mie_scattering_x"] = sky.mie_scattering.x;
    json["sky_mie_scattering_y"] = sky.mie_scattering.y;
    json["sky_mie_scattering_z"] = sky.mie_scattering.z;
    json["sky_rayleigh_density_0_const_term"] = sky.rayleigh_density[0].const_term;
    json["sky_rayleigh_density_0_exp_scale"] = sky.rayleigh_density[0].exp_scale;
    json["sky_rayleigh_density_0_exp_term"] = sky.rayleigh_density[0].exp_term;
    json["sky_rayleigh_density_0_layer_width"] = sky.rayleigh_density[0].layer_width;
    json["sky_rayleigh_density_0_lin_term"] = sky.rayleigh_density[0].lin_term;
    json["sky_rayleigh_density_1_const_term"] = sky.rayleigh_density[1].const_term;
    json["sky_rayleigh_density_1_exp_scale"] = sky.rayleigh_density[1].exp_scale;
    json["sky_rayleigh_density_1_exp_term"] = sky.rayleigh_density[1].exp_term;
    json["sky_rayleigh_density_1_layer_width"] = sky.rayleigh_density[1].layer_width;
    json["sky_rayleigh_density_1_lin_term"] = sky.rayleigh_density[1].lin_term;
    json["sky_rayleigh_scale_height"] = sky.rayleigh_scale_height;
    json["sky_rayleigh_scattering_x"] = sky.rayleigh_scattering.x;
    json["sky_rayleigh_scattering_y"] = sky.rayleigh_scattering.y;
    json["sky_rayleigh_scattering_z"] = sky.rayleigh_scattering.z;

    json["sun_angle_x"] = sun_angle.x;
    json["sun_angle_y"] = sun_angle.y;
    json["sun_angular_radius"] = sun_angular_radius;

    json["auto_exposure_histogram_clip_low"] = auto_exposure.histogram_clip_low;
    json["auto_exposure_histogram_clip_high"] = auto_exposure.histogram_clip_high;
    json["auto_exposure_speed"] = auto_exposure.speed;
    json["auto_exposure_ev_shift"] = auto_exposure.ev_shift;

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
    grab_value("render_res_scl_id", render_res_scl_id);
    grab_value("world_seed_str", world_seed_str);

    grab_value("show_debug_info", show_debug_info);
    grab_value("show_console", show_console);
    grab_value("show_help", show_help);
    grab_value("autosave", autosave);
    grab_value("battery_saving_mode", battery_saving_mode);
    grab_value("global_illumination", global_illumination);

    grab_value("sky_absorption_density_0_const_term", sky.absorption_density[0].const_term);
    grab_value("sky_absorption_density_0_exp_scale", sky.absorption_density[0].exp_scale);
    grab_value("sky_absorption_density_0_exp_term", sky.absorption_density[0].exp_term);
    grab_value("sky_absorption_density_0_layer_width", sky.absorption_density[0].layer_width);
    grab_value("sky_absorption_density_0_lin_term", sky.absorption_density[0].lin_term);
    grab_value("sky_absorption_density_1_const_term", sky.absorption_density[1].const_term);
    grab_value("sky_absorption_density_1_exp_scale", sky.absorption_density[1].exp_scale);
    grab_value("sky_absorption_density_1_exp_term", sky.absorption_density[1].exp_term);
    grab_value("sky_absorption_density_1_layer_width", sky.absorption_density[1].layer_width);
    grab_value("sky_absorption_density_1_lin_term", sky.absorption_density[1].lin_term);
    grab_value("sky_absorption_extinction_x", sky.absorption_extinction.x);
    grab_value("sky_absorption_extinction_y", sky.absorption_extinction.y);
    grab_value("sky_absorption_extinction_z", sky.absorption_extinction.z);
    grab_value("sky_atmosphere_bottom", sky.atmosphere_bottom);
    grab_value("sky_atmosphere_top", sky.atmosphere_top);
    grab_value("sky_mie_density_0_const_term", sky.mie_density[0].const_term);
    grab_value("sky_mie_density_0_exp_scale", sky.mie_density[0].exp_scale);
    grab_value("sky_mie_density_0_exp_term", sky.mie_density[0].exp_term);
    grab_value("sky_mie_density_0_layer_width", sky.mie_density[0].layer_width);
    grab_value("sky_mie_density_0_lin_term", sky.mie_density[0].lin_term);
    grab_value("sky_mie_density_1_const_term", sky.mie_density[1].const_term);
    grab_value("sky_mie_density_1_exp_scale", sky.mie_density[1].exp_scale);
    grab_value("sky_mie_density_1_exp_term", sky.mie_density[1].exp_term);
    grab_value("sky_mie_density_1_layer_width", sky.mie_density[1].layer_width);
    grab_value("sky_mie_density_1_lin_term", sky.mie_density[1].lin_term);
    grab_value("sky_mie_extinction_x", sky.mie_extinction.x);
    grab_value("sky_mie_extinction_y", sky.mie_extinction.y);
    grab_value("sky_mie_extinction_z", sky.mie_extinction.z);
    grab_value("sky_mie_phase_function_g", sky.mie_phase_function_g);
    grab_value("sky_mie_scale_height", sky.mie_scale_height);
    grab_value("sky_mie_scattering_x", sky.mie_scattering.x);
    grab_value("sky_mie_scattering_y", sky.mie_scattering.y);
    grab_value("sky_mie_scattering_z", sky.mie_scattering.z);
    grab_value("sky_rayleigh_density_0_const_term", sky.rayleigh_density[0].const_term);
    grab_value("sky_rayleigh_density_0_exp_scale", sky.rayleigh_density[0].exp_scale);
    grab_value("sky_rayleigh_density_0_exp_term", sky.rayleigh_density[0].exp_term);
    grab_value("sky_rayleigh_density_0_layer_width", sky.rayleigh_density[0].layer_width);
    grab_value("sky_rayleigh_density_0_lin_term", sky.rayleigh_density[0].lin_term);
    grab_value("sky_rayleigh_density_1_const_term", sky.rayleigh_density[1].const_term);
    grab_value("sky_rayleigh_density_1_exp_scale", sky.rayleigh_density[1].exp_scale);
    grab_value("sky_rayleigh_density_1_exp_term", sky.rayleigh_density[1].exp_term);
    grab_value("sky_rayleigh_density_1_layer_width", sky.rayleigh_density[1].layer_width);
    grab_value("sky_rayleigh_density_1_lin_term", sky.rayleigh_density[1].lin_term);
    grab_value("sky_rayleigh_scale_height", sky.rayleigh_scale_height);
    grab_value("sky_rayleigh_scattering_x", sky.rayleigh_scattering.x);
    grab_value("sky_rayleigh_scattering_y", sky.rayleigh_scattering.y);
    grab_value("sky_rayleigh_scattering_z", sky.rayleigh_scattering.z);

    grab_value("sun_angle_x", sun_angle.x);
    grab_value("sun_angle_y", sun_angle.y);
    grab_value("sun_angular_radius", sun_angular_radius);
    recompute_sun_direction();

    grab_value("auto_exposure_histogram_clip_low", auto_exposure.histogram_clip_low);
    grab_value("auto_exposure_histogram_clip_high", auto_exposure.histogram_clip_high);
    grab_value("auto_exposure_speed", auto_exposure.speed);
    grab_value("auto_exposure_ev_shift", auto_exposure.ev_shift);

    for (daxa_i32 key_i = 0; key_i < GLFW_KEY_LAST + 1; ++key_i) {
        auto str = fmt::format("key_{}", key_i);
        if (json.contains(str)) {
            keybinds[key_i] = json[str];
        }
    }
    for (daxa_i32 mouse_button_i = 0; mouse_button_i < GLFW_MOUSE_BUTTON_LAST + 1; ++mouse_button_i) {
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
    render_res_scl_id = RenderResScl::SCL_100_PCT;
    world_seed_str = "gvox";

    show_debug_info = false;
    show_console = false;
    show_help = false;
    autosave = true;
    battery_saving_mode = false;
    global_illumination = true;

    keybinds.clear();
    mouse_button_binds.clear();

    sky = {};
    recompute_sun_direction();
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

    // Sky
    sky.absorption_density[0] = {
        .const_term = -0.6666600108146667f,
        .exp_scale = 0.0f,
        .exp_term = 0.0f,
        .layer_width = 25.0f,
        .lin_term = 0.06666599959135056f,
    };
    sky.absorption_density[1] = {
        .const_term = 2.6666600704193115f,
        .exp_scale = 0.0f,
        .exp_term = 0.0f,
        .layer_width = 0.0f,
        .lin_term = -0.06666599959135056f,
    };
    sky.absorption_extinction = {
        .x = 0.00229072f,
        .y = 0.00214036f,
        .z = 0.0f,
    };
    sky.atmosphere_bottom = 6360.0f;
    sky.atmosphere_top = 6460.0f;
    sky.mie_phase_function_g = 0.800000011920929f,
    sky.mie_scale_height = 1.2000000476837158f,
    sky.mie_density[0] = {
        .const_term = 0.0f,
        .exp_scale = 0.0f,
        .exp_term = 0.0f,
        .layer_width = 0.0f,
        .lin_term = 0.0f,
    };
    sky.mie_density[1] = {
        .const_term = 0.0f,
        .exp_scale = -1.0f / sky.mie_scale_height,
        .exp_term = 1.0f,
        .layer_width = 0.0f,
        .lin_term = 0.0f,
    };
    sky.mie_extinction = {
        .x = 0.00443999981507659f,
        .y = 0.00443999981507659f,
        .z = 0.00443999981507659f,
    };
    sky.mie_scattering = {
        .x = 0.003996000159531832f,
        .y = 0.003996000159531832f,
        .z = 0.003996000159531832f,
    };
    sky.rayleigh_scale_height = 8.696f;
    sky.rayleigh_density[0] = {
        .const_term = 0.0f,
        .exp_scale = 0.0f,
        .exp_term = 0.0f,
        .layer_width = 0.0f,
        .lin_term = 0.0f,
    };
    sky.rayleigh_density[1] = {
        .const_term = 0.0f,
        .exp_scale = -1.0f / sky.rayleigh_scale_height,
        .exp_term = 1.0f,
        .layer_width = 0.0f,
        .lin_term = 0.0f,
    };
    sky.rayleigh_scattering = {
        .x = 0.006604931f,
        .y = 0.013344918f,
        .z = 0.029412623f,
    };

    sun_angle = {210.0f, 25.0f};
    sun_angular_radius = 0.25f;
    recompute_sun_direction();

    auto_exposure.histogram_clip_low = 0.1f;
    auto_exposure.histogram_clip_high = 0.1f;
    auto_exposure.speed = 3.0f;
    auto_exposure.ev_shift = -2.5f;
}

void AppSettings::recompute_sun_direction() {
    auto radians = [](float x) -> float {
        return x * std::numbers::pi_v<float> / 180.0f;
    };
    sky.sun_direction = {
        daxa_f32(std::cos(radians(sun_angle.x)) * std::sin(radians(sun_angle.y))),
        daxa_f32(std::sin(radians(sun_angle.x)) * std::sin(radians(sun_angle.y))),
        daxa_f32(std::cos(radians(sun_angle.y))),
    };

    sky.sun_angular_radius_cos = std::cos(radians(sun_angular_radius));
}
