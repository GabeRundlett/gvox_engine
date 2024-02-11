#pragma once

#include <map>
#include <filesystem>
#include <variant>

#include <GLFW/glfw3.h>
#include "settings.inl"

enum struct RenderResScl {
    SCL_33_PCT,
    SCL_50_PCT,
    SCL_67_PCT,
    SCL_75_PCT,
    SCL_100_PCT,
};

using SettingCategoryId = std::string;
using SettingId = std::string;

namespace settings {
    struct InputFloat {
        float value;
    };
    struct InputFloat3 {
        daxa_f32vec3 value;
    };
    struct SliderFloat {
        float value;
        float min;
        float max;
    };
} // namespace settings

using SettingType = std::variant<
    settings::InputFloat,
    settings::InputFloat3,
    settings::SliderFloat>;

struct SettingEntry {
    SettingType data;
    SettingType factory_default;
    SettingType user_default;
};

template <typename T>
struct SettingInfo {
    SettingCategoryId category_id = "General";
    SettingId id;
    T factory_default;
};

struct AppSettings {
    std::map<daxa_i32, daxa_i32> keybinds;
    std::map<daxa_i32, daxa_i32> mouse_button_binds;

    // TODO: remove
    daxa_f32 ui_scl;
    daxa_f32 mouse_sensitivity;
    RenderResScl render_res_scl_id;
    std::string world_seed_str;

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
    //

    static inline AppSettings *s_instance = nullptr;

    std::map<SettingCategoryId, std::map<SettingId, SettingEntry>> categories;

    AppSettings();
    ~AppSettings();

    static bool default_entry_ui(SettingId const &id, SettingType &data);

    static void add(SettingCategoryId const &category_id, SettingId const &id, SettingEntry const &entry);
    template <typename T>
    static void add(SettingInfo<T> const &info) {
        add(info.category_id,
            info.id,
            SettingEntry{
                .data = info.factory_default,
                .factory_default = info.factory_default,
                .user_default = info.factory_default,
            });
    }

    static auto get(SettingCategoryId const &category_id, SettingId const &id) -> SettingEntry;

    template <typename T>
    static auto get(SettingCategoryId const &category_id, SettingId const &id) -> T {
        return std::get<T>(get(category_id, id).data);
    }

    void save(std::filesystem::path const &filepath);
    void load(std::filesystem::path const &filepath);
    void clear();
    void reset_default();

    void recompute_sun_direction();
};
