#pragma once

#include <map>
#include <filesystem>
#include <variant>

#include <GLFW/glfw3.h>
#include "settings.inl"

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
    struct Checkbox {
        bool value;
    };
} // namespace settings

using SettingValue = std::variant<
    settings::InputFloat,
    settings::InputFloat3,
    settings::SliderFloat,
    settings::Checkbox>;

struct SettingEntry {
    SettingValue data;
    SettingValue factory_default;
    SettingValue user_default;
};

template <typename T>
struct SettingInfo {
    SettingCategoryId category_id = "General";
    SettingId id;
    T factory_default;
};

struct AppSettings {
    // TODO: remove these explicit settings in favor of settings registry
    std::map<daxa_i32, daxa_i32> keybinds;
    std::map<daxa_i32, daxa_i32> mouse_button_binds;

    daxa_f32 mouse_sensitivity;
    std::string world_seed_str;

    static inline AppSettings *s_instance = nullptr;

    std::map<SettingCategoryId, std::map<SettingId, SettingEntry>> categories;

    AppSettings();
    ~AppSettings();

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

    static void set(SettingCategoryId const &category_id, SettingId const &id, SettingValue const &value);

    static auto get(SettingCategoryId const &category_id, SettingId const &id) -> SettingEntry;
    template <typename T>
    static auto get(SettingCategoryId const &category_id, SettingId const &id) -> T {
        return std::get<T>(get(category_id, id).data);
    }

    void save(std::filesystem::path const &filepath);
    void load(std::filesystem::path const &filepath);
    void clear();
    void reset_default();
};
