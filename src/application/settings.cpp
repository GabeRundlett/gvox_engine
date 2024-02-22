#include "ui.hpp"
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <fstream>
#include <numbers>

#include "input.inl"

AppSettings::AppSettings() {
    // assert(s_instance == nullptr);
    s_instance = this;
}
AppSettings::~AppSettings() {
    s_instance = nullptr;
}

void AppSettings::add(SettingCategoryId const &category_id, SettingId const &id, SettingEntry const &entry) {
    // TODO: make threadsafe
    auto &self = *s_instance;
    auto &category = self.categories[category_id];
    auto entry_iter = category.find(id);
    if (entry_iter == category.end()) {
        category.insert({id, entry});
    } else {
        entry_iter->second.factory_default = entry.factory_default;
        entry_iter->second.config = entry.config;
    }
}

auto AppSettings::get(SettingCategoryId const &category_id, SettingId const &id) -> SettingEntry {
    // TODO: make threadsafe
    // TODO: make lookup faster
    auto &self = *s_instance;
    auto &category = self.categories[category_id];
    auto entry_iter = category.find(id);
    if (entry_iter != category.end()) {
        return entry_iter->second;
    }
    return {};
}

void AppSettings::set(SettingCategoryId const &category_id, SettingId const &id, SettingValue const &value) {
    // TODO: make threadsafe
    auto &self = *s_instance;
    auto &category = self.categories[category_id];
    auto entry_iter = category.find(id);
    if (entry_iter != category.end()) {
        entry_iter->second.data = value;
    }
}

namespace settings {
    void to_json(nlohmann::json &j, InputFloat const &x) {
        j = nlohmann::json{{"value", x.value}};
    }
    void from_json(const nlohmann::json &j, InputFloat &x) {
        j.at("value").get_to(x.value);
    }

    void to_json(nlohmann::json &j, InputFloat3 const &x) {
        j = nlohmann::json{{"x", x.value.x}, {"y", x.value.y}, {"z", x.value.z}};
    }
    void from_json(const nlohmann::json &j, InputFloat3 &x) {
        j.at("x").get_to(x.value.x);
        j.at("y").get_to(x.value.y);
        j.at("z").get_to(x.value.z);
    }

    void to_json(nlohmann::json &j, SliderFloat const &x) {
        j = nlohmann::json{{"value", x.value}, {"min", x.min}, {"max", x.max}};
    }
    void from_json(const nlohmann::json &j, SliderFloat &x) {
        j.at("value").get_to(x.value);
        j.at("min").get_to(x.min);
        j.at("max").get_to(x.max);
    }

    void to_json(nlohmann::json &j, Checkbox const &x) {
        j = nlohmann::json{{"value", x.value}};
    }
    void from_json(const nlohmann::json &j, Checkbox &x) {
        j.at("value").get_to(x.value);
    }

    void to_json(nlohmann::json &j, ComboBox const &x) {
        j = nlohmann::json{{"value", x.value}};
    }
    void from_json(const nlohmann::json &j, ComboBox &x) {
        j.at("value").get_to(x.value);
    }
} // namespace settings

#include <typeinfo>

void to_json(nlohmann::json &j, SettingValue const &x) {
    j = nlohmann::json{};
    std::visit(
        [&](auto &&entry_data) {
            j["type"] = typeid(entry_data).name();
            j["setting"] = entry_data;
        },
        x);
}

namespace {
    template <typename... Ts>
    auto make_type_name_table(std::variant<Ts...> const &) {
        return std::map<std::string, std::variant<Ts...>>{
            {std::string{typeid(Ts).name()}, Ts{}}...};
    }
} // namespace

static const std::map<std::string, SettingValue> setting_type_name_table = make_type_name_table(SettingValue{});

void from_json(const nlohmann::json &j, SettingValue &x) {
    x = setting_type_name_table.at(j["type"]);
    std::visit([&](auto &entry_data) { j["setting"].get_to(entry_data); }, x);
}

void to_json(nlohmann::json &j, SettingEntry const &x) {
    j = nlohmann::json{};
    to_json(j["data"], x.data);
    to_json(j["user_default"], x.user_default);
}
void from_json(const nlohmann::json &j, SettingEntry &x) {
    from_json(j["data"], x.data);
    from_json(j["user_default"], x.user_default);
}

void AppSettings::save(std::filesystem::path const &filepath) {
    auto json = nlohmann::json{};

    json["_version"] = 1;

    auto &categories_json = json["categories"];
    for (auto const &[cat_id, category] : categories) {
        auto &category_json = categories_json[cat_id];
        for (auto const &[entry_key, entry] : category) {
            category_json[entry_key] = entry;
        }
    }

    json["mouse_sensitivity"] = mouse_sensitivity;
    json["world_seed_str"] = world_seed_str;

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

    {
        auto categories_json = json["categories"];
        for (auto &[category_id, category_json] : categories_json.items()) {
            auto &category = categories[category_id];
            for (auto &[entry_id, entry_json] : category_json.items()) {
                SettingEntry entry;
                from_json(entry_json, entry);
                category.insert({entry_id, entry});
            }
        }
    }

    grab_value("mouse_sensitivity", mouse_sensitivity);
    grab_value("world_seed_str", world_seed_str);

    auto load_brush_settings = [&grab_value](std::string const &brush_name, BrushSettings &brush_settings) {
        grab_value(brush_name + ".flags", brush_settings.flags);
        grab_value(brush_name + ".radius", brush_settings.radius);
    };

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
    mouse_sensitivity = 1.0f;
    world_seed_str = "gvox";

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
