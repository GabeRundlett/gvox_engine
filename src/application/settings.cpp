#include "ui.hpp"
#include <nlohmann/json.hpp>
#include <base/format.hpp>
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

namespace {
    template <typename K, typename V>
    auto get_or_add(HashMap<K, V> &map, K const &key) -> V & {
        auto *existing = map.get(key);
        if (existing == nullptr) {
            map.set(key, V{});
            existing = map.get(key);
        }
        return *existing;
    }
} // namespace

void AppSettings::add(SettingCategoryId const &category_id, SettingId const &id, SettingEntry const &entry) {
    // TODO: make threadsafe
    auto &self = *s_instance;
    auto &category = get_or_add(self.categories, category_id);
    auto *existing_entry = category.get(id);
    if (existing_entry == nullptr) {
        category.set(id, entry);
    } else {
        existing_entry->factory_default = entry.factory_default;
        existing_entry->config = entry.config;
    }
}

auto AppSettings::get(SettingCategoryId const &category_id, SettingId const &id) -> SettingEntry {
    // TODO: make threadsafe
    // TODO: make lookup faster
    auto &self = *s_instance;
    auto *category = self.categories.get(category_id);
    if (category != nullptr) {
        auto *entry = category->get(id);
        if (entry != nullptr) {
            return *entry;
        }
    }
    return {};
}

void AppSettings::set(SettingCategoryId const &category_id, SettingId const &id, SettingValue const &value) {
    // TODO: make threadsafe
    auto &self = *s_instance;
    auto *category = self.categories.get(category_id);
    if (category != nullptr) {
        auto *entry = category->get(id);
        if (entry != nullptr) {
            entry->data = value;
        }
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

void AppSettings::save(char const *filepath) {
    auto json = nlohmann::json{};

    json["_version"] = 1;

    auto &categories_json = json["categories"];
    for (auto const &cat_slot : categories) {
        auto &category_json = categories_json[cat_slot.key.c_str()];
        for (auto const &entry_slot : cat_slot.value) {
            category_json[entry_slot.key.c_str()] = entry_slot.value;
        }
    }

    json["mouse_sensitivity"] = mouse_sensitivity;
    json["world_seed_str"] = world_seed_str.c_str();

    for (auto const &slot : keybinds) {
        auto str = format("key_%d", slot.key);
        json[str.data] = slot.value;
    }
    for (auto const &slot : mouse_button_binds) {
        auto str = format("mouse_button_%d", slot.key);
        json[str.data] = slot.value;
    }

    auto f = std::ofstream(filepath);
    f << std::setw(4) << json;
}

void AppSettings::load(char const *filepath) {
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
            auto &category = get_or_add(categories, SettingCategoryId{category_id.c_str()});
            for (auto &[entry_id, entry_json] : category_json.items()) {
                SettingEntry entry;
                from_json(entry_json, entry);
                category.set(SettingId{entry_id.c_str()}, entry);
            }
        }
    }

    grab_value("mouse_sensitivity", mouse_sensitivity);
    {
        auto seed_str = std::string{};
        grab_value("world_seed_str", seed_str);
        if (!seed_str.empty()) {
            world_seed_str = seed_str.c_str();
        }
    }

    for (daxa_i32 key_i = 0; key_i < GLFW_KEY_LAST + 1; ++key_i) {
        auto str = format("key_%d", key_i);
        if (json.contains(str.data)) {
            keybinds.set(key_i, json[str.data]);
        }
    }
    for (daxa_i32 mouse_button_i = 0; mouse_button_i < GLFW_MOUSE_BUTTON_LAST + 1; ++mouse_button_i) {
        auto str = format("mouse_button_%d", mouse_button_i);
        if (json.contains(str.data)) {
            mouse_button_binds.set(mouse_button_i, json[str.data]);
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
    keybinds.set(GLFW_KEY_W,            GAME_ACTION_MOVE_FORWARD);
    keybinds.set(GLFW_KEY_A,            GAME_ACTION_MOVE_LEFT);
    keybinds.set(GLFW_KEY_S,            GAME_ACTION_MOVE_BACKWARD);
    keybinds.set(GLFW_KEY_D,            GAME_ACTION_MOVE_RIGHT);
    keybinds.set(GLFW_KEY_R,            GAME_ACTION_RELOAD);
    keybinds.set(GLFW_KEY_F,            GAME_ACTION_TOGGLE_FLY);
    keybinds.set(GLFW_KEY_E,            GAME_ACTION_INTERACT0);
    keybinds.set(GLFW_KEY_Q,            GAME_ACTION_INTERACT1);
    keybinds.set(GLFW_KEY_SPACE,        GAME_ACTION_JUMP);
    keybinds.set(GLFW_KEY_LEFT_CONTROL, GAME_ACTION_CROUCH);
    keybinds.set(GLFW_KEY_LEFT_SHIFT,   GAME_ACTION_SPRINT);
    keybinds.set(GLFW_KEY_LEFT_ALT,     GAME_ACTION_WALK);
    keybinds.set(GLFW_KEY_F5,           GAME_ACTION_CYCLE_VIEW);
    keybinds.set(GLFW_KEY_B,            GAME_ACTION_TOGGLE_BRUSH);

    mouse_button_binds.set(GLFW_MOUSE_BUTTON_1, GAME_ACTION_BRUSH_A);
    mouse_button_binds.set(GLFW_MOUSE_BUTTON_2, GAME_ACTION_BRUSH_B);
    // clang-format on
}
