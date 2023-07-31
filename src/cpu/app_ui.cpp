#include "app_ui.hpp"

#include <imgui_stdlib.h>
#include <imgui_impl_glfw.h>
#include <fmt/format.h>
#include <sago/platform_folders.h>
#include <nfd.h>

#include <vector>
#include <iostream>

#include <shared/voxel_malloc.inl>

using namespace std::literals;

static constexpr std::array control_strings = {
    "Move Forward",
    "Strafe Left",
    "Move Backward",
    "Strafe Right",
    "Reload Chunks",
    "Toggle Fly",
    "Interact 0",
    "Interact 1",
    "Jump",
    "Crouch",
    "Sprint",
    "Walk",
    "Change Camera",
    "Toggle Brush Placement",
    "Brush A",
    "Brush B",
};

static constexpr std::array conflict_resolution_strings = {
    "swap",
    "remove old",
    "cancel",
};

inline auto get_key_string(i32 glfw_key_id) -> char const * {
    const auto *result = glfwGetKeyName(glfw_key_id, 0);
    if (result == nullptr) {
        switch (glfw_key_id) {
        case GLFW_KEY_SPACE: result = "space"; break;
        case GLFW_KEY_LEFT_SHIFT: result = "left shift"; break;
        case GLFW_KEY_LEFT_CONTROL: result = "left ctrl"; break;
        case GLFW_KEY_LEFT_ALT: result = "left alt"; break;
        case GLFW_KEY_RIGHT_SHIFT: result = "right shift"; break;
        case GLFW_KEY_RIGHT_CONTROL: result = "right ctrl"; break;
        case GLFW_KEY_RIGHT_ALT: result = "right alt"; break;
        case GLFW_KEY_UP: result = "arrow up"; break;
        case GLFW_KEY_DOWN: result = "arrow down"; break;
        case GLFW_KEY_LEFT: result = "arrow left"; break;
        case GLFW_KEY_RIGHT: result = "arrow right"; break;
        case GLFW_KEY_F1: result = "f1"; break;
        case GLFW_KEY_F2: result = "f2"; break;
        case GLFW_KEY_F3: result = "f3"; break;
        case GLFW_KEY_F4: result = "f4"; break;
        case GLFW_KEY_F5: result = "f5"; break;
        case GLFW_KEY_F6: result = "f6"; break;
        case GLFW_KEY_F7: result = "f7"; break;
        case GLFW_KEY_F8: result = "f8"; break;
        case GLFW_KEY_F9: result = "f9"; break;
        case GLFW_KEY_F10: result = "f10"; break;
        case GLFW_KEY_F11: result = "f11"; break;
        case GLFW_KEY_F12: result = "f12"; break;
        default: result = "unknown key"; break;
        }
    }
    return result;
}

inline auto get_button_string(i32 glfw_key_id) -> char const * {
    switch (glfw_key_id) {
    case GLFW_MOUSE_BUTTON_1: return "left mouse button";
    case GLFW_MOUSE_BUTTON_2: return "right mouse button";
    case GLFW_MOUSE_BUTTON_3: return "middle mouse button";
    case GLFW_MOUSE_BUTTON_4: return "mouse button 4";
    case GLFW_MOUSE_BUTTON_5: return "mouse button 5";
    default: return "unknown button";
    }
}

static auto Stricmp(const char *s1, const char *s2) -> int {
    int d = 0;
    while ((d = toupper(*s2) - toupper(*s1)) == 0 && (*s1 != 0)) {
        s1++;
        s2++;
    }
    return d;
}

static auto Strnicmp(const char *s1, const char *s2, int n) -> int {
    int d = 0;
    while (n > 0 && (d = toupper(*s2) - toupper(*s1)) == 0 && (*s1 != 0)) {
        s1++;
        s2++;
        n--;
    }
    return d;
}

static auto Strdup(const char *s) -> char * {
    IM_ASSERT(s);
    size_t const len = strlen(s) + 1;
    void *buf = malloc(len);
    IM_ASSERT(buf);
    return static_cast<char *>(memcpy(buf, static_cast<const void *>(s), len));
}

static void Strtrim(char *s) {
    char *str_end = s + strlen(s);
    while (str_end > s && str_end[-1] == ' ') {
        str_end--;
    }
    *str_end = 0;
}

AppUi::Console::Console() {
    s_instance = this;
    clear_log();
    memset(input_buffer, 0, sizeof(input_buffer));
}

AppUi::Console::~Console() {
    if (s_instance == this) {
        s_instance = nullptr;
    }
    clear_log();
    for (auto &i : history) {
        free(i);
    }
}

void AppUi::Console::clear_log() {
    items.clear();
}

void AppUi::Console::add_log(std::string const &str) {
    items.push_back(str);
    std::cout << str << std::endl;
}

void AppUi::Console::draw(const char *title, bool *p_open) {
    ImGui::SetNextWindowSize(ImVec2(520, 600), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(title, p_open)) {
        ImGui::End();
        return;
    }
    if (ImGui::BeginPopupContextItem()) {
        if (ImGui::MenuItem("Close Console")) {
            *p_open = false;
        }
        ImGui::EndPopup();
    }
    // if (ImGui::SmallButton("Add Debug Text")) {
    //     add_log("{} some text", items.size());
    //     add_log("some more text");
    //     add_log("display very important message here!");
    // }
    // ImGui::SameLine();
    // if (ImGui::SmallButton("Add Debug Error")) {
    //     add_log("[error] something went wrong");
    // }
    // ImGui::SameLine();
    if (ImGui::SmallButton("Clear")) {
        clear_log();
    }
    ImGui::SameLine();
    bool const copy_to_clipboard = ImGui::SmallButton("Copy");
    ImGui::Separator();
    if (ImGui::BeginPopup("Options")) {
        ImGui::Checkbox("Auto-scroll", &auto_scroll);
        ImGui::EndPopup();
    }
    if (ImGui::Button("Options")) {
        ImGui::OpenPopup("Options");
    }
    ImGui::SameLine();
    filter.Draw(R"(Filter ("incl,-excl") ("error"))", 180);
    ImGui::Separator();
    const float footer_height_to_reserve = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
    ImGui::BeginChild("ScrollingRegion", ImVec2(0, -footer_height_to_reserve), false, ImGuiWindowFlags_HorizontalScrollbar);
    if (ImGui::BeginPopupContextWindow()) {
        if (ImGui::Selectable("Clear")) {
            clear_log();
        }
        ImGui::EndPopup();
    }
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 1));
    if (copy_to_clipboard) {
        ImGui::LogToClipboard();
    }
    for (auto const &item : items) {
        if (!filter.PassFilter(item.c_str())) {
            continue;
        }
        ImVec4 color;
        bool has_color = false;
        if (strstr(item.c_str(), "[error]") != nullptr) {
            color = ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
            has_color = true;
        } else if (strncmp(item.c_str(), "# ", 2) == 0) {
            color = ImVec4(1.0f, 0.8f, 0.6f, 1.0f);
            has_color = true;
        }
        if (has_color) {
            ImGui::PushStyleColor(ImGuiCol_Text, color);
        }
        ImGui::TextUnformatted(item.c_str());
        if (has_color) {
            ImGui::PopStyleColor();
        }
    }
    if (copy_to_clipboard) {
        ImGui::LogFinish();
    }
    if (scroll_to_bottom || (auto_scroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())) {
        ImGui::SetScrollHereY(1.0f);
    }
    scroll_to_bottom = false;
    ImGui::PopStyleVar();
    ImGui::EndChild();
    ImGui::Separator();
    bool reclaim_focus = false;
    ImGuiInputTextFlags const input_text_flags = ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackCompletion | ImGuiInputTextFlags_CallbackHistory;
    if (ImGui::InputText(
            "Input", input_buffer, IM_ARRAYSIZE(input_buffer), input_text_flags, [](ImGuiInputTextCallbackData *data) -> int {
                auto *user_console = static_cast<Console *>(data->UserData);
                return user_console->on_text_edit(data);
            },
            static_cast<void *>(this))) {
        char *s = input_buffer;
        Strtrim(s);
        if (s[0] != 0) {
            exec_command(s);
        }
        s[0] = '\0';
        reclaim_focus = true;
    }
    ImGui::SetItemDefaultFocus();
    if (reclaim_focus) {
        ImGui::SetKeyboardFocusHere(-1);
    }
    ImGui::End();
}

void AppUi::Console::exec_command(const char *command_line) {
    add_log("# {}\n", command_line);
    history_pos = -1;
    for (i32 i = static_cast<i32>(history.size()) - 1; i >= 0; i--) {
        if (Stricmp(history[static_cast<usize>(i)], command_line) == 0) {
            free(history[static_cast<usize>(i)]);
            history.erase(history.begin() + i);
            break;
        }
    }
    history.push_back(Strdup(command_line));
    add_log("Unknown command: '{}'\n", command_line);
    scroll_to_bottom = true;
}

auto AppUi::Console::on_text_edit(ImGuiInputTextCallbackData *data) -> int {
    switch (data->EventFlag) {
    case ImGuiInputTextFlags_CallbackCompletion: {
        const char *word_end = data->Buf + data->CursorPos;
        const char *word_start = word_end;
        while (word_start > data->Buf) {
            const char c = word_start[-1];
            if (c == ' ' || c == '\t' || c == ',' || c == ';') {
                break;
            }
            word_start--;
        }
        ImVector<const char *> candidates;
        for (auto &command : commands) {
            if (Strnicmp(command, word_start, static_cast<i32>(word_end - word_start)) == 0) {
                candidates.push_back(command);
            }
        }
        if (candidates.empty()) {
            add_log("No match for \"{}\"!\n", /* (int)(word_end - word_start), */ word_start);
        } else if (candidates.size() == 1) {
            data->DeleteChars(static_cast<i32>(word_start - data->Buf), static_cast<i32>(word_end - word_start));
            data->InsertChars(data->CursorPos, candidates[0]);
            data->InsertChars(data->CursorPos, " ");
        } else {
            int match_len = static_cast<i32>(word_end - word_start);
            for (;;) {
                int c = 0;
                bool all_candidates_matches = true;
                for (int i = 0; i < candidates.size() && all_candidates_matches; i++) {
                    if (i == 0) {
                        c = toupper(candidates[i][match_len]);
                    } else if (c == 0 || c != toupper(candidates[i][match_len])) {
                        all_candidates_matches = false;
                    }
                }
                if (!all_candidates_matches) {
                    break;
                }
                match_len++;
            }
            if (match_len > 0) {
                data->DeleteChars(static_cast<i32>(word_start - data->Buf), static_cast<i32>(word_end - word_start));
                data->InsertChars(data->CursorPos, candidates[0], candidates[0] + match_len);
            }
            add_log("Possible matches:\n");
            for (auto &candidate : candidates) {
                add_log("- {}\n", candidate);
            }
        }
        break;
    }
    case ImGuiInputTextFlags_CallbackHistory: {
        const int prev_history_pos = history_pos;
        if (data->EventKey == ImGuiKey_UpArrow) {
            if (history_pos == -1) {
                history_pos = static_cast<i32>(history.size()) - 1;
            } else if (history_pos > 0) {
                history_pos--;
            }
        } else if (data->EventKey == ImGuiKey_DownArrow) {
            if (history_pos != -1) {
                if (static_cast<usize>(++history_pos) >= history.size()) {
                    history_pos = -1;
                }
            }
        }
        if (prev_history_pos != history_pos) {
            const char *history_str = (history_pos >= 0) ? history[static_cast<usize>(history_pos)] : "";
            data->DeleteChars(0, data->BufTextLen);
            data->InsertChars(0, history_str);
        }
    }
    }
    return 0;
}

AppUi::AppUi(GLFWwindow *glfw_window_ptr)
    : glfw_window_ptr{glfw_window_ptr},
      data_directory{std::filesystem::path(sago::getDataHome()) / "GabeVoxelGame"} {
    ImGui::CreateContext();
    auto &style = ImGui::GetStyle();
    auto &io = ImGui::GetIO();
    mono_font = io.Fonts->AddFontFromFileTTF("assets/fonts/Roboto_Mono/RobotoMono-Regular.ttf", 14.0f * 2.0f);
    menu_font = io.Fonts->AddFontFromFileTTF("assets/fonts/Inter_Tight/InterTight-Regular.ttf", 14.0f * 2.0f);
    if (menu_font == nullptr) {
        menu_font = io.Fonts->AddFontDefault();
    }
    if (mono_font == nullptr) {
        mono_font = io.Fonts->AddFontDefault();
    }

    constexpr auto ColorFromBytes = [](u8 r, u8 g, u8 b, u8 a = 255) {
        return ImVec4(static_cast<f32>(r) / 255.0f, static_cast<f32>(g) / 255.0f, static_cast<f32>(b) / 255.0f, static_cast<f32>(a) / 255.0f);
    };
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImVec4 *colors = style.Colors;
    auto bgColor = ColorFromBytes(37, 37, 38);
    auto lightBgColor = ColorFromBytes(82, 82, 85);
    auto veryLightBgColor = ColorFromBytes(90, 90, 95);
    auto panelColor = ColorFromBytes(51, 51, 55);
    auto panelHoverColor = ColorFromBytes(29, 151, 236);
    auto panelActiveColor = ColorFromBytes(0, 119, 200);
    auto textColor = ColorFromBytes(255, 255, 255);
    auto textDisabledColor = ColorFromBytes(151, 151, 151);
    auto borderColor = ColorFromBytes(0, 0, 0, 80);
    colors[ImGuiCol_Text] = textColor;
    colors[ImGuiCol_TextDisabled] = textDisabledColor;
    colors[ImGuiCol_TextSelectedBg] = panelActiveColor;
    colors[ImGuiCol_WindowBg] = bgColor;
    colors[ImGuiCol_ChildBg] = bgColor;
    colors[ImGuiCol_PopupBg] = bgColor;
    colors[ImGuiCol_Border] = borderColor;
    colors[ImGuiCol_BorderShadow] = borderColor;
    colors[ImGuiCol_FrameBg] = panelColor;
    colors[ImGuiCol_FrameBgHovered] = panelHoverColor;
    colors[ImGuiCol_FrameBgActive] = panelActiveColor;
    colors[ImGuiCol_TitleBg] = bgColor;
    colors[ImGuiCol_TitleBgActive] = bgColor;
    colors[ImGuiCol_TitleBgCollapsed] = bgColor;
    colors[ImGuiCol_MenuBarBg] = panelColor;
    colors[ImGuiCol_ScrollbarBg] = panelColor;
    colors[ImGuiCol_ScrollbarGrab] = lightBgColor;
    colors[ImGuiCol_ScrollbarGrabHovered] = veryLightBgColor;
    colors[ImGuiCol_ScrollbarGrabActive] = veryLightBgColor;
    colors[ImGuiCol_CheckMark] = panelActiveColor;
    colors[ImGuiCol_SliderGrab] = panelHoverColor;
    colors[ImGuiCol_SliderGrabActive] = panelActiveColor;
    colors[ImGuiCol_Button] = panelColor;
    colors[ImGuiCol_ButtonHovered] = panelHoverColor;
    colors[ImGuiCol_ButtonActive] = panelHoverColor;
    colors[ImGuiCol_Header] = panelColor;
    colors[ImGuiCol_HeaderHovered] = panelHoverColor;
    colors[ImGuiCol_HeaderActive] = panelActiveColor;
    colors[ImGuiCol_Separator] = borderColor;
    colors[ImGuiCol_SeparatorHovered] = borderColor;
    colors[ImGuiCol_SeparatorActive] = borderColor;
    colors[ImGuiCol_ResizeGrip] = bgColor;
    colors[ImGuiCol_ResizeGripHovered] = panelColor;
    colors[ImGuiCol_ResizeGripActive] = lightBgColor;
    colors[ImGuiCol_PlotLines] = panelActiveColor;
    colors[ImGuiCol_PlotLinesHovered] = panelHoverColor;
    colors[ImGuiCol_PlotHistogram] = panelActiveColor;
    colors[ImGuiCol_PlotHistogramHovered] = panelHoverColor;
    colors[ImGuiCol_DragDropTarget] = bgColor;
    colors[ImGuiCol_NavHighlight] = bgColor;
    colors[ImGuiCol_Tab] = bgColor;
    colors[ImGuiCol_TabActive] = panelActiveColor;
    colors[ImGuiCol_TabUnfocused] = bgColor;
    colors[ImGuiCol_TabUnfocusedActive] = panelActiveColor;
    colors[ImGuiCol_TabHovered] = panelHoverColor;
    style.WindowRounding = 4.0f;
    style.ChildRounding = 4.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.PopupRounding = 4.0f;
    style.ScrollbarRounding = 4.0f;
    style.TabRounding = 4.0f;
    style.FramePadding = {4.0f, 3.0f};

    if (!std::filesystem::exists(data_directory)) {
        std::filesystem::create_directory(data_directory);
    }

    if (std::filesystem::exists(data_directory / "user_settings.json")) {
        settings.load(data_directory / "user_settings.json");
    } else {
        settings.reset_default();
        settings.save(data_directory / "user_settings.json");
    }

    rescale_ui();

    ImGui_ImplGlfw_InitForVulkan(glfw_window_ptr, true);
}

AppUi::~AppUi() {
    if ((settings.autosave || autosave_override) && needs_saving) {
        settings.save(data_directory / "user_settings.json");
    }
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void AppUi::rescale_ui() {
    settings.ui_scl = std::clamp(settings.ui_scl, 0.5f, 2.0f);
    mono_font->Scale = settings.ui_scl * 0.5f;
    menu_font->Scale = settings.ui_scl * 0.5f;
    auto &style = ImGui::GetStyle();
    style.FramePadding = {settings.ui_scl * 4.0f, settings.ui_scl * 3.0f};
}

void AppUi::settings_ui() {
    ImGui::Begin("Settings");
    if (ImGui::BeginTabBar("##settings_tabs")) {
        if (ImGui::BeginTabItem("Game")) {
            if (ImGui::InputText("World Seed", &settings.world_seed_str)) {
                should_upload_seed_data = true;
            }
            if (ImGui::Button("Re-run Startup")) {
                should_run_startup = true;
            }
            if (ImGui::Button("Open Model")) {
                nfdchar_t *out_path = nullptr;
                nfdresult_t const result = NFD_OpenDialog("gvox,vox,vxl,gvp,rle,oct,glp,brk", (data_directory / "models").string().c_str(), &out_path);
                if (result == NFD_OKAY) {
                    gvox_model_path = out_path;
                    should_upload_gvox_model = true;
                    console.add_log("Loaded {}", out_path);
                    free(out_path);
                } else if (result != NFD_CANCEL) {
                    console.add_log("[error]: {}", NFD_GetError());
                }
            }
            {
                ImGui::InputInt3("Load Offset", &gvox_region_range.offset.x);
                auto temp_i32vec3 = gvox_region_range.offset;
                temp_i32vec3.x = static_cast<i32>(gvox_region_range.extent.x);
                temp_i32vec3.y = static_cast<i32>(gvox_region_range.extent.y);
                temp_i32vec3.z = static_cast<i32>(gvox_region_range.extent.z);
                ImGui::InputInt3("Load Extent", &temp_i32vec3.x);
                gvox_region_range.extent.x = static_cast<u32>(std::max(temp_i32vec3.x, 0));
                gvox_region_range.extent.y = static_cast<u32>(std::max(temp_i32vec3.y, 0));
                gvox_region_range.extent.z = static_cast<u32>(std::max(temp_i32vec3.z, 0));
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Graphics")) {
            if (ImGui::Checkbox("Battery Saving Mode", &settings.battery_saving_mode)) {
                needs_saving = true;
            }
            if (ImGui::SliderFloat("Camera FOV", &settings.camera_fov, 0.01f, 170.0f)) {
                needs_saving = true;
            }
            if (ImGui::SliderFloat("Resolution Scale", &settings.render_res_scl, 0.1f, 2.0f)) {
                settings.render_res_scl = std::clamp(std::round(settings.render_res_scl * 20.0f) / 20.0f, 0.1f, 2.0f);
                needs_saving = true;
            }
            // {
            //     auto temp_int = static_cast<int32_t>(1u << settings.log2_chunks_per_axis);
            //     ImGui::InputInt("Render Distance", &temp_int, (temp_int + 1) / 2);
            //     temp_int = std::max(temp_int, 1);
            //     auto new_log2_chunks_per_axis = static_cast<uint32_t>(round(log2(static_cast<double>(temp_int))));
            //     if (new_log2_chunks_per_axis != settings.log2_chunks_per_axis) {
            //         settings.log2_chunks_per_axis = new_log2_chunks_per_axis;
            //         should_recreate_voxel_buffers = true;
            //         needs_saving = true;
            //     }
            // }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("UI")) {
            if (ImGui::InputFloat("Scale", &settings.ui_scl, 0.1f)) {
                needs_saving = true;
                rescale_ui();
            }
            if (ImGui::Checkbox("Show Debug Info", &settings.show_debug_info)) {
                needs_saving = true;
            }
            if (ImGui::Checkbox("Show Console", &settings.show_console)) {
                needs_saving = true;
            }
            if (ImGui::Checkbox("Show Help Menu", &settings.show_help)) {
                needs_saving = true;
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Controls")) {
            settings_controls_ui();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    auto menu_size = ImGui::GetWindowSize().y;
    ImGui::SetCursorPosY(menu_size - (32 * settings.ui_scl + 12));

    ImGui::Separator();

    ImGui::Text("Settings");
    ImGui::SameLine();
    if (ImGui::Checkbox("Auto-save", &settings.autosave)) {
        needs_saving = true;
        autosave_override = true;
    }
    if (!settings.autosave) {
        ImGui::SameLine();
        if (ImGui::Button("Save")) {
            settings.save(data_directory / "user_settings.json");
        }
        ImGui::SameLine();
        if (ImGui::Button("Load")) {
            settings.load(data_directory / "user_settings.json");
            rescale_ui();
            needs_saving = true;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        settings.reset_default();
        rescale_ui();
        needs_saving = true;
    }
    ImGui::End();
}

void AppUi::settings_controls_ui() {
    if (ImGui::BeginCombo("Conflict Resolution Mode", conflict_resolution_strings[conflict_resolution_mode])) {
        for (u32 mode_i = 0; mode_i < conflict_resolution_strings.size(); ++mode_i) {
            bool const is_selected = (mode_i == conflict_resolution_mode);
            if (ImGui::Selectable(conflict_resolution_strings[mode_i], is_selected)) {
                conflict_resolution_mode = mode_i;
            }
        }
        ImGui::EndCombo();
    }
    if (ImGui::SliderFloat("Mouse Sensitivity", &settings.mouse_sensitivity, 0.1f, 10.0f)) {
        needs_saving = true;
    }
    if (ImGui::BeginTable("controls_table", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY, ImVec2(0, -(32 * settings.ui_scl + 12)))) {
        ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 0.0f, 0);
        ImGui::TableSetupColumn("Key", ImGuiTableColumnFlags_WidthStretch, 0.0f, 1);
        ImGui::TableHeadersRow();
        for (usize i = 0; i < control_strings.size(); ++i) {
            ImGui::TableNextRow(ImGuiTableRowFlags_None);
            if (ImGui::TableSetColumnIndex(0)) {
                ImGui::Text("%s", control_strings[i]);
            }
            if (ImGui::TableSetColumnIndex(1)) {
                if (static_cast<i32>(i) == limbo_action_index) {
                    ImGui::Button("<press any key>", ImVec2(-FLT_MIN, 0.0f));
                    if (ImGui::IsKeyDown(ImGuiKey_Escape)) {
                        if (limbo_is_button) {
                            settings.mouse_button_binds.erase(limbo_key_index);
                        } else {
                            settings.keybinds.erase(limbo_key_index);
                        }
                        limbo_action_index = INVALID_GAME_ACTION;
                    } else {
                        auto resolve_action = [this](i32 key_i, std::map<i32, i32> &bindings, bool contains_override) {
                            // set new key
                            new_key_id = key_i;
                            if (bindings.contains(key_i)) {
                                if (limbo_key_index != new_key_id || contains_override) {
                                    // new key to set, but already in bindings
                                    switch (conflict_resolution_mode) {
                                    case 0: {
                                        auto prev_action = bindings[key_i];
                                        bindings[key_i] = limbo_action_index;
                                        if (limbo_is_button) {
                                            settings.mouse_button_binds[limbo_key_index] = prev_action;
                                        } else {
                                            settings.keybinds[limbo_key_index] = prev_action;
                                        }
                                    } break;
                                    case 1: {
                                        bindings[key_i] = limbo_action_index;
                                        if (limbo_is_button) {
                                            settings.mouse_button_binds.erase(limbo_key_index);
                                        } else {
                                            settings.keybinds.erase(limbo_key_index);
                                        }
                                    } break;
                                    case 2: // cancel
                                        break;
                                    }
                                    needs_saving = true;
                                } else {
                                    // same key was pressed. No need to do anything
                                }
                            } else {
                                if (limbo_is_button) {
                                    settings.mouse_button_binds.erase(limbo_key_index);
                                } else {
                                    settings.keybinds.erase(limbo_key_index);
                                }
                                bindings[key_i] = limbo_action_index;
                                needs_saving = true;
                            }
                            limbo_action_index = INVALID_GAME_ACTION;
                        };
                        for (i32 key_i = 0; key_i < GLFW_KEY_LAST + 1; ++key_i) {
                            auto key_state = glfwGetKey(glfw_window_ptr, key_i);
                            if (key_state != GLFW_RELEASE) {
                                resolve_action(key_i, settings.keybinds, limbo_is_button);
                                break;
                            }
                        }
                        if (limbo_action_index != INVALID_GAME_ACTION) {
                            for (i32 button_i = 0; button_i < GLFW_MOUSE_BUTTON_LAST + 1; ++button_i) {
                                auto key_state = glfwGetMouseButton(glfw_window_ptr, button_i);
                                if (key_state != GLFW_RELEASE) {
                                    resolve_action(button_i, settings.mouse_button_binds, !limbo_is_button);
                                    break;
                                }
                            }
                        }
                    }
                } else {
                    char const *key_name = nullptr;
                    auto temp_limbo_key_index = GLFW_KEY_LAST + 1;
                    auto temp_limbo_is_button = false;
                    if (key_name == nullptr) {
                        auto action_key_iter = std::find_if(
                            settings.keybinds.begin(),
                            settings.keybinds.end(),
                            [i](const auto &mo) { return mo.second == static_cast<i32>(i); });
                        if (action_key_iter != settings.keybinds.end()) {
                            key_name = get_key_string(action_key_iter->first);
                            temp_limbo_key_index = action_key_iter->first;
                            temp_limbo_is_button = false;
                        }
                    }
                    if (key_name == nullptr) {
                        auto action_button_iter = std::find_if(
                            settings.mouse_button_binds.begin(),
                            settings.mouse_button_binds.end(),
                            [i](const auto &mo) { return mo.second == static_cast<i32>(i); });
                        if (action_button_iter != settings.mouse_button_binds.end()) {
                            key_name = get_button_string(action_button_iter->first);
                            temp_limbo_key_index = action_button_iter->first;
                            temp_limbo_is_button = true;
                        }
                    }
                    if (key_name == nullptr) {
                        key_name = "Un-set";
                    }
                    auto key_str = std::string{key_name} + "##" + std::to_string(i);
                    if (ImGui::Button(key_str.c_str(), ImVec2(-FLT_MIN, 0.0f))) {
                        if (limbo_action_index == INVALID_GAME_ACTION) {
                            limbo_action_index = static_cast<i32>(i);
                            limbo_key_index = temp_limbo_key_index;
                            limbo_is_button = temp_limbo_is_button;
                        }
                    }
                }
            }
        }
        ImGui::EndTable();
    }
}

static ImGuiTableSortSpecs *current_gpu_resource_info_sort_specs = nullptr;

static auto compare_gpu_resource_infos(const void *lhs, const void *rhs) -> int {
    auto const *a = static_cast<AppUi::GpuResourceInfo const *>(lhs);
    auto const *b = static_cast<AppUi::GpuResourceInfo const *>(rhs);
    for (int n = 0; n < current_gpu_resource_info_sort_specs->SpecsCount; n++) {
        auto const *sort_spec = &current_gpu_resource_info_sort_specs->Specs[n];
        int delta = 0;
        switch (sort_spec->ColumnUserID) {
        case 0: delta = a->type.compare(b->type); break;
        case 1: delta = a->name.compare(b->name); break;
        case 2: delta = static_cast<i32>(static_cast<i64>(a->size) - static_cast<i64>(b->size)); break;
        default: break;
        }
        if (delta > 0) {
            return (sort_spec->SortDirection == ImGuiSortDirection_Ascending) ? +1 : -1;
        }
        if (delta < 0) {
            return (sort_spec->SortDirection == ImGuiSortDirection_Ascending) ? -1 : +1;
        }
    }
    return static_cast<i32>(static_cast<i64>(a->size) - static_cast<i64>(b->size));
}

void AppUi::update(f32 delta_time, f32 cpu_delta_time) {
    full_frametimes[frametime_rotation_index] = delta_time;
    cpu_frametimes[frametime_rotation_index] = cpu_delta_time;
    frametime_rotation_index = (frametime_rotation_index + 1) % full_frametimes.size();

    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::PushFont(menu_font);

    if (paused) {
        ImGuiDockNodeFlags const dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoBackground;
        const ImGuiViewport *viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
        window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
        ImGuiID const dockspace_id = ImGui::GetID("MyDockSpace");
        ImGui::Begin("DockSpace Demo", nullptr, window_flags);
        ImGui::PopStyleVar(3);
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
        if (ImGui::BeginMenuBar()) {
            if (ImGui::Button("Settings")) {
                show_settings = !show_settings;
            }
            ImGui::EndMenuBar();
        }
        ImGui::End();

        if (settings.show_console) {
            console.draw("Console", &settings.show_console);
        }
        if (show_settings) {
            settings_ui();
        }
    }

    if (settings.show_debug_info) {
        ImGui::PushFont(mono_font);
        const ImGuiViewport *viewport = ImGui::GetMainViewport();
        auto pos = viewport->WorkPos;
        pos.x += viewport->WorkSize.x - debug_menu_size;
        ImGui::SetNextWindowPos(pos);
        ImGui::Begin("Debug Menu", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDecoration);
        auto frametime_graph = [&](auto &frametimes) {
            float average = 0.0f;
            for (auto frametime : frametimes) {
                average += frametime;
            }
            average /= static_cast<float>(frametimes.size());
            auto fmt_str = std::string();
            auto [min_frametime_iter, max_frametime_iter] = std::minmax_element(frametimes.begin(), frametimes.end());
            auto min_frametime = *min_frametime_iter;
            auto max_frametime = *max_frametime_iter;
            auto frametime_plot_min = floor(min_frametime * 100.0f) * 0.01f;
            auto frametime_plot_max = ceil(max_frametime * 100.0f) * 0.01f;
            fmt::format_to(std::back_inserter(fmt_str), "avg {:.2f} ms ({:.2f} fps)", average * 1000, 1.0f / average);
            ImGui::PlotLines("", frametimes.data(), static_cast<int>(frametimes.size()), static_cast<int>(frametime_rotation_index), fmt_str.c_str(), frametime_plot_min, frametime_plot_max, ImVec2(0, 120.0f));
            ImGui::Text("min: %.2f ms, max: %.2f ms", static_cast<double>(min_frametime) * 1000, static_cast<double>(max_frametime) * 1000);
        };
        if (ImGui::TreeNode("Full frame-time")) {
            frametime_graph(full_frametimes);
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("CPU only frame-time")) {
            frametime_graph(cpu_frametimes);
            ImGui::TreePop();
        }
        ImGui::Text("GPU: %s", debug_gpu_name);
        ImGui::Text("Est. VRAM usage: %.2f MB", static_cast<double>(debug_vram_usage) / 1000000);
        ImGui::Text("Page count: %u pages (%.2f MB)", debug_page_count, static_cast<double>(debug_page_count) * VOXEL_MALLOC_PAGE_SIZE_BYTES / 1'000'000.0);
        ImGui::Text("GPU heap usage: %.2f MB", static_cast<double>(debug_gpu_heap_usage) / 1'000'000);
        ImGui::Text("Player pos: %.2f, %.2f, %.2f", static_cast<double>(debug_player_pos.x), static_cast<double>(debug_player_pos.y), static_cast<double>(debug_player_pos.z));
        ImGui::Text("Player y/p/r: %.2f, %.2f, %.2f", static_cast<double>(debug_player_rot.x), static_cast<double>(debug_player_rot.y), static_cast<double>(debug_player_rot.z));
        ImGui::Text("Chunk offs: %.2f, %.2f, %.2f", static_cast<double>(debug_chunk_offset.x), static_cast<double>(debug_chunk_offset.y), static_cast<double>(debug_chunk_offset.z));
        ImGui::Text("job_queue_bottom: %u", debug_job_counters.available_threads_queue_bottom);
        ImGui::Text("job_queue_top:    %u", debug_job_counters.available_threads_queue_top);
        ImGui::Text("total_jobs_ran:   %u", debug_total_jobs_ran);

        if (ImGui::TreeNode("GPU Resources")) {
            static ImGuiTableFlags const flags =
                ImGuiTableFlags_Resizable |
                ImGuiTableFlags_Reorderable |
                ImGuiTableFlags_Hideable |
                ImGuiTableFlags_Sortable |
                ImGuiTableFlags_SortMulti |
                ImGuiTableFlags_RowBg |
                ImGuiTableFlags_BordersOuter |
                ImGuiTableFlags_BordersV |
                ImGuiTableFlags_NoBordersInBody |
                ImGuiTableFlags_ScrollY |
                ImGuiTableFlags_SortMulti;

            if (ImGui::BeginTable("#gpu_resource_infos", 3, flags, ImVec2(0.0f, 200.0f), 0.0f)) {
                ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, 0);
                ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 0.0f, 1);
                ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthStretch, 0.0f, 2);
                ImGui::TableSetupScrollFreeze(0, 1);
                ImGui::TableHeadersRow();

                if (ImGuiTableSortSpecs *sorts_specs = ImGui::TableGetSortSpecs()) {
                    if (sorts_specs->SpecsDirty) {
                        current_gpu_resource_info_sort_specs = sorts_specs;
                        if (debug_gpu_resource_infos.size() > 1) {
                            qsort(debug_gpu_resource_infos.data(), debug_gpu_resource_infos.size(), sizeof(debug_gpu_resource_infos[0]), compare_gpu_resource_infos);
                        }
                        current_gpu_resource_info_sort_specs = nullptr;
                        sorts_specs->SpecsDirty = false;
                    }
                }

                ImGuiListClipper clipper;
                clipper.Begin(static_cast<i32>(debug_gpu_resource_infos.size()));
                while (clipper.Step()) {
                    for (int row_i = clipper.DisplayStart; row_i < clipper.DisplayEnd; row_i++) {
                        auto const &res_info = debug_gpu_resource_infos[static_cast<usize>(row_i)];
                        ImGui::PushID(&res_info);
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted(res_info.type.c_str());
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted(res_info.name.c_str());
                        ImGui::TableNextColumn();
                        ImGui::Text("%.4f MB", static_cast<double>(res_info.size) / 1000000);
                        ImGui::PopID();
                    }
                }
                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        debug_menu_size = ImGui::GetWindowSize().x;
        ImGui::End();
        ImGui::PopFont();
    }

    ImGui::PopFont();
    ImGui::Render();

    // Auto-save
    auto now = Clock::now();
    using namespace std::chrono_literals;
    if ((settings.autosave || autosave_override) && needs_saving && now - last_save_time > 0.1s) {
        settings.save(data_directory / "user_settings.json");
        needs_saving = false;
        autosave_override = false;
    }
}

void AppUi::toggle_pause() {
    if (show_settings) {
        show_settings = false;
    } else {
        paused = !paused;
    }
}

void AppUi::toggle_debug() {
    settings.show_debug_info = !settings.show_debug_info;
    needs_saving = true;
}

void AppUi::toggle_help() {
    settings.show_help = !settings.show_help;
    needs_saving = true;
}

void AppUi::toggle_console() {
    settings.show_console = !settings.show_console;
    needs_saving = true;
}
