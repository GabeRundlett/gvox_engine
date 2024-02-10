#include <debug_utils.hpp>
#include <fmt/format.h>

debug_utils::Console::Console() {
    s_instance = this;
    clear_log();
    memset(input_buffer, 0, sizeof(input_buffer));
}

debug_utils::Console::~Console() {
    clear_log();
    if (s_instance == this) {
        s_instance = nullptr;
    }
    for (auto &i : history) {
        free(i);
    }
}

void debug_utils::Console::clear_log() {
    auto &self = *s_instance;
    auto lock = std::lock_guard{*self.items_mtx};
    self.items.clear();
}

void debug_utils::Console::add_log(std::string const &str) {
    auto &self = *s_instance;
    {
        auto lock = std::lock_guard{*self.items_mtx};
        self.items.push_back(str);
    }
    std::cout << str << std::endl;
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

void debug_utils::Console::draw(const char *title, bool *p_open) {
    auto &self = *s_instance;
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
    if (ImGui::SmallButton("Clear")) {
        clear_log();
    }
    ImGui::SameLine();
    bool const copy_to_clipboard = ImGui::SmallButton("Copy");
    ImGui::Separator();
    if (ImGui::BeginPopup("Options")) {
        ImGui::Checkbox("Auto-scroll", &self.auto_scroll);
        ImGui::EndPopup();
    }
    if (ImGui::Button("Options")) {
        ImGui::OpenPopup("Options");
    }
    ImGui::SameLine();
    self.filter.Draw(R"(Filter ("incl,-excl") ("error"))", 180);
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
    {
        auto lock = std::lock_guard{*self.items_mtx};
        for (auto const &item : self.items) {
            if (!self.filter.PassFilter(item.c_str())) {
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
    }
    if (copy_to_clipboard) {
        ImGui::LogFinish();
    }
    if (self.scroll_to_bottom || (self.auto_scroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())) {
        ImGui::SetScrollHereY(1.0f);
    }
    self.scroll_to_bottom = false;
    ImGui::PopStyleVar();
    ImGui::EndChild();
    ImGui::Separator();
    bool reclaim_focus = false;
    ImGuiInputTextFlags const input_text_flags = ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackCompletion | ImGuiInputTextFlags_CallbackHistory;
    if (ImGui::InputText(
            "Input", self.input_buffer, IM_ARRAYSIZE(self.input_buffer), input_text_flags, [](ImGuiInputTextCallbackData *data) -> int {
                auto *user_console = static_cast<Console *>(data->UserData);
                return user_console->on_text_edit(data);
            },
            static_cast<void *>(&self))) {
        char *s = self.input_buffer;
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

void debug_utils::Console::exec_command(const char *command_line) {
    auto &self = *s_instance;
    add_log(fmt::format("# {}\n", command_line));
    self.history_pos = -1;
    for (daxa_i32 i = static_cast<daxa_i32>(self.history.size()) - 1; i >= 0; i--) {
        if (Stricmp(self.history[static_cast<size_t>(i)], command_line) == 0) {
            free(self.history[static_cast<size_t>(i)]);
            self.history.erase(self.history.begin() + i);
            break;
        }
    }
    self.history.push_back(Strdup(command_line));
    add_log(fmt::format("Unknown command: '{}'\n", command_line));
    self.scroll_to_bottom = true;
}

auto debug_utils::Console::on_text_edit(ImGuiInputTextCallbackData *data) -> int {
    auto &self = *s_instance;
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
        for (auto &command : self.commands) {
            if (Strnicmp(command, word_start, static_cast<daxa_i32>(word_end - word_start)) == 0) {
                candidates.push_back(command);
            }
        }
        if (candidates.empty()) {
            add_log(fmt::format("No match for \"{}\"!\n", /* (int)(word_end - word_start), */ word_start));
        } else if (candidates.size() == 1) {
            data->DeleteChars(static_cast<daxa_i32>(word_start - data->Buf), static_cast<daxa_i32>(word_end - word_start));
            data->InsertChars(data->CursorPos, candidates[0]);
            data->InsertChars(data->CursorPos, " ");
        } else {
            int match_len = static_cast<daxa_i32>(word_end - word_start);
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
                data->DeleteChars(static_cast<daxa_i32>(word_start - data->Buf), static_cast<daxa_i32>(word_end - word_start));
                data->InsertChars(data->CursorPos, candidates[0], candidates[0] + match_len);
            }
            add_log("Possible matches:\n");
            for (auto &candidate : candidates) {
                add_log(fmt::format("- {}\n", candidate));
            }
        }
        break;
    }
    case ImGuiInputTextFlags_CallbackHistory: {
        const int prev_history_pos = self.history_pos;
        if (data->EventKey == ImGuiKey_UpArrow) {
            if (self.history_pos == -1) {
                self.history_pos = static_cast<daxa_i32>(self.history.size()) - 1;
            } else if (self.history_pos > 0) {
                self.history_pos--;
            }
        } else if (data->EventKey == ImGuiKey_DownArrow) {
            if (self.history_pos != -1) {
                if (static_cast<size_t>(++self.history_pos) >= self.history.size()) {
                    self.history_pos = -1;
                }
            }
        }
        if (prev_history_pos != self.history_pos) {
            const char *history_str = (self.history_pos >= 0) ? self.history[static_cast<size_t>(self.history_pos)] : "";
            data->DeleteChars(0, data->BufTextLen);
            data->InsertChars(0, history_str);
        }
    }
    }
    return 0;
}

debug_utils::DebugDisplay::DebugDisplay() {
    s_instance = this;
}

debug_utils::DebugDisplay::~DebugDisplay() {
    if (s_instance == this) {
        s_instance = nullptr;
    }
}

void debug_utils::DebugDisplay::begin_passes() {
    auto &self = *s_instance;
    std::swap(self.prev_passes, self.passes);
    self.passes.clear();
}

void debug_utils::DebugDisplay::add_pass(Pass const &info) {
    auto &self = *s_instance;
    auto prev_iter = std::find_if(
        self.prev_passes.begin(), self.prev_passes.end(),
        [&](Pass const &other) { return info.name == other.name; });
    auto new_info = info;
    if (prev_iter != self.prev_passes.end()) {
        new_info.settings = prev_iter->settings;
    }
    self.passes.push_back(new_info);
}
