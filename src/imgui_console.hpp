#include <vector>

struct ImGuiConsole {
    char input_buffer[256];
    std::vector<char *> items;
    std::vector<const char *> commands;
    std::vector<char *> history;
    int history_pos;
    ImGuiTextFilter filter;
    bool auto_scroll;
    bool scroll_to_bottom;

    ImGuiConsole() {
        clear_log();
        memset(input_buffer, 0, sizeof(input_buffer));
        history_pos = -1;

        auto_scroll = true;
        scroll_to_bottom = false;
    }
    ~ImGuiConsole() {
        clear_log();
        for (int i = 0; i < history.size(); i++)
            free(history[i]);
    }
    static int Stricmp(const char *s1, const char *s2) {
        int d;
        while ((d = toupper(*s2) - toupper(*s1)) == 0 && *s1) {
            s1++;
            s2++;
        }
        return d;
    }
    static int Strnicmp(const char *s1, const char *s2, int n) {
        int d = 0;
        while (n > 0 && (d = toupper(*s2) - toupper(*s1)) == 0 && *s1) {
            s1++;
            s2++;
            n--;
        }
        return d;
    }
    static char *Strdup(const char *s) {
        IM_ASSERT(s);
        size_t len = strlen(s) + 1;
        void *buf = malloc(len);
        IM_ASSERT(buf);
        return (char *)memcpy(buf, (const void *)s, len);
    }
    static void Strtrim(char *s) {
        char *str_end = s + strlen(s);
        while (str_end > s && str_end[-1] == ' ')
            str_end--;
        *str_end = 0;
    }
    void clear_log() {
        for (int i = 0; i < items.size(); i++)
            free(items[i]);
        items.clear();
    }
    void add_log(const char *fmt, ...) IM_FMTARGS(2) {
        char buf[1024];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, IM_ARRAYSIZE(buf), fmt, args);
        buf[IM_ARRAYSIZE(buf) - 1] = 0;
        va_end(args);
        items.push_back(Strdup(buf));
    }
    void draw(const char *title, bool *p_open) {
        ImGui::SetNextWindowSize(ImVec2(520, 600), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin(title, p_open)) {
            ImGui::End();
            return;
        }
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::MenuItem("Close Console"))
                *p_open = false;
            ImGui::EndPopup();
        }
        // if (ImGui::SmallButton("Add Debug Text")) {
        //     add_log("%d some text", items.size());
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
        bool copy_to_clipboard = ImGui::SmallButton("Copy");
        ImGui::Separator();
        if (ImGui::BeginPopup("Options")) {
            ImGui::Checkbox("Auto-scroll", &auto_scroll);
            ImGui::EndPopup();
        }
        if (ImGui::Button("Options"))
            ImGui::OpenPopup("Options");
        ImGui::SameLine();
        filter.Draw("Filter (\"incl,-excl\") (\"error\")", 180);
        ImGui::Separator();
        const float footer_height_to_reserve = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
        ImGui::BeginChild("ScrollingRegion", ImVec2(0, -footer_height_to_reserve), false, ImGuiWindowFlags_HorizontalScrollbar);
        if (ImGui::BeginPopupContextWindow()) {
            if (ImGui::Selectable("Clear"))
                clear_log();
            ImGui::EndPopup();
        }
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 1));
        if (copy_to_clipboard)
            ImGui::LogToClipboard();
        for (int i = 0; i < items.size(); i++) {
            const char *item = items[i];
            if (!filter.PassFilter(item))
                continue;
            ImVec4 color;
            bool has_color = false;
            if (strstr(item, "[error]")) {
                color = ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
                has_color = true;
            } else if (strncmp(item, "# ", 2) == 0) {
                color = ImVec4(1.0f, 0.8f, 0.6f, 1.0f);
                has_color = true;
            }
            if (has_color)
                ImGui::PushStyleColor(ImGuiCol_Text, color);
            ImGui::TextUnformatted(item);
            if (has_color)
                ImGui::PopStyleColor();
        }
        if (copy_to_clipboard)
            ImGui::LogFinish();
        if (scroll_to_bottom || (auto_scroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY()))
            ImGui::SetScrollHereY(1.0f);
        scroll_to_bottom = false;
        ImGui::PopStyleVar();
        ImGui::EndChild();
        ImGui::Separator();
        bool reclaim_focus = false;
        ImGuiInputTextFlags input_text_flags = ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackCompletion | ImGuiInputTextFlags_CallbackHistory;
        if (ImGui::InputText(
                "Input", input_buffer, IM_ARRAYSIZE(input_buffer), input_text_flags, [](ImGuiInputTextCallbackData *data) -> int {
                    ImGuiConsole *console = (ImGuiConsole *)data->UserData;
                    return console->on_text_edit(data);
                },
                (void *)this)) {
            char *s = input_buffer;
            Strtrim(s);
            if (s[0])
                exec_command(s);
            s[0] = '\0';
            reclaim_focus = true;
        }
        ImGui::SetItemDefaultFocus();
        if (reclaim_focus)
            ImGui::SetKeyboardFocusHere(-1);
        ImGui::End();
    }
    void exec_command(const char *command_line) {
        add_log("# %s\n", command_line);
        history_pos = -1;
        for (int i = history.size() - 1; i >= 0; i--) {
            if (Stricmp(history[i], command_line) == 0) {
                free(history[i]);
                history.erase(history.begin() + i);
                break;
            }
        }
        history.push_back(Strdup(command_line));
        add_log("Unknown command: '%s'\n", command_line);
        scroll_to_bottom = true;
    }
    int on_text_edit(ImGuiInputTextCallbackData *data) {
        switch (data->EventFlag) {
        case ImGuiInputTextFlags_CallbackCompletion: {
            const char *word_end = data->Buf + data->CursorPos;
            const char *word_start = word_end;
            while (word_start > data->Buf) {
                const char c = word_start[-1];
                if (c == ' ' || c == '\t' || c == ',' || c == ';')
                    break;
                word_start--;
            }
            ImVector<const char *> candidates;
            for (int i = 0; i < commands.size(); i++)
                if (Strnicmp(commands[i], word_start, (int)(word_end - word_start)) == 0)
                    candidates.push_back(commands[i]);
            if (candidates.size() == 0) {
                add_log("No match for \"%.*s\"!\n", (int)(word_end - word_start), word_start);
            } else if (candidates.size() == 1) {
                data->DeleteChars((int)(word_start - data->Buf), (int)(word_end - word_start));
                data->InsertChars(data->CursorPos, candidates[0]);
                data->InsertChars(data->CursorPos, " ");
            } else {
                int match_len = (int)(word_end - word_start);
                for (;;) {
                    int c = 0;
                    bool all_candidates_matches = true;
                    for (int i = 0; i < candidates.size() && all_candidates_matches; i++)
                        if (i == 0)
                            c = toupper(candidates[i][match_len]);
                        else if (c == 0 || c != toupper(candidates[i][match_len]))
                            all_candidates_matches = false;
                    if (!all_candidates_matches)
                        break;
                    match_len++;
                }
                if (match_len > 0) {
                    data->DeleteChars((int)(word_start - data->Buf), (int)(word_end - word_start));
                    data->InsertChars(data->CursorPos, candidates[0], candidates[0] + match_len);
                }
                add_log("Possible matches:\n");
                for (int i = 0; i < candidates.size(); i++)
                    add_log("- %s\n", candidates[i]);
            }
            break;
        }
        case ImGuiInputTextFlags_CallbackHistory: {
            const int prev_history_pos = history_pos;
            if (data->EventKey == ImGuiKey_UpArrow) {
                if (history_pos == -1)
                    history_pos = history.size() - 1;
                else if (history_pos > 0)
                    history_pos--;
            } else if (data->EventKey == ImGuiKey_DownArrow) {
                if (history_pos != -1)
                    if (++history_pos >= history.size())
                        history_pos = -1;
            }
            if (prev_history_pos != history_pos) {
                const char *history_str = (history_pos >= 0) ? history[history_pos] : "";
                data->DeleteChars(0, data->BufTextLen);
                data->InsertChars(0, history_str);
            }
        }
        }
        return 0;
    }
};

static inline ImGuiConsole imgui_console;
