#pragma once

#include <mutex>

#include <base/vec.hpp>
#include <base/str.hpp>
#include <base/hash_map.hpp>
#include <application/settings.inl>
#include <imgui.h>

namespace debug_utils {
    struct Console {
        char input_buffer[256]{};
        Vec<Str> items;
        Vec<const char *> commands;
        Vec<char *> history;
        int history_pos{-1};
        ImGuiTextFilter filter;
        bool auto_scroll{true};
        bool scroll_to_bottom{false};
        std::mutex items_mtx = std::mutex();
        inline static Console *s_instance = nullptr;

        Console();
        ~Console();

        static void clear_log();
        static void add_log(char const *str);
        static void draw(const char *title, bool *p_open);
        static void exec_command(const char *command_line);
        static int on_text_edit(ImGuiInputTextCallbackData *data);
    };

    struct Pass {
        Str name;
        daxa::TaskImageView task_image_id;
        daxa_u32 type;
        DebugImageSettings settings = {.flags = 0, .brightness = 1.0f};
    };

    struct DebugDisplay {
        struct GpuResourceInfo {
            Str type;
            Str name;
            size_t size;
        };
        Vec<GpuResourceInfo> gpu_resource_infos;
        Vec<Pass> prev_passes{};
        Vec<Pass> passes{};
        uint32_t selected_pass{};
        Str selected_pass_name{};

        HashMap<Str, Str> debug_strings{};

        inline static DebugDisplay *s_instance = nullptr;

        DebugDisplay();
        ~DebugDisplay();

        static void begin_passes();
        static void add_pass(Pass const &info);

        static void set_debug_string(char const *id, char const *value);
    };
} // namespace debug_utils
