#pragma once

#include <mutex>

#include <application/settings.inl>
#include <imgui.h>

namespace debug_utils {
    struct Console {
        char input_buffer[256]{};
        std::vector<std::string> items;
        std::vector<const char *> commands;
        std::vector<char *> history;
        int history_pos{-1};
        ImGuiTextFilter filter;
        bool auto_scroll{true};
        bool scroll_to_bottom{false};
        std::shared_ptr<std::mutex> items_mtx = std::make_shared<std::mutex>();
        inline static Console *s_instance = nullptr;

        Console();
        ~Console();

        static void clear_log();
        static void add_log(std::string const &str);
        static void draw(const char *title, bool *p_open);
        static void exec_command(const char *command_line);
        static int on_text_edit(ImGuiInputTextCallbackData *data);
    };

    struct Pass {
        std::string name;
        daxa::TaskImageView task_image_id;
        daxa_u32 type;
        DebugImageSettings settings = {.flags = 0, .brightness = 1.0f};
    };

    struct DebugDisplay {
        struct GpuResourceInfo {
            std::string type;
            std::string name;
            size_t size;
        };
        std::vector<GpuResourceInfo> gpu_resource_infos;
        std::vector<Pass> prev_passes{};
        std::vector<Pass> passes{};
        uint32_t selected_pass{};
        std::string selected_pass_name{};

        inline static DebugDisplay *s_instance = nullptr;

        DebugDisplay();
        ~DebugDisplay();

        static void begin_passes();
        static void add_pass(Pass const &info);
    };
} // namespace debug_utils
