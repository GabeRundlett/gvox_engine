#pragma once

#include "window.hpp"

#include <thread>
using namespace std::chrono_literals;
#include <filesystem>
#include <cmath>

#include <daxa/utils/imgui.hpp>
#include <imgui_impl_glfw.h>
#include "imgui_console.hpp"

#include <daxa/utils/pipeline_manager.hpp>

#include <daxa/utils/math_operators.hpp>
using namespace daxa::math_operators;

using namespace daxa::types;
#include <shared/shared.inl>

using Clock = std::chrono::high_resolution_clock;

#define APPNAME "Voxel Game"
#define APPNAME_PREFIX(x) ("[" APPNAME "] " x)

#define PIPELINE_MANAGER_OPTIONS                                           \
    {                                                                      \
        .root_paths = {                                                    \
            DAXA_SHADER_INCLUDE_DIR,                                       \
            "assets",                                                      \
            "shaders",                                                     \
            "src",                                                         \
        },                                                                 \
        .language = daxa::ShaderLanguage::GLSL, .enable_debug_info = true, \
    }

template <typename T>
struct BaseApp : AppWindow<T> {
    daxa::Context daxa_ctx = daxa::create_context({
        .enable_validation = false,
    });
    daxa::Device device = daxa_ctx.create_device({
        .selector = [](daxa::DeviceProperties const &device_props) -> i32 {
            i32 score = 0;
            switch (device_props.device_type) {
            case daxa::DeviceType::DISCRETE_GPU: score += 10000; break;
            case daxa::DeviceType::VIRTUAL_GPU: score += 1000; break;
            case daxa::DeviceType::INTEGRATED_GPU: score += 100; break;
            default: break;
            }
            return score;
        },
#if defined(NDEBUG)
        .enable_buffer_device_address_capture_replay = false,
#endif
        .debug_name = APPNAME_PREFIX("device"),
    });

    daxa::Swapchain swapchain = device.create_swapchain({
        .native_window = AppWindow<T>::get_native_handle(),
        .native_window_platform = AppWindow<T>::get_native_platform(),
        .present_mode = daxa::PresentMode::DO_NOT_WAIT_FOR_VBLANK,
        .image_usage = daxa::ImageUsageFlagBits::TRANSFER_DST,
        .debug_name = APPNAME_PREFIX("swapchain"),
    });

    daxa::PipelineManager basic_pipeline_manager = daxa::PipelineManager({
        .device = device,
        .shader_compile_options = PIPELINE_MANAGER_OPTIONS,
        .debug_name = APPNAME_PREFIX("pipeline_manager"),
    });
    daxa::PipelineManager chunkgen_pipeline_manager = daxa::PipelineManager({
        .device = device,
        .shader_compile_options = PIPELINE_MANAGER_OPTIONS,
        .debug_name = APPNAME_PREFIX("pipeline_manager"),
    });
    daxa::PipelineManager startup_pipeline_manager = daxa::PipelineManager({
        .device = device,
        .shader_compile_options = PIPELINE_MANAGER_OPTIONS,
        .debug_name = APPNAME_PREFIX("pipeline_manager"),
    });
    daxa::PipelineManager optical_depth_pipeline_manager = daxa::PipelineManager({
        .device = device,
        .shader_compile_options = PIPELINE_MANAGER_OPTIONS,
        .debug_name = APPNAME_PREFIX("pipeline_manager"),
    });

    ImFont *mono_font = nullptr;
    ImFont *menu_font = nullptr;
    daxa::ImGuiRenderer imgui_renderer = create_imgui_renderer();

    void load_imgui_fonts(f32 scl) {
        ImGuiIO &io = ImGui::GetIO();
        mono_font = io.Fonts->AddFontFromFileTTF("assets/fonts/Roboto_Mono/RobotoMono-Regular.ttf", 14.0f * scl);
        menu_font = io.Fonts->AddFontFromFileTTF("assets/fonts/Inter_Tight/InterTight-Regular.ttf", 14.0f * scl);
        if (!menu_font)
            menu_font = io.Fonts->AddFontDefault();
        if (!mono_font)
            mono_font = io.Fonts->AddFontDefault();
    }

    auto create_imgui_renderer() -> daxa::ImGuiRenderer {
        ImGui::CreateContext();
        load_imgui_fonts(1.0f);

        ImGui_ImplGlfw_InitForVulkan(AppWindow<T>::glfw_window_ptr, true);
        return daxa::ImGuiRenderer({
            .device = device,
            .format = swapchain.get_format(),
        });
    }

    Clock::time_point start = Clock::now(), prev_time = start;
    f32 elapsed_s = 1.0f;

    daxa::CommandSubmitInfo submit_info;

    daxa::ImageId swapchain_image;
    daxa::TaskImageId task_swapchain_image;

    BaseApp() : AppWindow<T>(APPNAME) {
        constexpr auto ColorFromBytes = [](u8 r, u8 g, u8 b, u8 a = 255) {
            return ImVec4((f32)r / 255.0f, (f32)g / 255.0f, (f32)b / 255.0f, (f32)a / 255.0f);
        };
        auto &style = ImGui::GetStyle();
        auto &io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        ImVec4 *colors = style.Colors;
        const ImVec4 bgColor = ColorFromBytes(37, 37, 38);
        const ImVec4 lightBgColor = ColorFromBytes(82, 82, 85);
        const ImVec4 veryLightBgColor = ColorFromBytes(90, 90, 95);
        const ImVec4 panelColor = ColorFromBytes(51, 51, 55);
        const ImVec4 panelHoverColor = ColorFromBytes(29, 151, 236);
        const ImVec4 panelActiveColor = ColorFromBytes(0, 119, 200);
        const ImVec4 textColor = ColorFromBytes(255, 255, 255);
        const ImVec4 textDisabledColor = ColorFromBytes(151, 151, 151);
        const ImVec4 borderColor = ColorFromBytes(0, 0, 0, 80);
        const ImVec4 borderShadowColor = ColorFromBytes(78, 78, 78, 90);
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
    }

    ~BaseApp() {
        ImGui_ImplGlfw_Shutdown();
    }

    auto update() -> bool {
        glfwPollEvents();
        if (glfwWindowShouldClose(AppWindow<T>::glfw_window_ptr)) {
            return true;
        }

        if (!AppWindow<T>::minimized) {
            reinterpret_cast<T *>(this)->on_update();
        } else {
            std::this_thread::sleep_for(1ms);
        }

        return false;
    }

    auto record_loop_task_list() -> daxa::TaskList {
        daxa::TaskList new_task_list = daxa::TaskList({
            .device = device,
            // .dont_use_split_barriers = true,
            .swapchain = swapchain,
            .debug_name = APPNAME_PREFIX("task_list"),
        });
        task_swapchain_image = new_task_list.create_task_image({
            .swapchain_image = true,
            .debug_name = APPNAME_PREFIX("task_swapchain_image"),
        });
        new_task_list.add_runtime_image(task_swapchain_image, swapchain_image);

        reinterpret_cast<T *>(this)->record_tasks(new_task_list);

        new_task_list.submit(&submit_info);
        new_task_list.present({});
        new_task_list.complete();

        // new_task_list.output_graphviz();

        return new_task_list;
    }
};
