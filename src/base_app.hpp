#pragma once

#include "window.hpp"

#include <thread>
using namespace std::chrono_literals;
#include <iostream>
#include <cmath>

#include <daxa/utils/imgui.hpp>
#include "imgui/imgui_impl_glfw.h"

#include <daxa/utils/math_operators.hpp>
using namespace daxa::math_operators;

using namespace daxa::types;
#include <shared/shared.inl>

using Clock = std::chrono::high_resolution_clock;

#define APPNAME "Voxel Game"
#define APPNAME_PREFIX(x) ("[" APPNAME "] " x)

template <typename T>
struct BaseApp : AppWindow<T> {
    daxa::Context daxa_ctx = daxa::create_context({
        .enable_validation = true,
    });
    daxa::Device device = daxa_ctx.create_device({
        .debug_name = APPNAME_PREFIX("device"),
    });

    daxa::Swapchain swapchain = device.create_swapchain({
        .native_window = AppWindow<T>::get_native_handle(),
        .present_mode = daxa::PresentMode::DO_NOT_WAIT_FOR_VBLANK,
        .image_usage = daxa::ImageUsageFlagBits::TRANSFER_DST,
        .debug_name = APPNAME_PREFIX("swpachain"),
    });

    daxa::PipelineCompiler pipeline_compiler = device.create_pipeline_compiler({
        .shader_compile_options = {
            .root_paths = {
#if _WIN32
                ".out/debug/vcpkg_installed/x64-windows/include",
#elif __linux__
                ".out/debug/vcpkg_installed/x64-linux/include",
#endif
                "shaders",
                "src",
            },
            .language = daxa::ShaderLanguage::GLSL,
        },
        .debug_name = APPNAME_PREFIX("pipeline_compiler"),
    });

    daxa::ImGuiRenderer imgui_renderer = create_imgui_renderer();
    auto create_imgui_renderer() -> daxa::ImGuiRenderer {
        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForVulkan(AppWindow<T>::glfw_window_ptr, true);
        return daxa::ImGuiRenderer({
            .device = device,
            .pipeline_compiler = pipeline_compiler,
            .format = swapchain.get_format(),
        });
    }

    daxa::BinarySemaphore binary_semaphore = device.create_binary_semaphore({
        .debug_name = APPNAME_PREFIX("binary_semaphore"),
    });
    static inline constexpr u64 FRAMES_IN_FLIGHT = 3;
    daxa::TimelineSemaphore gpu_framecount_timeline_sema = device.create_timeline_semaphore(daxa::TimelineSemaphoreInfo{
        .initial_value = 0,
        .debug_name = APPNAME_PREFIX("gpu_framecount_timeline_sema"),
    });
    u64 cpu_framecount = FRAMES_IN_FLIGHT - 1;

    Clock::time_point start = Clock::now(), prev_time = start;
    f32 elapsed_s = 1.0f;

    daxa::BinarySemaphore acquire_semaphore = device.create_binary_semaphore({.debug_name = APPNAME_PREFIX("acquire_semaphore")});
    daxa::BinarySemaphore present_semaphore = device.create_binary_semaphore({.debug_name = APPNAME_PREFIX("present_semaphore")});
    daxa::CommandSubmitInfo submit_info;

    daxa::ImageId swapchain_image;
    daxa::TaskImageId task_swapchain_image;
    daxa::TaskList loop_task_list = record_loop_task_list();

    BaseApp() : AppWindow<T>(APPNAME) {}

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

    auto reload_pipeline(auto &pipeline) -> bool {
        if (pipeline_compiler.check_if_sources_changed(pipeline)) {
            auto new_pipeline = pipeline_compiler.recreate_compute_pipeline(pipeline);
            std::cout << new_pipeline.to_string() << std::endl;
            if (new_pipeline.is_ok()) {
                pipeline = new_pipeline.value();
                return true;
            }
        }
        return false;
    }

    void submit_task_list() {
        swapchain_image = swapchain.acquire_next_image(acquire_semaphore);
        ++cpu_framecount;
        submit_info.signal_timeline_semaphores = {{gpu_framecount_timeline_sema, cpu_framecount}};
        loop_task_list.execute();
        gpu_framecount_timeline_sema.wait_for_value(cpu_framecount - FRAMES_IN_FLIGHT);
    }

    auto record_loop_task_list() -> daxa::TaskList {
        daxa::TaskList new_task_list = daxa::TaskList({
            .device = device,
            .debug_name = APPNAME_PREFIX("task_list"),
        });
        task_swapchain_image = new_task_list.create_task_image({
            .fetch_callback = [this]() { return swapchain_image; },
            .swapchain_parent = std::pair{swapchain, acquire_semaphore},
            .debug_name = APPNAME_PREFIX("task_swapchain_image"),
        });

        reinterpret_cast<T *>(this)->record_tasks(new_task_list);

        new_task_list.add_task({
            .used_images = {
                {task_swapchain_image, daxa::TaskImageAccess::COLOR_ATTACHMENT},
            },
            .task = [this](daxa::TaskInterface interf) {
                auto cmd_list = interf.get_command_list();
                imgui_renderer.record_commands(ImGui::GetDrawData(), cmd_list, swapchain_image, AppWindow<T>::size_x, AppWindow<T>::size_y);
            },
            .debug_name = APPNAME_PREFIX("ImGui Task"),
        });

        new_task_list.submit(&submit_info);
        new_task_list.present({});
        new_task_list.compile();

        // new_task_list.output_graphviz();

        return new_task_list;
    }
};
