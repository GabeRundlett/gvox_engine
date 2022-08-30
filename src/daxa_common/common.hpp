#pragma once

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include "window.hpp"

#include <daxa/utils/fsr2.hpp>

#include <daxa/utils/imgui.hpp>
#include "imgui/imgui_impl_glfw.h"

using namespace daxa::types;
using Clock = std::chrono::high_resolution_clock;

namespace daxa_common {
    template <typename T>
    struct App : AppWindow<T> {
        daxa::Context daxa_ctx = daxa::create_context({
            .enable_validation = false,
        });
        daxa::Device device = daxa_ctx.create_device({
            .selector = [](const daxa::DeviceProperties &props) {
                i32 score = 0;
                switch (props.device_type) {
                case daxa::DeviceType::DISCRETE_GPU: score += 10000; break;
                case daxa::DeviceType::INTEGRATED_GPU: score += 100; break;
                default: break;
                }
                return score;
            },
            .debug_name = APPNAME_PREFIX("device"),
        });

        daxa::Swapchain swapchain = device.create_swapchain({
            .native_window = AppWindow<T>::get_native_handle(),
            .width = AppWindow<T>::size_x,
            .height = AppWindow<T>::size_y,
            .present_mode = daxa::PresentMode::DO_NOT_WAIT_FOR_VBLANK,
            .image_usage = daxa::ImageUsageFlagBits::TRANSFER_DST,
            .debug_name = APPNAME_PREFIX("swapchain"),
        });

        daxa::PipelineCompiler pipeline_compiler = device.create_pipeline_compiler({
            .root_paths = {
                ".out/debug/vcpkg_installed/x64-windows/include",
                "shaders/pipelines",
                "shaders",
            },
            .opt_level = 3,
            .shader_model_major = 6,
            .shader_model_minor = 6,
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

        static inline constexpr u64 FRAMES_IN_FLIGHT = 1;
        daxa::TimelineSemaphore gpu_framecount_timeline_sema = device.create_timeline_semaphore(daxa::TimelineSemaphoreInfo{
            .initial_value = 0,
            .debug_name = APPNAME_PREFIX("gpu_framecount_timeline_sema"),
        });
        u64 cpu_framecount = FRAMES_IN_FLIGHT - 1;

        Clock::time_point start = Clock::now(), prev_time = start;
        f32 time, delta_time;

        bool paused = true;
        bool fsr_enabled = true;

        daxa::ImageId swapchain_image;
        daxa::TaskImageId task_swapchain_image;

        daxa::ImageId color_image, display_image, motion_vectors_image, depth_image;
        daxa::TaskImageId task_color_image, task_display_image, task_motion_vectors_image, task_depth_image;

        daxa::Fsr2Context fsr_context = daxa::Fsr2Context{{.device = device}};
        u32 render_size_x, render_size_y;
        f32vec2 prev_jitter{}, jitter{};
        f32 render_scl = 1.0f;
        f32 fov;

        daxa::TaskList loop_task_list;

        App(const char *name) : AppWindow<T>(name), loop_task_list{record_loop_task_list()} {
            create_render_images();
        }

        ~App() {
            ImGui_ImplGlfw_Shutdown();
            destroy_render_images();
        }

        void create_render_images() {
            render_size_x = std::max<u32>(1, static_cast<u32>(static_cast<f32>(AppWindow<T>::size_x) * render_scl));
            render_size_y = std::max<u32>(1, static_cast<u32>(static_cast<f32>(AppWindow<T>::size_y) * render_scl));

            color_image = device.create_image({
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .aspect = daxa::ImageAspectFlagBits::COLOR,
                .size = {render_size_x, render_size_y, 1},
                .usage = daxa::ImageUsageFlagBits::COLOR_ATTACHMENT | daxa::ImageUsageFlagBits::SHADER_READ_ONLY | daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::TRANSFER_SRC,
                .debug_name = "FSR sample Render Color Image",
            });
            display_image = device.create_image({
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .aspect = daxa::ImageAspectFlagBits::COLOR,
                .size = {AppWindow<T>::size_x, AppWindow<T>::size_y, 1},
                .usage = daxa::ImageUsageFlagBits::COLOR_ATTACHMENT | daxa::ImageUsageFlagBits::SHADER_READ_ONLY | daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .debug_name = "FSR sample Display Color Image",
            });
            motion_vectors_image = device.create_image({
                .format = daxa::Format::R16G16_SFLOAT,
                .aspect = daxa::ImageAspectFlagBits::COLOR,
                .size = {render_size_x, render_size_y, 1},
                .usage = daxa::ImageUsageFlagBits::COLOR_ATTACHMENT | daxa::ImageUsageFlagBits::SHADER_READ_ONLY | daxa::ImageUsageFlagBits::SHADER_READ_WRITE,
                .debug_name = "FSR sample Render MV Image",
            });
            depth_image = device.create_image({
                .format = daxa::Format::D32_SFLOAT,
                .aspect = daxa::ImageAspectFlagBits::DEPTH,
                .size = {render_size_x, render_size_y, 1},
                .usage = daxa::ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT | daxa::ImageUsageFlagBits::SHADER_READ_ONLY,
                .debug_name = "FSR sample Render Depth Image",
            });

            fsr_context.resize({
                .render_size_x = render_size_x,
                .render_size_y = render_size_y,
                .display_size_x = AppWindow<T>::size_x,
                .display_size_y = AppWindow<T>::size_y,
            });
        }
        void destroy_render_images() {
            device.destroy_image(color_image);
            device.destroy_image(display_image);
            device.destroy_image(motion_vectors_image);
            device.destroy_image(depth_image);
        }

        auto record_loop_task_list() -> daxa::TaskList {
            daxa::TaskList new_task_list = daxa::TaskList({
                .device = device,
                .debug_name = "task_list",
            });
            task_swapchain_image = new_task_list.create_task_image({
                .fetch_callback = [this]() { return swapchain_image; },
                .debug_name = "task_swapchain_image",
            });
            task_color_image = new_task_list.create_task_image({
                .fetch_callback = [this]() { return color_image; },
                .debug_name = "task_color_image",
            });
            task_display_image = new_task_list.create_task_image({
                .fetch_callback = [this]() { return display_image; },
                .debug_name = "task_display_image",
            });
            task_motion_vectors_image = new_task_list.create_task_image({
                .fetch_callback = [this]() { return motion_vectors_image; },
                .debug_name = "task_motion_vectors_image",
            });
            auto &self = *reinterpret_cast<T *>(this);
            self.record_loop_task_list(new_task_list);

            new_task_list.add_task({
                .resources = {
                    .images = {
                        {task_color_image, daxa::TaskImageAccess::TRANSFER_READ},
                        {task_display_image, daxa::TaskImageAccess::TRANSFER_WRITE},
                    },
                },
                .task = [this](daxa::TaskInterface interf) {
                    if (!fsr_enabled) {
                        auto cmd_list = interf.get_command_list();
                        cmd_list.blit_image_to_image({
                            .src_image = color_image,
                            .src_image_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            .dst_image = display_image,
                            .dst_image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                            .src_slice = {.image_aspect = daxa::ImageAspectFlagBits::COLOR},
                            .src_offsets = {{{0, 0, 0}, {static_cast<i32>(render_size_x), static_cast<i32>(render_size_y), 1}}},
                            .dst_slice = {.image_aspect = daxa::ImageAspectFlagBits::COLOR},
                            .dst_offsets = {{{0, 0, 0}, {static_cast<i32>(AppWindow<T>::size_x), static_cast<i32>(AppWindow<T>::size_y), 1}}},
                        });
                    }
                },
                .debug_name = "Blit Task (render to display)",
            });

            new_task_list.add_task({
                .resources = {
                    .images = {
                        {task_color_image, daxa::TaskImageAccess::SHADER_READ_ONLY},
                        {task_motion_vectors_image, daxa::TaskImageAccess::SHADER_READ_ONLY},
                        {task_depth_image, daxa::TaskImageAccess::SHADER_READ_ONLY},
                        {task_display_image, daxa::TaskImageAccess::SHADER_WRITE_ONLY},
                    },
                },
                .task = [this](daxa::TaskInterface interf) {
                    if (fsr_enabled) {
                        auto cmd_list = interf.get_command_list();
                        fsr_context.upscale(
                            cmd_list,
                            {
                                .color = color_image,
                                .depth = depth_image,
                                .motion_vectors = motion_vectors_image,
                                .output = display_image,
                                .should_reset = false,
                                .delta_time = delta_time,
                                .jitter = jitter,
                                .should_sharpen = false,
                                .sharpening = 0.0f,
                                .camera_info = {
                                    .near_plane = 0.01,
                                    .far_plane = 1000.0,
                                    .vertical_fov = deg2rad(fov),
                                },
                            });
                    }
                },
                .debug_name = "TaskList Upscale Task",
            });

            new_task_list.add_task({
                .resources = {
                    .images = {
                        {task_display_image, daxa::TaskImageAccess::TRANSFER_READ},
                        {task_swapchain_image, daxa::TaskImageAccess::TRANSFER_WRITE},
                    },
                },
                .task = [this](daxa::TaskInterface interf) {
                    auto cmd_list = interf.get_command_list();
                    cmd_list.blit_image_to_image({
                        .src_image = display_image,
                        .src_image_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        .dst_image = swapchain_image,
                        .dst_image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                        .src_slice = {.image_aspect = daxa::ImageAspectFlagBits::COLOR},
                        .src_offsets = {{{0, 0, 0}, {static_cast<i32>(AppWindow<T>::size_x), static_cast<i32>(AppWindow<T>::size_y), 1}}},
                        .dst_slice = {.image_aspect = daxa::ImageAspectFlagBits::COLOR},
                        .dst_offsets = {{{0, 0, 0}, {static_cast<i32>(AppWindow<T>::size_x), static_cast<i32>(AppWindow<T>::size_y), 1}}},
                    });
                },
                .debug_name = "Blit Task (display to swapchain)",
            });

            new_task_list.add_task({
                .resources = {
                    .images = {
                        {task_swapchain_image, daxa::TaskImageAccess::COLOR_ATTACHMENT},
                    },
                },
                .task = [this](daxa::TaskInterface interf) {
                    auto cmd_list = interf.get_command_list();
                    imgui_renderer.record_commands(ImGui::GetDrawData(), cmd_list, swapchain_image, AppWindow<T>::size_x, AppWindow<T>::size_y);
                },
                .debug_name = "ImGui Task",
            });
            new_task_list.compile();
            // new_task_list.output_graphviz();
            return new_task_list;
        }

        bool update() {
            glfwPollEvents();
            if (glfwWindowShouldClose(AppWindow<T>::glfw_window_ptr)) {
                return true;
            }

            if (!AppWindow<T>::minimized) {
                auto now = Clock::now();
                time = std::chrono::duration<f32>(now - start).count();
                delta_time = std::chrono::duration<f32>(now - prev_time).count();
                prev_time = now;

                prev_jitter = jitter;
                // jitter = {0, 0};
                jitter = fsr_context.get_jitter(cpu_framecount);

                ImGui_ImplGlfw_NewFrame();

                auto &self = *reinterpret_cast<T *>(this);
                self.on_update();
            } else {
                using namespace std::literals;
                std::this_thread::sleep_for(1ms);
            }

            return false;
        }

        void execute_loop_task_list() {
            swapchain_image = swapchain.acquire_next_image();
            loop_task_list.execute();
            auto command_lists = loop_task_list.command_lists();
            auto cmd_list = device.create_command_list({});
            cmd_list.pipeline_barrier_image_transition({
                .awaited_pipeline_access = loop_task_list.last_access(task_swapchain_image),
                .before_layout = loop_task_list.last_layout(task_swapchain_image),
                .after_layout = daxa::ImageLayout::PRESENT_SRC,
                .image_id = swapchain_image,
            });
            cmd_list.complete();
            ++cpu_framecount;
            command_lists.push_back(cmd_list);
            device.submit_commands({
                .command_lists = command_lists,
                .signal_binary_semaphores = {binary_semaphore},
                .signal_timeline_semaphores = {{gpu_framecount_timeline_sema, cpu_framecount}},
            });
            device.present_frame({
                .wait_binary_semaphores = {binary_semaphore},
                .swapchain = swapchain,
            });
            gpu_framecount_timeline_sema.wait_for_value(cpu_framecount - 1);
        }

        bool try_recreate_pipeline(daxa::ComputePipeline &pipeline) {
            if (pipeline_compiler.check_if_sources_changed(pipeline)) {
                auto new_pipeline = pipeline_compiler.recreate_compute_pipeline(pipeline);
                if (new_pipeline.is_ok()) {
                    pipeline = new_pipeline.value();
                    std::cout << "Shader Compilation SUCCESS!" << std::endl;
                } else {
                    std::cout << new_pipeline.message() << std::endl;
                }
                return true;
            }
            return false;
        }

        bool try_recreate_pipeline(daxa::RasterPipeline &pipeline) {
            if (pipeline_compiler.check_if_sources_changed(pipeline)) {
                auto new_pipeline = pipeline_compiler.recreate_compute_pipeline(pipeline);
                if (new_pipeline.is_ok()) {
                    pipeline = new_pipeline.value();
                    std::cout << "Shader Compilation SUCCESS!" << std::endl;
                } else {
                    std::cout << new_pipeline.message() << std::endl;
                }
                return true;
            }
            return false;
        }

        void toggle_pause() {
            AppWindow<T>::set_mouse_capture(paused);
            paused = !paused;
        }
    };
} // namespace daxa_common
