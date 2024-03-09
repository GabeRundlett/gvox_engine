#include "voxel_app.hpp"

#include <fmt/format.h>

#include <thread>
#include <numbers>
#include <fstream>
#include <unordered_map>

// #include <voxels/gvox_model.inl>

static_assert(IsVoxelWorld<VoxelWorld>);

#define APPNAME "Voxel App"

using namespace std::chrono_literals;

#include <iostream>

constexpr auto round_frame_dim(daxa_u32vec2 size) {
    auto result = size;
    // constexpr auto over_estimation = daxa_u32vec2{32, 32};
    // auto result = (size + daxa_u32vec2{over_estimation.x - 1u, over_estimation.y - 1u}) / over_estimation * over_estimation;
    // not necessary, since it rounds up!
    // result = {std::max(result.x, over_estimation.x), std::max(result.y, over_estimation.y)};
    return result;
}

VoxelApp::VoxelApp() : AppWindow(APPNAME, {1280, 720}), ui{AppUi(AppWindow::glfw_window_ptr)} {
    gpu_context.create_swapchain({
        .native_window = AppWindow::get_native_handle(),
        .native_window_platform = AppWindow::get_native_platform(),
        .surface_format_selector = [](daxa::Format format) -> daxa_i32 {
            switch (format) {
            case daxa::Format::B8G8R8A8_SRGB: return 90;
            case daxa::Format::R8G8B8A8_SRGB: return 80;
            default: return 0;
            }
        },
        .present_mode = daxa::PresentMode::IMMEDIATE,
        .image_usage = daxa::ImageUsageFlagBits::TRANSFER_DST,
        .max_allowed_frames_in_flight = FRAMES_IN_FLIGHT,
        .name = "swapchain",
    });

    AppSettings::add<settings::SliderFloat>({"Camera", "FOV", {.value = 74.0f, .min = 0.0f, .max = 179.0f}});

    AppSettings::add<settings::InputFloat>({"UI", "Scale", {.value = 1.0f}});
    AppSettings::add<settings::Checkbox>({"UI", "show_debug_info", {.value = false}});
    AppSettings::add<settings::Checkbox>({"UI", "show_console", {.value = false}});
    AppSettings::add<settings::Checkbox>({"UI", "autosave", {.value = true}});
    AppSettings::add<settings::Checkbox>({"General", "battery_saving_mode", {.value = false}});

    AppSettings::add<settings::SliderFloat>({"Graphics", "Render Res Scale", {.value = 1.0f, .min = 0.2f, .max = 4.0f}, {.task_graph_depends = true}});

    auto const &device_props = gpu_context.device.properties();
    debug_utils::DebugDisplay::set_debug_string("GPU", reinterpret_cast<char const *>(device_props.device_name));
    imgui_renderer = daxa::ImGuiRenderer({
        .device = gpu_context.device,
        .format = gpu_context.swapchain.get_format(),
        .context = ImGui::GetCurrentContext(),
        .use_custom_config = false,
    });

    voxel_model_loader.create(gpu_context);

    record_tasks();
    gpu_context.pipeline_manager->wait();
    debug_utils::Console::add_log(fmt::format("startup: {} s\n", std::chrono::duration<float>(Clock::now() - start).count()));
}
VoxelApp::~VoxelApp() {
    gpu_context.device.wait_idle();
    gpu_context.device.collect_garbage();

    voxel_model_loader.destroy();
}

void VoxelApp::run() {
    while (true) {
        glfwPollEvents();
        if (glfwWindowShouldClose(AppWindow::glfw_window_ptr) != 0) {
            break;
        }

        if (!AppWindow::minimized) {
            on_resize(window_size.x, window_size.y);

            if (AppSettings::get<settings::Checkbox>("General", "battery_saving_mode").value) {
                std::this_thread::sleep_for(10ms);
            }

            on_update();
        } else {
            std::this_thread::sleep_for(1ms);
        }
    }
}

void VoxelApp::on_update() {
    auto now = Clock::now();

    gpu_context.swapchain_image = gpu_context.swapchain.acquire_next_image();

    auto t0 = Clock::now();
    gpu_input.time = std::chrono::duration<daxa_f32>(now - start).count();
    gpu_input.delta_time = std::chrono::duration<daxa_f32>(now - prev_time).count();
    prev_time = now;
    gpu_input.render_res_scl = render_res_scl;

    audio.set_frequency(gpu_input.delta_time * 1000.0f * 200.0f);

    if (ui.should_hotload_shaders) {
        auto reload_result = gpu_context.pipeline_manager->reload_all();
        if (auto *reload_err = daxa::get_if<daxa::PipelineReloadError>(&reload_result)) {
            debug_utils::Console::add_log(reload_err->message);
        }
    }

    gpu_context.task_swapchain_image.set_images({.images = {&gpu_context.swapchain_image, 1}});
    if (gpu_context.swapchain_image.is_empty()) {
        return;
    }

    if (ui.should_upload_seed_data) {
        gpu_context.update_seeded_value_noise(gpu_context.device, std::hash<std::string>{}(ui.settings.world_seed_str));
        ui.should_upload_seed_data = false;
    }

    if (ui.should_run_startup || voxel_model_loader.model_is_ready) {
        run_startup();
        ui.should_run_startup = false;
    }

    voxel_model_loader.update(ui);

    if (ui.should_record_task_graph) {
        gpu_context.device.wait_idle();
        record_tasks();
    }

    gpu_input.flags &= ~GAME_FLAG_BITS_PAUSED;
    gpu_input.flags |= GAME_FLAG_BITS_PAUSED * static_cast<daxa_u32>(ui.paused);

    gpu_input.flags &= ~GAME_FLAG_BITS_NEEDS_PHYS_UPDATE;

    renderer.begin_frame(gpu_input);

    if (now - prev_phys_update_time > std::chrono::duration<float>(GAME_PHYS_UPDATE_DT)) {
        gpu_input.flags |= GAME_FLAG_BITS_NEEDS_PHYS_UPDATE;
        prev_phys_update_time = now;
    }

    if (needs_vram_calc) {
        calc_vram_usage();
    }

    voxel_world.begin_frame(gpu_context.device, gpu_output.voxel_world);

    player_input.frame_dim = gpu_input.frame_dim;
    player_input.halton_jitter = gpu_input.halton_jitter;
    player_input.delta_time = gpu_input.delta_time;
    player_input.sensitivity = ui.settings.mouse_sensitivity;
    player_input.fov = AppSettings::get<settings::SliderFloat>("Camera", "FOV").value * (std::numbers::pi_v<daxa_f32> / 180.0f);
    player_input.mouse = gpu_input.mouse;
    std::copy(std::begin(gpu_input.actions), std::end(gpu_input.actions), std::begin(player_input.actions));
    player_perframe(player_input, gpu_input.player);

    gpu_input.fif_index = gpu_input.frame_index % (FRAMES_IN_FLIGHT + 1);
    gpu_context.frame_task_graph.execute({});

    gpu_input.resize_factor = 1.0f;

    gpu_input.mouse.pos_delta = {0.0f, 0.0f};
    gpu_input.mouse.scroll_delta = {0.0f, 0.0f};

    renderer.end_frame(gpu_context.device, gpu_input.delta_time);

    auto t1 = Clock::now();
    ui.update(gpu_input.delta_time, std::chrono::duration<daxa_f32>(t1 - t0).count());

    ++gpu_input.frame_index;
    gpu_context.device.collect_garbage();
}
void VoxelApp::on_mouse_move(daxa_f32 x, daxa_f32 y) {
    daxa_f32vec2 const center = {static_cast<daxa_f32>(window_size.x / 2), static_cast<daxa_f32>(window_size.y / 2)};
    gpu_input.mouse.pos = daxa_f32vec2{x, y};
    auto offset = daxa_f32vec2{gpu_input.mouse.pos.x - center.x, gpu_input.mouse.pos.y - center.y};
    gpu_input.mouse.pos = daxa_f32vec2{
        gpu_input.mouse.pos.x * static_cast<daxa_f32>(gpu_input.frame_dim.x) / static_cast<daxa_f32>(window_size.x),
        gpu_input.mouse.pos.y * static_cast<daxa_f32>(gpu_input.frame_dim.y) / static_cast<daxa_f32>(window_size.y),
    };
    if (!ui.paused) {
        gpu_input.mouse.pos_delta = daxa_f32vec2{gpu_input.mouse.pos_delta.x + offset.x, gpu_input.mouse.pos_delta.y + offset.y};
        set_mouse_pos(center.x, center.y);
    }
}
void VoxelApp::on_mouse_scroll(daxa_f32 dx, daxa_f32 dy) {
    auto &io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        return;
    }

    gpu_input.mouse.scroll_delta = daxa_f32vec2{gpu_input.mouse.scroll_delta.x + dx, gpu_input.mouse.scroll_delta.y + dy};
}
void VoxelApp::on_mouse_button(daxa_i32 button_id, daxa_i32 action) {
    auto &io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        return;
    }
    if (ui.limbo_action_index != INVALID_GAME_ACTION) {
        return;
    }

    if (ui.settings.mouse_button_binds.contains(button_id)) {
        gpu_input.actions[ui.settings.mouse_button_binds.at(button_id)] = static_cast<daxa_u32>(action);
    }
}
void VoxelApp::on_key(daxa_i32 key_id, daxa_i32 action) {
    auto &io = ImGui::GetIO();
    if (io.WantCaptureKeyboard) {
        return;
    }
    if (ui.limbo_action_index != INVALID_GAME_ACTION) {
        return;
    }

    if (key_id == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        std::fill(std::begin(gpu_input.actions), std::end(gpu_input.actions), 0);
        ui.toggle_pause();
        set_mouse_capture(!ui.paused);
    }

    if (key_id == GLFW_KEY_F3 && action == GLFW_PRESS) {
        ui.toggle_debug();
    }

    if (ui.paused) {
        if (key_id == GLFW_KEY_GRAVE_ACCENT && action == GLFW_PRESS) {
            ui.toggle_console();
        }
    }

    if (key_id == GLFW_KEY_R && action == GLFW_PRESS) {
        ui.should_run_startup = true;
        start = Clock::now();
    }

    if (!ui.paused) {
        if (ui.settings.keybinds.contains(key_id)) {
            gpu_input.actions[ui.settings.keybinds.at(key_id)] = static_cast<daxa_u32>(action);
        }
    }
}
void VoxelApp::on_resize(daxa_u32 sx, daxa_u32 sy) {
    minimized = (sx == 0 || sy == 0);
    auto new_render_res_scl = AppSettings::get<settings::SliderFloat>("Graphics", "Render Res Scale").value;
    auto resized = sx != window_size.x || sy != window_size.y || render_res_scl != new_render_res_scl;
    if (!minimized && resized) {
        gpu_context.swapchain.resize();
        window_size.x = gpu_context.swapchain.get_surface_extent().x;
        window_size.y = gpu_context.swapchain.get_surface_extent().y;
        render_res_scl = new_render_res_scl;
        {
            // resize render images
            // gpu_context.render_images.size.x = static_cast<daxa_u32>(static_cast<daxa_f32>(window_size.x) * render_res_scl);
            // gpu_context.render_images.size.y = static_cast<daxa_u32>(static_cast<daxa_f32>(window_size.y) * render_res_scl);
            gpu_context.device.wait_idle();
            needs_vram_calc = true;
        }
        record_tasks();
        gpu_input.resize_factor = 0.0f;
        on_update();
    }
}
void VoxelApp::on_drop(std::span<char const *> filepaths) {
    ui.gvox_model_path = filepaths[0];
    ui.should_upload_gvox_model = true;
}

void VoxelApp::run_startup() {
    player_startup(gpu_input.player);
    gpu_context.startup_task_graph.execute({});

    ui.should_run_startup = false;
}

void VoxelApp::record_tasks() {
    ui.should_record_task_graph = false;

    gpu_input.frame_dim.x = static_cast<daxa_u32>(static_cast<daxa_f32>(window_size.x) * render_res_scl);
    gpu_input.frame_dim.y = static_cast<daxa_u32>(static_cast<daxa_f32>(window_size.y) * render_res_scl);
    gpu_input.rounded_frame_dim = round_frame_dim(gpu_input.frame_dim);
    gpu_input.output_resolution = window_size;

    gpu_context.frame_task_graph = daxa::TaskGraph({
        .device = gpu_context.device,
        .swapchain = gpu_context.swapchain,
        .alias_transients = GVOX_ENGINE_INSTALL,
        .name = "frame_task_graph",
    });
    gpu_context.startup_task_graph = daxa::TaskGraph({
        .device = gpu_context.device,
        .alias_transients = GVOX_ENGINE_INSTALL,
        .name = "startup_task-graph",
    });
    gpu_context.use_resources();
    gpu_context.render_resolution = gpu_input.rounded_frame_dim;
    gpu_context.output_resolution = gpu_input.output_resolution;

    voxel_world.record_startup(gpu_context);
    particles.record_startup(gpu_context);

    gpu_context.frame_task_graph.use_persistent_buffer(voxel_model_loader.task_gvox_model_buffer);

    gpu_context.frame_task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, gpu_context.task_input_buffer),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            auto staging_input_buffer = ti.device.create_buffer({
                .size = sizeof(GpuInput),
                .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = "staging_input_buffer",
            });
            ti.recorder.destroy_buffer_deferred(staging_input_buffer);
            auto *buffer_ptr = ti.device.get_host_address_as<GpuInput>(staging_input_buffer).value();
            *buffer_ptr = gpu_input;
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer = staging_input_buffer,
                .dst_buffer = gpu_context.task_input_buffer.get_state().buffers[0],
                .size = sizeof(GpuInput),
            });
        },
        .name = "GpuInputUploadTransferTask",
    });

    voxel_world.record_frame(gpu_context, voxel_model_loader.task_gvox_model_buffer, gpu_context.task_value_noise_image);
    particles.simulate(gpu_context, voxel_world.buffers);

    renderer.render(gpu_context, voxel_world.buffers, particles, gpu_context.task_swapchain_image, gpu_context.swapchain.get_format());

    gpu_context.frame_task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_READ, gpu_context.task_output_buffer),
            daxa::inl_attachment(daxa::TaskBufferAccess::HOST_TRANSFER_WRITE, gpu_context.task_staging_output_buffer),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            auto output_buffer = gpu_context.task_output_buffer.get_state().buffers[0];
            auto staging_output_buffer = gpu_context.staging_output_buffer;
            auto frame_index = gpu_input.frame_index + 1;
            auto *buffer_ptr = ti.device.get_host_address_as<std::array<GpuOutput, (FRAMES_IN_FLIGHT + 1)>>(staging_output_buffer).value();
            daxa_u32 const offset = frame_index % (FRAMES_IN_FLIGHT + 1);
            gpu_output = (*buffer_ptr)[offset];
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer = output_buffer,
                .dst_buffer = staging_output_buffer,
                .size = sizeof(GpuOutput) * (FRAMES_IN_FLIGHT + 1),
            });
        },
        .name = "GpuOutputDownloadTransferTask",
    });

    gpu_context.frame_task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskImageAccess::COLOR_ATTACHMENT, daxa::ImageViewType::REGULAR_2D, gpu_context.task_swapchain_image),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            imgui_renderer.record_commands(ImGui::GetDrawData(), ti.recorder, gpu_context.swapchain_image, window_size.x, window_size.y);
        },
        .name = "ImGui draw",
    });

    gpu_context.frame_task_graph.submit({});
    gpu_context.frame_task_graph.present({});
    gpu_context.frame_task_graph.complete({});

    gpu_context.startup_task_graph.submit({});
    gpu_context.startup_task_graph.complete({});

    needs_vram_calc = true;
}

// void VoxelApp::gpu_app_draw_ui() {
//     for (auto const &str : ui_strings) {
//         ImGui::Text("%s", str.c_str());
//     }
//     if (ImGui::TreeNode("Player")) {
//         ImGui::Text("pos: %.2f, %.2f, %.2f", static_cast<double>(gpu_output.player_pos.x), static_cast<double>(gpu_output.player_pos.y), static_cast<double>(gpu_output.player_pos.z));
//         ImGui::Text("y/p/r: %.2f, %.2f, %.2f", static_cast<double>(gpu_output.player_rot.x), static_cast<double>(gpu_output.player_rot.y), static_cast<double>(gpu_output.player_rot.z));
//         ImGui::Text("unit offs: %.2f, %.2f, %.2f", static_cast<double>(gpu_output.player_unit_offset.x), static_cast<double>(gpu_output.player_unit_offset.y), static_cast<double>(gpu_output.player_unit_offset.z));
//         ImGui::TreePop();
//     }
//     if (ImGui::TreeNode("Auto-Exposure")) {
//         ImGui::Text("Exposure multiple: %.2f", static_cast<double>(gpu_input.pre_exposure));
//         auto hist_float = std::array<float, LUMINANCE_HISTOGRAM_BIN_COUNT>{};
//         auto hist_min = static_cast<float>(kajiya_renderer.post_processor.histogram[0]);
//         auto hist_max = static_cast<float>(kajiya_renderer.post_processor.histogram[0]);
//         auto first_bin_with_value = -1;
//         auto last_bin_with_value = -1;
//         for (uint32_t i = 0; i < LUMINANCE_HISTOGRAM_BIN_COUNT; ++i) {
//             if (first_bin_with_value == -1 && kajiya_renderer.post_processor.histogram[i] != 0) {
//                 first_bin_with_value = i;
//             }
//             if (kajiya_renderer.post_processor.histogram[i] != 0) {
//                 last_bin_with_value = i;
//             }
//             hist_float[i] = static_cast<float>(kajiya_renderer.post_processor.histogram[i]);
//             hist_min = std::min(hist_min, hist_float[i]);
//             hist_max = std::max(hist_max, hist_float[i]);
//         }
//         ImGui::PlotHistogram("Histogram", hist_float.data(), static_cast<int>(hist_float.size()), 0, "hist", hist_min, hist_max, ImVec2(0, 120.0f));
//         ImGui::Text("min %.2f | max %.2f", static_cast<double>(hist_min), static_cast<double>(hist_max));
//         auto a = double(first_bin_with_value) / 256.0 * (LUMINANCE_HISTOGRAM_MAX_LOG2 - LUMINANCE_HISTOGRAM_MIN_LOG2) + LUMINANCE_HISTOGRAM_MIN_LOG2;
//         auto b = double(last_bin_with_value) / 256.0 * (LUMINANCE_HISTOGRAM_MAX_LOG2 - LUMINANCE_HISTOGRAM_MIN_LOG2) + LUMINANCE_HISTOGRAM_MIN_LOG2;
//         ImGui::Text("first bin %d (%.2f) | last bin %d (%.2f)", first_bin_with_value, exp2(a), last_bin_with_value, exp2(b));
//         ImGui::TreePop();
//     }
// }
void VoxelApp::calc_vram_usage() {
    std::vector<debug_utils::DebugDisplay::GpuResourceInfo> &debug_gpu_resource_infos = debug_utils::DebugDisplay::s_instance->gpu_resource_infos;

    debug_gpu_resource_infos.clear();
    ui_strings.clear();

    size_t result_size = 0;

    auto format_to_pixel_size = [](daxa::Format format) -> daxa_u32 {
        switch (format) {
        case daxa::Format::R16G16B16_SFLOAT: return 3 * 2;
        case daxa::Format::R16G16B16A16_SFLOAT: return 4 * 2;
        case daxa::Format::R32G32B32_SFLOAT: return 3 * 4;
        default:
        case daxa::Format::R32G32B32A32_SFLOAT: return 4 * 4;
        }
    };

    auto image_size = [this, &format_to_pixel_size, &result_size, &debug_gpu_resource_infos](daxa::ImageId image) {
        if (image.is_empty()) {
            return;
        }
        auto image_info = gpu_context.device.info_image(image).value();
        auto size = format_to_pixel_size(image_info.format) * image_info.size.x * image_info.size.y * image_info.size.z;
        debug_gpu_resource_infos.push_back({
            .type = "image",
            .name = image_info.name.data(),
            .size = size,
        });
        result_size += size;
    };
    auto buffer_size = [this, &result_size, &debug_gpu_resource_infos](daxa::BufferId buffer) {
        if (buffer.is_empty()) {
            return;
        }
        auto buffer_info = gpu_context.device.info_buffer(buffer).value();
        debug_gpu_resource_infos.push_back({
            .type = "buffer",
            .name = buffer_info.name.data(),
            .size = buffer_info.size,
        });
        result_size += buffer_info.size;
    };

    buffer_size(gpu_context.input_buffer);

    for (auto &[name, temporal_buffer] : gpu_context.temporal_buffers) {
        buffer_size(temporal_buffer.resource_id);
    }
    for (auto &[name, temporal_image] : gpu_context.temporal_images) {
        image_size(temporal_image.resource_id);
    }

#if defined(VOXELS_ORIGINAL_IMPL)
    buffer_size(voxel_world.buffers.voxel_malloc.allocator_buffer);
    buffer_size(voxel_world.buffers.voxel_malloc.element_buffer);
    buffer_size(voxel_world.buffers.voxel_malloc.available_element_stack_buffer);
    buffer_size(voxel_world.buffers.voxel_malloc.released_element_stack_buffer);
#endif

    {
        auto size = gpu_context.frame_task_graph.get_transient_memory_size();
        debug_gpu_resource_infos.push_back({
            .type = "buffer",
            .name = "Per-frame Transient Memory Buffer",
            .size = size,
        });
        result_size += size;
    }

    needs_vram_calc = false;

    ui_strings.push_back(fmt::format("Est. VRAM usage: {} MB", static_cast<float>(result_size) / 1000000));
}
