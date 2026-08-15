#include "voxel_app.hpp"
#include "base/profiler.hpp"
#include "renderer/pipeline_manager.hpp"
#include "renderer/render_voxel_object.hpp"
#include "scene.hpp"

#include <base/format.hpp>

#include <thread>
#include <numbers>
#include <fstream>
#include <unordered_map>

#define APPNAME "Voxel App"

using namespace std::chrono_literals;

#include <iostream>
#include "renderer/render_scene.hpp"
#include <voxels/animation_playground/animation_playground.hpp>

constexpr auto round_frame_dim(daxa_u32vec2 size) {
    auto result = size;
    // constexpr auto over_estimation = daxa_u32vec2{32, 32};
    // auto result = (size + daxa_u32vec2{over_estimation.x - 1u, over_estimation.y - 1u}) / over_estimation * over_estimation;
    // not necessary, since it rounds up!
    // result = {std::max(result.x, over_estimation.x), std::max(result.y, over_estimation.y)};
    return result;
}

VoxelApp::VoxelApp() : AppWindow(APPNAME, {1280, 720}), ui{AppUi(AppWindow::glfw_window_ptr)} {
    PROFILE_FUNC();

    gpu_context.create_swapchain({
        .native_window_info = AppWindow::get_native_window_info(),
        .surface_format = gpu_context.device.choose_swapchain_surface_format({
            .native_window_info = AppWindow::get_native_window_info(),
            .preferred_formats = std::array{daxa::SurfaceFormat{.format = daxa::Format::B8G8R8A8_UNORM}},
        }),
        .present_mode = daxa::PresentMode::FIFO,
        .image_usage = daxa::ImageUsageFlagBits::TRANSFER_DST,
        .max_allowed_frames_in_flight = FRAMES_IN_FLIGHT,
        .name = "my swapchain",
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

    imgui_renderer = daxa::ImGuiRenderer(daxa::ImGuiRendererInfo{
        .device = gpu_context.device,
        .format = gpu_context.swapchain.get_format(),
        .imgui_context = ImGui::GetCurrentContext(),
        .use_custom_config = false,
    });

    scene = new Scene(gpu_context);
    ui.animation_playground = scene->animation_playground;
    ui.profiler_ui = &profiler_ui;

    record_tasks();
}
VoxelApp::~VoxelApp() {
    gpu_context.device.wait_idle();
    gpu_context.device.collect_garbage();
    delete scene;

    // TODO: Remove this
    // gpu_context.device.destroy_tlas(voxel_world.buffers.tlas);
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
    auto now = std::chrono::high_resolution_clock::now();
    profiler_ui.paused = ui.show_profiler_view;
    profiler_ui.update();
    PROFILE_FUNC();

    {
        PROFILE_SCOPE("acquire_next_image");
        gpu_context.swapchain_image = gpu_context.swapchain.acquire_next_image();
    }

    gpu_input.time = std::chrono::duration<daxa_f32>(now - start).count();
    gpu_input.delta_time = std::chrono::duration<daxa_f32>(now - prev_time).count();
    prev_time = now;
    gpu_input.render_res_scl = render_res_scl;

    audio.set_frequency(gpu_input.delta_time * 1000.0f * 200.0f);

    // Hot-reload: the manager's watcher thread flags when any shader source (or
    // any file it #includes) changed on disk; recompiling assigns new pipelines
    // in place, so already-recorded task-graph closures pick them up as-is.
    if (needs_hot_reload(gpu_context.pipeline_manager)) {
        auto reload_result = try_hot_reload(gpu_context.pipeline_manager, gpu_context.device, false);
        if (reload_result == RELOAD_ERROR) {
            debug_utils::Console::add_log("shader hot-reload failed; see log for details");
        }
        if (reload_result != RELOAD_NO_CHANGE) {
            // The SBT references the old pipeline's shader groups, so it has to
            // be rebuilt against the freshly created ray tracing pipelines.
            for (auto &slot : gpu_context.ray_tracing_pipelines) {
                slot.value->recreate_sbt();
            }
            scene->animation_playground->dirty = true;
        }
    }

    gpu_context.task_swapchain_image.set_image(gpu_context.swapchain_image);
    if (gpu_context.swapchain_image.is_empty()) {
        return;
    }

    if (ui.should_upload_seed_data) {
        gpu_context.update_seeded_value_noise(hash_key(ui.settings.world_seed_str));
        ui.should_upload_seed_data = false;
    }

    if (ui.should_run_startup) {
        run_startup();
        ui.should_run_startup = false;
    }

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

    if (!ui.show_profiler_view) {
        player_input.frame_dim = gpu_input.frame_dim;
        player_input.halton_jitter = gpu_input.halton_jitter;
        player_input.delta_time = gpu_input.delta_time;
        player_input.sensitivity = ui.settings.mouse_sensitivity;
        player_input.fov = AppSettings::get<settings::SliderFloat>("Camera", "FOV").value * (std::numbers::pi_v<daxa_f32> / 180.0f);
        player_input.mouse = gpu_input.mouse;
        std::copy(std::begin(gpu_input.actions), std::end(gpu_input.actions), std::begin(player_input.actions));
        player_perframe(player_input, gpu_input.player);

        scene->update(renderer, gpu_input);
    }

    {
        PROFILE_SCOPE("frame_task_graph.execute()");
        gpu_input.fif_index = gpu_input.frame_index % (FRAMES_IN_FLIGHT + 1);
        gpu_context.frame_task_graph.execute({});
    }

    gpu_input.resize_factor = 1.0f;

    gpu_input.mouse.pos_delta = {0.0f, 0.0f};
    gpu_input.mouse.scroll_delta = {0.0f, 0.0f};

    renderer.end_frame(gpu_context.device, gpu_input.delta_time);

    ui.update();

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

    if (auto *action_index = ui.settings.mouse_button_binds.get(button_id)) {
        gpu_input.actions[*action_index] = static_cast<daxa_u32>(action);
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

    if (key_id == GLFW_KEY_F7 && action == GLFW_PRESS) {
        ui.toggle_profiler_view();
        set_mouse_capture(!ui.paused);
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
        if (auto *action_index = ui.settings.keybinds.get(key_id)) {
            gpu_input.actions[*action_index] = static_cast<daxa_u32>(action);
        }
    }
}
void VoxelApp::on_resize(daxa_u32 sx, daxa_u32 sy) {
    minimized = (sx == 0 || sy == 0);
    auto new_render_res_scl = AppSettings::get<settings::SliderFloat>("Graphics", "Render Res Scale").value;
    auto resized = sx != window_size.x || sy != window_size.y || render_res_scl != new_render_res_scl;
    if (!minimized && resized) {
        {
            PROFILE_SCOPE("resize");
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
        }
        on_update();
    }
}
void VoxelApp::on_drop(char const *const *filepaths, int filepath_count) {
    if (filepath_count <= 0)
        return;
    ui.gvox_model_path = filepaths[0];
    ui.should_upload_gvox_model = true;
}

void VoxelApp::run_startup() {
    player_startup(gpu_input.player);
    // gpu_context.startup_task_graph.execute({});

    ui.should_run_startup = false;
}

#define GVOX_ENGINE_INSTALL false

void VoxelApp::record_tasks() {
    PROFILE_FUNC();
    ui.should_record_task_graph = false;
    gpu_context.task_states.clear();
    gpu_context.task_states.reserve(500);

    gpu_input.frame_dim.x = static_cast<daxa_u32>(static_cast<daxa_f32>(window_size.x) * render_res_scl);
    gpu_input.frame_dim.y = static_cast<daxa_u32>(static_cast<daxa_f32>(window_size.y) * render_res_scl);
    gpu_input.rounded_frame_dim = round_frame_dim(gpu_input.frame_dim);
    gpu_input.output_resolution = window_size;

    gpu_context.frame_task_graph = daxa::TaskGraph({
        .device = gpu_context.device,
        .swapchain = gpu_context.swapchain,
        .alias_transients = GVOX_ENGINE_INSTALL,
        .staging_memory_pool_size = 1 << 20,
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

    // voxel_world.record_startup(gpu_context);
    // particles.record_startup(gpu_context);

    debug_utils::DebugDisplay::begin_passes();

    gpu_context.frame_task_graph.add_task(
        daxa::InlineTask::Transfer("GpuInputUploadTransferTask")
            .writes(gpu_context.task_input_buffer.view())
            .executes([this](daxa::TaskInterface ti) {
                auto staging_input_buffer = ti.device.create_buffer({
                    .size = sizeof(GpuInput),
                    .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "staging_input_buffer",
                });
                ti.recorder.destroy_buffer_deferred(staging_input_buffer);
                auto *buffer_ptr = ti.device.buffer_host_address_as<GpuInput>(staging_input_buffer).value();
                *buffer_ptr = gpu_input;
                ti.recorder.copy_buffer_to_buffer({
                    .src_buffer = staging_input_buffer,
                    .dst_buffer = gpu_context.task_input_buffer.id(),
                    .size = sizeof(GpuInput),
                });
            }));

    // voxel_world.record_frame(gpu_context, particles);
    // particles.simulate(gpu_context, voxel_world.buffers);
    record_render_scene(gpu_context, scene->render_scene);

    renderer.render(gpu_context, scene->render_scene, gpu_context.task_swapchain_image, gpu_context.swapchain.get_format());

    gpu_context.frame_task_graph.add_task(
        daxa::InlineTask("ImGui Draw")
            .color_attachment.reads_writes(gpu_context.task_swapchain_image)
            .executes(
                [this](daxa::TaskInterface const &ti) {
                    auto swapchain_image = gpu_context.task_swapchain_image.info().image;
                    auto size = ti.info(gpu_context.task_swapchain_image.view()).value().size;
                    imgui_renderer.record_commands({ImGui::GetDrawData(), ti.recorder, swapchain_image, size.x, size.y});
                }));
    gpu_context.frame_task_graph.submit({});
    gpu_context.frame_task_graph.present({});
    gpu_context.frame_task_graph.complete({});

    // gpu_context.startup_task_graph.submit({});
    // gpu_context.startup_task_graph.complete({});

    // Recording the graph above only *registered* every pipeline; compile them
    // all now, in parallel and in one batch, then create the pipeline objects.
    // Everything is ready by the time the first frame executes, so tasks never
    // have to skip themselves waiting on an in-flight compile.
    Clock::time_point shader_start = Clock::now();
    compile_all_shaders(gpu_context.pipeline_manager);
    create_all_pipelines(gpu_context.pipeline_manager, gpu_context.device);
    debug_utils::Console::add_log(format("compiling shaders: %f s\n", double(std::chrono::duration<float>(Clock::now() - shader_start).count())).data);

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
    Vec<debug_utils::DebugDisplay::GpuResourceInfo> &debug_gpu_resource_infos = debug_utils::DebugDisplay::s_instance->gpu_resource_infos;

    debug_gpu_resource_infos.clear();

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
        auto image_info = gpu_context.device.image_info(image).value();
        auto size = format_to_pixel_size(image_info.format) * image_info.size.x * image_info.size.y * image_info.size.z;
        debug_gpu_resource_infos.push_back({
            .type = "image",
            .name = image_info.name.data(),
            .size = size,
        });
        result_size += size;
    };
    auto buffer_size = [this, &result_size, &debug_gpu_resource_infos](daxa::BufferId buffer, bool individual = true) -> size_t {
        if (buffer.is_empty()) {
            return 0;
        }
        auto buffer_info = gpu_context.device.buffer_info(buffer).value();
        if (individual) {
            debug_gpu_resource_infos.push_back({
                .type = "buffer",
                .name = buffer_info.name.data(),
                .size = buffer_info.size,
            });
        }
        result_size += buffer_info.size;
        return buffer_info.size;
    };

    buffer_size(gpu_context.input_buffer);

    for (auto &slot : gpu_context.temporal_buffers) {
        buffer_size(slot.value.task_resource.id());
    }
    for (auto &slot : gpu_context.temporal_images) {
        image_size(slot.value.task_resource.id());
    }

#if defined(VOXELS_ORIGINAL_IMPL)
    // buffer_size(voxel_world.buffers.blas_attr_pointers.task_resource.id());
    // buffer_size(voxel_world.buffers.blas_geom_pointers.task_resource.id());
    // buffer_size(voxel_world.buffers.blas_transforms.task_resource.id());
    // buffer_size(voxel_world.buffers.voxel_bricks.task_resource.id());
    // buffer_size(voxel_world.buffers.voxel_globals.task_resource.id());
    // buffer_size(voxel_world.buffers.brick_update_heap.task_resource.id());
    // buffer_size(voxel_world.buffers.brick_updates.task_resource.id());
    // auto total_tlas_size = buffer_size(voxel_world.buffers.tlas_buffer);
    // auto total_blas_size = size_t{};
    // auto total_attr_size = size_t{};
    // auto total_geom_size = size_t{};
    // auto total_non_empty_blas_count = size_t{};
    // auto total_geom_count = size_t{};
    // for (auto const &blas_brick : voxel_world.blas_bricks) {
    //     total_blas_size += buffer_size(blas_brick.blas_buffer, false);
    //     total_attr_size += buffer_size(blas_brick.attr_buffer, false);
    //     total_geom_size += buffer_size(blas_brick.geom_buffer, false);
    //     if (!blas_brick.blas_geoms.empty()) {
    //         ++total_non_empty_blas_count;
    //         total_geom_count += blas_brick.blas_geoms.size();
    //     }
    // }
    // debug_utils::DebugDisplay::set_debug_string("total_tlas_size", fmt::format("{:.3f} MB", static_cast<float>(total_tlas_size) / 1000000));
    // debug_utils::DebugDisplay::set_debug_string("total_blas_size", fmt::format("{:.3f} MB ({:.3f} KB/blas)", static_cast<float>(total_blas_size) / 1000000, static_cast<float>(total_blas_size) / total_non_empty_blas_count / 1000));
    // debug_utils::DebugDisplay::set_debug_string("total_attr_size", fmt::format("{:.3f} MB ({:.3f} KB/blas)", static_cast<float>(total_attr_size) / 1000000, static_cast<float>(total_attr_size) / total_non_empty_blas_count / 1000));
    // debug_utils::DebugDisplay::set_debug_string("total_geom_size", fmt::format("{:.3f} MB ({:.3f} KB/blas)", static_cast<float>(total_geom_size) / 1000000, static_cast<float>(total_geom_size) / total_non_empty_blas_count / 1000));
    // debug_utils::DebugDisplay::set_debug_string("total_geom_count", fmt::format("{}", total_geom_count));
    // debug_utils::DebugDisplay::set_debug_string("total_blas_count", fmt::format("{}", total_non_empty_blas_count));
    // debug_utils::DebugDisplay::set_debug_string("avg #geom per blas", fmt::format("{:.3f}", float(total_geom_count) / float(total_non_empty_blas_count)));

#endif

    {
        auto size = gpu_context.frame_task_graph.get_resource_memory_block_size();
        debug_gpu_resource_infos.push_back({
            .type = "buffer",
            .name = "Per-frame Transient Memory Buffer",
            .size = size,
        });
        result_size += size;
    }

    needs_vram_calc = false;

    debug_utils::DebugDisplay::set_debug_string("Est. VRAM usage", format("%.3f MB", double(static_cast<float>(result_size) / 1000000)).data);
}
