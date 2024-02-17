#include "voxel_app.hpp"

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

// Code flow
// VoxelApp::VoxelApp()
// GpuResources::create()
// VoxelApp::record_main_task_graph()
// VoxelApp::run()
// VoxelApp::on_update()

// [App initialization]
// Creates daxa instance, device, swapchain, pipeline manager
// Creates ui and imgui_renderer
// Creates task states
// Creates GPU Resources: GpuResources::create()
// Creates main task graph: VoxelApp::record_main_task_graph()
// Creates GVOX Context (gvox_ctx)
// Creates temp task graph
VoxelApp::VoxelApp()
    : AppWindow(APPNAME, {1280, 720}),
      daxa_instance{daxa::create_instance({})},
      device{daxa_instance.create_device({
          .flags = daxa::DeviceFlags2{
              // .buffer_device_address_capture_replay_bit = false,
              // .conservative_rasterization = true,
          },
          .name = "device",
      })},
      swapchain{device.create_swapchain({
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
      })},
      main_pipeline_manager{std::make_shared<AsyncPipelineManager>(daxa::PipelineManagerInfo{
          .device = device,
          .shader_compile_options = {
              .root_paths = {
                  DAXA_SHADER_INCLUDE_DIR,
                  "assets",
                  "src",
                  "gpu",
                  "src/gpu",
                  "src/renderer",
              },
              // .write_out_preprocessed_code = ".out/",
              // .write_out_shader_binary = ".out/",
              .spirv_cache_folder = ".out/spirv_cache",
              .language = daxa::ShaderLanguage::GLSL,
              .enable_debug_info = true,
          },
          .register_null_pipelines_when_first_compile_fails = true,
          .name = "pipeline_manager",
      })},
      ui{AppUi(AppWindow::glfw_window_ptr)} {

    AppSettings::add<settings::SliderFloat>({"Camera", "FOV", {.value = 74.0f, .min = 0.0f, .max = 179.0f}});

    AppSettings::add<settings::InputFloat>({"UI", "Scale", {.value = 1.0f}});
    AppSettings::add<settings::Checkbox>({"UI", "show_debug_info", {.value = false}});
    AppSettings::add<settings::Checkbox>({"UI", "show_console", {.value = false}});
    AppSettings::add<settings::Checkbox>({"UI", "autosave", {.value = true}});
    AppSettings::add<settings::Checkbox>({"General", "battery_saving_mode", {.value = false}});

    AppSettings::add<settings::SliderFloat>({"Graphics", "Render Res Scale", {.value = 1.0f, .min = 0.2f, .max = 4.0f}, {.task_graph_depends = true}});

    auto const &device_props = device.properties();
    ui.debug_gpu_name = reinterpret_cast<char const *>(device_props.device_name);
    imgui_renderer = daxa::ImGuiRenderer({
        .device = device,
        .format = swapchain.get_format(),
        .context = ImGui::GetCurrentContext(),
        .use_custom_config = false,
    });

    renderer.create(device);
    gpu_resources.create(device);
    voxel_world.create(device);
    particles.create(device);
    voxel_model_loader.create(device);
    voxel_model_loader.main_pipeline_manager = main_pipeline_manager;

    // constexpr auto IMMEDIATE_LOAD_MODEL_FROM_GABES_DRIVE = false;
    // if constexpr (IMMEDIATE_LOAD_MODEL_FROM_GABES_DRIVE) {
    //     // ui.gvox_model_path = "C:/Users/gabe/AppData/Roaming/GabeVoxelGame/models/building.vox";
    //     // ui.gvox_model_path = "C:/dev/models/half-life/test.dae";
    //     ui.gvox_model_path = "C:/dev/models/Bistro_v5_2/BistroExterior.fbx";
    //     gvox_model_data = load_gvox_data();
    //     if (gvox_model_data.size != 0) {
    //         model_is_ready = true;
    //         prev_gvox_model_buffer = gpu_resources.gvox_model_buffer;
    //         gpu_resources.gvox_model_buffer = device.create_buffer({
    //             .size = static_cast<daxa_u32>(gvox_model_data.size),
    //             .name = "gvox_model_buffer",
    //         });
    //         task_gvox_model_buffer.set_buffers({.buffers = std::array{gpu_resources.gvox_model_buffer}});
    //     }
    // }

    auto radical_inverse = [](daxa_u32 n, daxa_u32 base) -> daxa_f32 {
        auto val = 0.0f;
        auto inv_base = 1.0f / static_cast<daxa_f32>(base);
        auto inv_bi = inv_base;
        while (n > 0) {
            auto d_i = n % base;
            val += static_cast<daxa_f32>(d_i) * inv_bi;
            n = static_cast<daxa_u32>(static_cast<daxa_f32>(n) * inv_base);
            inv_bi *= inv_base;
        }
        return val;
    };

    for (daxa_u32 i = 0; i < halton_offsets.size(); ++i) {
        halton_offsets[i] = daxa_f32vec2{radical_inverse(i, 2) - 0.5f, radical_inverse(i, 3) - 0.5f};
    }

    main_pipeline_manager->add_virtual_file({
        .name = "FULL_SCREEN_TRIANGLE_VERTEX_SHADER",
        .contents = R"glsl(
            void main() {
                switch (gl_VertexIndex) {
                case 0: gl_Position = vec4(-1, -1, 0, 1); break;
                case 1: gl_Position = vec4(-1, +4, 0, 1); break;
                case 2: gl_Position = vec4(+4, -1, 0, 1); break;
                }
            }
        )glsl",
    });

    main_task_graph = record_main_task_graph();
    main_pipeline_manager->wait();
    debug_utils::Console::add_log(std::format("startup: {} s\n", std::chrono::duration<float>(Clock::now() - start).count()));
}
VoxelApp::~VoxelApp() {
    device.wait_idle();
    device.collect_garbage();

    renderer.destroy(device);
    gpu_resources.destroy(device);
    voxel_world.destroy(device);
    particles.destroy(device);
    voxel_model_loader.destroy();

    for (auto const &[id, temporal_buffer] : temporal_buffers) {
        device.destroy_buffer(temporal_buffer.buffer_id);
    }
}

// [Main loop]
// handle resize event
// VoxelApp::on_update()
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

// [Update engine state]
// Reload pipeline manager
// Update UI
// Update swapchain
// Recreate voxel chunks (conditional)
// Handle GVOX model upload
// Voxel malloc management
// Execute main task graph
void VoxelApp::on_update() {
    auto now = Clock::now();

    swapchain_image = swapchain.acquire_next_image();

    auto t0 = Clock::now();
    gpu_input.time = std::chrono::duration<daxa_f32>(now - start).count();
    gpu_input.delta_time = std::chrono::duration<daxa_f32>(now - prev_time).count();
    prev_time = now;
    gpu_input.render_res_scl = render_res_scl;
    gpu_input.fov = AppSettings::get<settings::SliderFloat>("Camera", "FOV").value * (std::numbers::pi_v<daxa_f32> / 180.0f);
    gpu_input.sensitivity = ui.settings.mouse_sensitivity;

#if ENABLE_TAA
    gpu_input.halton_jitter = halton_offsets[gpu_input.frame_index % halton_offsets.size()];
#endif

    audio.set_frequency(gpu_input.delta_time * 1000.0f * 200.0f);

    if (ui.should_hotload_shaders) {
        auto reload_result = main_pipeline_manager->reload_all();
        if (auto *reload_err = daxa::get_if<daxa::PipelineReloadError>(&reload_result)) {
            debug_utils::Console::add_log(reload_err->message);
        }
    }

    task_swapchain_image.set_images({.images = {&swapchain_image, 1}});
    if (swapchain_image.is_empty()) {
        return;
    }

    if (ui.should_upload_seed_data) {
        gpu_resources.update_seeded_value_noise(device, std::hash<std::string>{}(ui.settings.world_seed_str));
        ui.should_upload_seed_data = false;
    }

    // gpu_app_draw_ui();

    if (ui.should_run_startup || voxel_model_loader.model_is_ready) {
        run_startup(main_task_graph);
    }

    voxel_model_loader.update(ui);

    if (ui.should_record_task_graph) {
        device.wait_idle();
        main_task_graph = record_main_task_graph();
    }

    gpu_app_begin_frame(main_task_graph);

    gpu_input.fif_index = gpu_input.frame_index % (FRAMES_IN_FLIGHT + 1);
    // condition_values[static_cast<size_t>(Conditions::DYNAMIC_BUFFERS_REALLOC)] = should_realloc;
    // main_task_graph.execute({.permutation_condition_values = condition_values});
    main_task_graph.execute({});

    gpu_input.resize_factor = 1.0f;
    gpu_input.mouse.pos_delta = {0.0f, 0.0f};
    gpu_input.mouse.scroll_delta = {0.0f, 0.0f};

    renderer.end_frame(device, gpu_input.delta_time);

    auto t1 = Clock::now();
    ui.update(gpu_input.delta_time, std::chrono::duration<daxa_f32>(t1 - t0).count());

    ++gpu_input.frame_index;
    device.collect_garbage();
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

    if (ui.settings.keybinds.contains(key_id)) {
        gpu_input.actions[ui.settings.keybinds.at(key_id)] = static_cast<daxa_u32>(action);
    }
}
void VoxelApp::on_resize(daxa_u32 sx, daxa_u32 sy) {
    minimized = (sx == 0 || sy == 0);
    auto new_render_res_scl = AppSettings::get<settings::SliderFloat>("Graphics", "Render Res Scale").value;
    auto resized = sx != window_size.x || sy != window_size.y || render_res_scl != new_render_res_scl;
    if (!minimized && resized) {
        swapchain.resize();
        window_size.x = swapchain.get_surface_extent().x;
        window_size.y = swapchain.get_surface_extent().y;
        render_res_scl = new_render_res_scl;
        {
            // resize render images
            // gpu_resources.render_images.size.x = static_cast<daxa_u32>(static_cast<daxa_f32>(window_size.x) * render_res_scl);
            // gpu_resources.render_images.size.y = static_cast<daxa_u32>(static_cast<daxa_f32>(window_size.y) * render_res_scl);
            device.wait_idle();
            needs_vram_calc = true;
        }
        main_task_graph = record_main_task_graph();
        gpu_input.resize_factor = 0.0f;
        on_update();
    }
}
void VoxelApp::on_drop(std::span<char const *> filepaths) {
    ui.gvox_model_path = filepaths[0];
    ui.should_upload_gvox_model = true;
}

void VoxelApp::compute_image_sizes() {
    gpu_input.frame_dim.x = static_cast<daxa_u32>(static_cast<daxa_f32>(window_size.x) * render_res_scl);
    gpu_input.frame_dim.y = static_cast<daxa_u32>(static_cast<daxa_f32>(window_size.y) * render_res_scl);
    gpu_input.rounded_frame_dim = round_frame_dim(gpu_input.frame_dim);
    gpu_input.output_resolution = window_size;
}

// [Engine initializations tasks]
//
// Startup Task (Globals Clear):
// Clear task_globals_buffer
// Clear task_temp_voxel_chunks_buffer
// Clear task_voxel_chunks_buffer
// Clear task_voxel_malloc_pages_buffer (x3)
//
// GPU Task:
// startup.comp.glsl (Run on 1 thread)
//
// Initialize Task:
// init VoxelMallocPageAllocator buffer
void VoxelApp::run_startup(daxa::TaskGraph & /*unused*/) {
    auto temp_task_graph = daxa::TaskGraph({
        .device = device,
        .name = "temp_task_graph",
    });

    auto record_ctx = RecordContext{
        .device = this->device,
        .task_graph = temp_task_graph,
        .pipeline_manager = main_pipeline_manager.get(),
        .compute_pipelines = &this->compute_pipelines,
        .raster_pipelines = &this->raster_pipelines,
    };

    gpu_app_record_startup(record_ctx);

    temp_task_graph.submit({});
    temp_task_graph.complete({});
    temp_task_graph.execute({});

    ui.should_run_startup = false;
}

// [Record the command list sent to the GPU each frame]

// List of tasks:

// (Conditional tasks)
// Startup (startup.comp.glsl), run on 1 thread
// Upload settings, Upload model, Voxel malloc realloc

// GpuInputUploadTransferTask
// -> copy buffer from gpu_input to task_input_buffer

// PerframeTask (perframe.comp.glsl), run on 1 thread
// -> player_perframe() : update player
// -> voxel_world_perframe() : init voxel world
// -> update brush
// -> update voxel_malloc_page_allocator
// -> update thread pool
// -> update particles

// VoxelParticleSimComputeTask
// -> Simulate the particles

// ChunkEdit (voxel_world.comp.glsl)
// -> Actually build the chunks depending on the chunk work items (containing brush infos)

// ChunkOpt_x2x4                 [Optim]
// ChunkOpt_x8up                 [Optim]
// ChunkAlloc                    [Optim]
// VoxelParticleRasterTask       [Particles]
// TraceDepthPrepassTask         [Optim]
// TracePrimaryTask              [Render]
// ColorSceneTask                [Render]
// GpuOutputDownloadTransferTask [I/O]
// PostprocessingTask            [Render]
// ImGui draw                    [GUI Render]
auto VoxelApp::record_main_task_graph() -> daxa::TaskGraph {
    ui.should_record_task_graph = false;

    compute_image_sizes();

    daxa::TaskGraph result_task_graph = daxa::TaskGraph({
        .device = device,
        .swapchain = swapchain,
        .alias_transients = GVOX_ENGINE_INSTALL,
        .permutation_condition_count = static_cast<size_t>(Conditions::COUNT),
        .name = "main_task_graph",
    });

    result_task_graph.use_persistent_image(task_swapchain_image);

    auto record_ctx = RecordContext{
        .device = this->device,
        .task_graph = result_task_graph,
        .pipeline_manager = main_pipeline_manager.get(),
        .render_resolution = gpu_input.rounded_frame_dim,
        .output_resolution = gpu_input.output_resolution,
        .task_swapchain_image = task_swapchain_image,
        .compute_pipelines = &this->compute_pipelines,
        .raster_pipelines = &this->raster_pipelines,
        .temporal_buffers = &this->temporal_buffers,
    };

    // task_value_noise_image.view().view({});

    // TODO: Pass settings into frame recording?
    // kajiya_renderer.do_global_illumination = ui.settings.global_illumination;
    gpu_app_record_frame(record_ctx);

    record_ctx.task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskImageAccess::COLOR_ATTACHMENT, daxa::ImageViewType::REGULAR_2D, task_swapchain_image),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            imgui_renderer.record_commands(ImGui::GetDrawData(), ti.recorder, swapchain_image, window_size.x, window_size.y);
        },
        .name = "ImGui draw",
    });

    result_task_graph.submit({});
    result_task_graph.present({});
    result_task_graph.complete({});

    return result_task_graph;
}

void VoxelApp::gpu_app_draw_ui() {
    // for (auto const &str : ui_strings) {
    //     ImGui::Text("%s", str.c_str());
    // }
    // if (ImGui::TreeNode("Player")) {
    //     ImGui::Text("pos: %.2f, %.2f, %.2f", static_cast<double>(gpu_output.player_pos.x), static_cast<double>(gpu_output.player_pos.y), static_cast<double>(gpu_output.player_pos.z));
    //     ImGui::Text("y/p/r: %.2f, %.2f, %.2f", static_cast<double>(gpu_output.player_rot.x), static_cast<double>(gpu_output.player_rot.y), static_cast<double>(gpu_output.player_rot.z));
    //     ImGui::Text("unit offs: %.2f, %.2f, %.2f", static_cast<double>(gpu_output.player_unit_offset.x), static_cast<double>(gpu_output.player_unit_offset.y), static_cast<double>(gpu_output.player_unit_offset.z));
    //     ImGui::TreePop();
    // }
    // if (ImGui::TreeNode("Auto-Exposure")) {
    //     ImGui::Text("Exposure multiple: %.2f", static_cast<double>(gpu_input.pre_exposure));
    //     auto hist_float = std::array<float, LUMINANCE_HISTOGRAM_BIN_COUNT>{};
    //     auto hist_min = static_cast<float>(kajiya_renderer.post_processor.histogram[0]);
    //     auto hist_max = static_cast<float>(kajiya_renderer.post_processor.histogram[0]);
    //     auto first_bin_with_value = -1;
    //     auto last_bin_with_value = -1;
    //     for (uint32_t i = 0; i < LUMINANCE_HISTOGRAM_BIN_COUNT; ++i) {
    //         if (first_bin_with_value == -1 && kajiya_renderer.post_processor.histogram[i] != 0) {
    //             first_bin_with_value = i;
    //         }
    //         if (kajiya_renderer.post_processor.histogram[i] != 0) {
    //             last_bin_with_value = i;
    //         }
    //         hist_float[i] = static_cast<float>(kajiya_renderer.post_processor.histogram[i]);
    //         hist_min = std::min(hist_min, hist_float[i]);
    //         hist_max = std::max(hist_max, hist_float[i]);
    //     }
    //     ImGui::PlotHistogram("Histogram", hist_float.data(), static_cast<int>(hist_float.size()), 0, "hist", hist_min, hist_max, ImVec2(0, 120.0f));
    //     ImGui::Text("min %.2f | max %.2f", static_cast<double>(hist_min), static_cast<double>(hist_max));
    //     auto a = double(first_bin_with_value) / 256.0 * (LUMINANCE_HISTOGRAM_MAX_LOG2 - LUMINANCE_HISTOGRAM_MIN_LOG2) + LUMINANCE_HISTOGRAM_MIN_LOG2;
    //     auto b = double(last_bin_with_value) / 256.0 * (LUMINANCE_HISTOGRAM_MAX_LOG2 - LUMINANCE_HISTOGRAM_MIN_LOG2) + LUMINANCE_HISTOGRAM_MIN_LOG2;
    //     ImGui::Text("first bin %d (%.2f) | last bin %d (%.2f)", first_bin_with_value, exp2(a), last_bin_with_value, exp2(b));
    //     ImGui::TreePop();
    // }
}
void VoxelApp::gpu_app_calc_vram_usage(daxa::TaskGraph &task_graph) {
    std::vector<debug_utils::DebugDisplay::GpuResourceInfo> &debug_gpu_resource_infos = debug_utils::DebugDisplay::s_instance->gpu_resource_infos;

    debug_gpu_resource_infos.clear();
    ui_strings.clear();

    size_t result_size = 0;

    // auto format_to_pixel_size = [](daxa::Format format) -> daxa_u32 {
    //     switch (format) {
    //     case daxa::Format::R16G16B16_SFLOAT: return 3 * 2;
    //     case daxa::Format::R16G16B16A16_SFLOAT: return 4 * 2;
    //     case daxa::Format::R32G32B32_SFLOAT: return 3 * 4;
    //     default:
    //     case daxa::Format::R32G32B32A32_SFLOAT: return 4 * 4;
    //     }
    // };

    // auto image_size = [&device, &format_to_pixel_size, &result_size, &debug_gpu_resource_infos](daxa::ImageId image) {
    //     if (image.is_empty()) {
    //         return;
    //     }
    //     auto image_info = device.info_image(image).value();
    //     auto size = format_to_pixel_size(image_info.format) * image_info.size.x * image_info.size.y * image_info.size.z;
    //     debug_gpu_resource_infos.push_back({
    //         .type = "image",
    //         .name = image_info.name.data(),
    //         .size = size,
    //     });
    //     result_size += size;
    // };
    auto buffer_size = [this, &result_size, &debug_gpu_resource_infos](daxa::BufferId buffer) {
        if (buffer.is_empty()) {
            return;
        }
        auto buffer_info = this->device.info_buffer(buffer).value();
        debug_gpu_resource_infos.push_back({
            .type = "buffer",
            .name = buffer_info.name.data(),
            .size = buffer_info.size,
        });
        result_size += buffer_info.size;
    };

    buffer_size(gpu_resources.input_buffer);
    buffer_size(gpu_resources.globals_buffer);

    voxel_world.for_each_buffer(buffer_size);

    // buffer_size(gpu_resources.gvox_model_buffer);
    buffer_size(particles.simulated_voxel_particles_buffer);
    buffer_size(particles.rendered_voxel_particles_buffer);
    buffer_size(particles.placed_voxel_particles_buffer);

    {
        auto size = task_graph.get_transient_memory_size();
        debug_gpu_resource_infos.push_back({
            .type = "buffer",
            .name = "Transient Memory Buffer",
            .size = size,
        });
        result_size += size;
    }

    needs_vram_calc = false;

    ui_strings.push_back(fmt::format("Est. VRAM usage: {} MB", static_cast<float>(result_size) / 1000000));
}
void VoxelApp::gpu_app_begin_frame(daxa::TaskGraph &task_graph) {
    gpu_input.sampler_nnc = gpu_resources.sampler_nnc;
    gpu_input.sampler_lnc = gpu_resources.sampler_lnc;
    gpu_input.sampler_llc = gpu_resources.sampler_llc;
    gpu_input.sampler_llr = gpu_resources.sampler_llr;

    gpu_input.flags &= ~GAME_FLAG_BITS_PAUSED;
    gpu_input.flags |= GAME_FLAG_BITS_PAUSED * static_cast<daxa_u32>(ui.paused);

    gpu_input.flags &= ~GAME_FLAG_BITS_NEEDS_PHYS_UPDATE;

    renderer.begin_frame(gpu_input, gpu_output);

    auto now = Clock::now();
    if (now - prev_phys_update_time > std::chrono::duration<float>(GAME_PHYS_UPDATE_DT)) {
        gpu_input.flags |= GAME_FLAG_BITS_NEEDS_PHYS_UPDATE;
        prev_phys_update_time = now;
    }

    if (needs_vram_calc) {
        gpu_app_calc_vram_usage(task_graph);
    }

    voxel_world.begin_frame(device, gpu_output.voxel_world);
}
void VoxelApp::gpu_app_record_startup(RecordContext &record_ctx) {
    record_ctx.task_graph.use_persistent_buffer(gpu_resources.task_input_buffer);
    record_ctx.task_graph.use_persistent_buffer(gpu_resources.task_globals_buffer);

    voxel_world.use_buffers(record_ctx);

    voxel_world.record_startup(record_ctx);
    record_ctx.task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, gpu_resources.task_globals_buffer),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            ti.recorder.clear_buffer({
                .buffer = gpu_resources.task_globals_buffer.get_state().buffers[0],
                .offset = 0,
                .size = sizeof(GpuGlobals),
                .clear_value = 0,
            });
        },
        .name = "StartupTask (Globals Clear)",
    });

    record_ctx.add(ComputeTask<StartupCompute, StartupComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"app.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{StartupCompute::gpu_input, gpu_resources.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{StartupCompute::globals, gpu_resources.task_globals_buffer}},
            VOXELS_BUFFER_USES_ASSIGN(StartupCompute, voxel_world.buffers),
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, StartupComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({1, 1, 1});
        },
    });
}
void VoxelApp::gpu_app_record_frame(RecordContext &record_ctx) {
    gpu_resources.use_resources(record_ctx);
    voxel_world.use_buffers(record_ctx);
    particles.use_buffers(record_ctx);
    record_ctx.task_graph.use_persistent_buffer(voxel_model_loader.task_gvox_model_buffer);

    record_ctx.task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, gpu_resources.task_input_buffer),
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
                .dst_buffer = gpu_resources.task_input_buffer.get_state().buffers[0],
                .size = sizeof(GpuInput),
            });
        },
        .name = "GpuInputUploadTransferTask",
    });

    // TODO: Refactor this. I hate that due to these tasks, I have to call this "use_buffers" thing above.
    record_ctx.add(ComputeTask<PerframeCompute, PerframeComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"app.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{PerframeCompute::gpu_input, gpu_resources.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{PerframeCompute::gpu_output, gpu_resources.task_output_buffer}},
            daxa::TaskViewVariant{std::pair{PerframeCompute::globals, gpu_resources.task_globals_buffer}},
            daxa::TaskViewVariant{std::pair{PerframeCompute::simulated_voxel_particles, particles.task_simulated_voxel_particles_buffer}},
            VOXELS_BUFFER_USES_ASSIGN(PerframeCompute, voxel_world.buffers),
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, PerframeComputePush &push, NoTaskInfo const &) {
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            ti.recorder.dispatch({1, 1, 1});
        },
    });

    particles.simulate(record_ctx, voxel_world.buffers);
    voxel_world.record_frame(record_ctx, voxel_model_loader.task_gvox_model_buffer, gpu_resources.task_value_noise_image);

    renderer.render(record_ctx, voxel_world.buffers, particles, record_ctx.task_swapchain_image, swapchain.get_format());

    record_ctx.task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_READ, gpu_resources.task_output_buffer),
            daxa::inl_attachment(daxa::TaskBufferAccess::HOST_TRANSFER_WRITE, gpu_resources.task_staging_output_buffer),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            auto output_buffer = gpu_resources.task_output_buffer.get_state().buffers[0];
            auto staging_output_buffer = gpu_resources.staging_output_buffer;
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

    // test_compute(record_ctx);

    needs_vram_calc = true;
}
