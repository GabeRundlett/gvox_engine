#include "voxel_app.hpp"

#include <thread>
#include <numbers>
#include <fstream>
#include <random>
#include <unordered_map>

#include <gvox/adapters/input/byte_buffer.h>
#include <gvox/adapters/output/byte_buffer.h>
#include <gvox/adapters/parse/voxlap.h>

#define APPNAME "Voxel App"

using namespace std::chrono_literals;

#include <iostream>

constexpr auto round_frame_dim(u32vec2 size) {
    constexpr auto over_estimation = u32vec2{32, 32};
    auto result = size;
    // auto result = (size + u32vec2{over_estimation.x - 1u, over_estimation.y - 1u}) / over_estimation * over_estimation;
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
    : AppWindow(APPNAME, {800, 600}),
      daxa_instance{daxa::create_instance({})},
      device{daxa_instance.create_device({
          .enable_buffer_device_address_capture_replay = !GVOX_ENGINE_INSTALL,
          .name = "device",
      })},
      swapchain{device.create_swapchain({
          .native_window = AppWindow::get_native_handle(),
          .native_window_platform = AppWindow::get_native_platform(),
          .present_mode = daxa::PresentMode::IMMEDIATE,
          .image_usage = daxa::ImageUsageFlagBits::TRANSFER_DST,
          .max_allowed_frames_in_flight = FRAMES_IN_FLIGHT,
          .name = "swapchain",
      })},
      main_pipeline_manager{[this]() {
          auto result = daxa::PipelineManager({
              .device = device,
              .shader_compile_options = {
                  .root_paths = {
                      DAXA_SHADER_INCLUDE_DIR,
                      "assets",
                      "src",
                      "gpu",
                      "src/gpu",
                      "src/gpu/renderer",
                  },
                  .language = daxa::ShaderLanguage::GLSL,
                  .enable_debug_info = true,
              },
              .register_null_pipelines_when_first_compile_fails = true,
              .name = "pipeline_manager",
          });
          result.add_virtual_file({
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
          return result;
      }()},
      ui{[this]() {
          auto result = AppUi(AppWindow::glfw_window_ptr);
          auto const &device_props = device.properties();
          result.debug_gpu_name = reinterpret_cast<char const *>(device_props.device_name);
          return result;
      }()},
      imgui_renderer{[this]() {
          return daxa::ImGuiRenderer({
              .device = device,
              .format = swapchain.get_format(),
              .use_custom_config = false,
          });
      }()},
      gpu_app{device, main_pipeline_manager, swapchain.get_format()},
      main_task_graph{[this]() {
          return record_main_task_graph();
      }()},
      gvox_ctx(gvox_create_context()) {

    start = Clock::now();

    // constexpr auto IMMEDIATE_LOAD_MODEL_FROM_GABES_DRIVE = false;
    // if constexpr (IMMEDIATE_LOAD_MODEL_FROM_GABES_DRIVE) {
    //     ui.gvox_model_path = "C:/Users/gabe/AppData/Roaming/GabeVoxelGame/models/building.vox";
    //     gvox_model_data = load_gvox_data();
    //     model_is_ready = true;
    //     prev_gvox_model_buffer = gpu_resources.gvox_model_buffer;
    //     gpu_resources.gvox_model_buffer = device.create_buffer({
    //         .size = static_cast<u32>(gvox_model_data.size),
    //         .name = "gvox_model_buffer",
    //     });
    //     task_gvox_model_buffer.set_buffers({.buffers = std::array{gpu_resources.gvox_model_buffer}});
    // }

    auto radical_inverse = [](u32 n, u32 base) -> f32 {
        auto val = 0.0f;
        auto inv_base = 1.0f / static_cast<f32>(base);
        auto inv_bi = inv_base;
        while (n > 0) {
            auto d_i = n % base;
            val += static_cast<f32>(d_i) * inv_bi;
            n = static_cast<u32>(static_cast<f32>(n) * inv_base);
            inv_bi *= inv_base;
        }
        return val;
    };

    for (u32 i = 0; i < halton_offsets.size(); ++i) {
        halton_offsets[i] = f32vec2{radical_inverse(i, 2) - 0.5f, radical_inverse(i, 3) - 0.5f};
    }
}
VoxelApp::~VoxelApp() {
    gvox_destroy_context(gvox_ctx);
    device.wait_idle();
    device.collect_garbage();
    gpu_app.destroy(device);
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
            auto resized = render_res_scl != ui.settings.render_res_scl;
            if (resized) {
                on_resize(window_size.x, window_size.y);
            }

            if (ui.settings.battery_saving_mode) {
                std::this_thread::sleep_for(10ms);
            }

            on_update();
        } else {
            std::this_thread::sleep_for(1ms);
        }
    }
}

auto VoxelApp::load_gvox_data() -> GvoxModelData {
    auto result = GvoxModelData{};
    auto file = std::ifstream(ui.gvox_model_path, std::ios::binary);
    if (!file.is_open()) {
        AppUi::Console::s_instance->add_log("[error] Failed to load the model");
        ui.should_upload_gvox_model = false;
        return result;
    }
    file.seekg(0, std::ios_base::end);
    auto temp_gvox_model_size = static_cast<u32>(file.tellg());
    auto temp_gvox_model = std::vector<uint8_t>{};
    temp_gvox_model.resize(temp_gvox_model_size);
    {
        // time_t start = clock();
        file.seekg(0, std::ios_base::beg);
        file.read(reinterpret_cast<char *>(temp_gvox_model.data()), static_cast<std::streamsize>(temp_gvox_model_size));
        file.close();
        // time_t end = clock();
        // double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        // AppUi::Console::s_instance->add_log("(pulling file into memory: {}s)", cpu_time_used);
    }
    GvoxByteBufferInputAdapterConfig i_config = {
        .data = temp_gvox_model.data(),
        .size = temp_gvox_model_size,
    };
    GvoxByteBufferOutputAdapterConfig o_config = {
        .out_size = &result.size,
        .out_byte_buffer_ptr = &result.ptr,
        .allocate = nullptr,
    };
    void *i_config_ptr = nullptr;
    auto voxlap_config = GvoxVoxlapParseAdapterConfig{
        .size_x = 512,
        .size_y = 512,
        .size_z = 64,
        .make_solid = 1,
        .is_ace_of_spades = 1,
    };
    char const *gvox_model_type = "gvox_palette";
    if (ui.gvox_model_path.has_extension()) {
        auto ext = ui.gvox_model_path.extension();
        if (ext == ".vox") {
            gvox_model_type = "magicavoxel";
        }
        if (ext == ".rle") {
            gvox_model_type = "gvox_run_length_encoding";
        }
        if (ext == ".oct") {
            gvox_model_type = "gvox_octree";
        }
        if (ext == ".glp") {
            gvox_model_type = "gvox_global_palette";
        }
        if (ext == ".brk") {
            gvox_model_type = "gvox_brickmap";
        }
        if (ext == ".gvr") {
            gvox_model_type = "gvox_raw";
        }
        if (ext == ".vxl") {
            i_config_ptr = &voxlap_config;
            gvox_model_type = "voxlap";
        }
    }
    GvoxAdapterContext *i_ctx = gvox_create_adapter_context(gvox_ctx, gvox_get_input_adapter(gvox_ctx, "byte_buffer"), &i_config);
    GvoxAdapterContext *o_ctx = gvox_create_adapter_context(gvox_ctx, gvox_get_output_adapter(gvox_ctx, "byte_buffer"), &o_config);
    GvoxAdapterContext *p_ctx = gvox_create_adapter_context(gvox_ctx, gvox_get_parse_adapter(gvox_ctx, gvox_model_type), i_config_ptr);
    GvoxAdapterContext *s_ctx = gvox_create_adapter_context(gvox_ctx, gvox_get_serialize_adapter(gvox_ctx, "gvox_palette"), nullptr);

    {
        // time_t start = clock();
        gvox_blit_region(
            i_ctx, o_ctx, p_ctx, s_ctx,
            nullptr,
            // &ui.gvox_region_range,
            // GVOX_CHANNEL_BIT_COLOR | GVOX_CHANNEL_BIT_MATERIAL_ID | GVOX_CHANNEL_BIT_EMISSIVITY);
            GVOX_CHANNEL_BIT_COLOR);
        // time_t end = clock();
        // double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        // AppUi::Console::s_instance->add_log("{}s, new size: {} bytes", cpu_time_used, result.size);

        GvoxResult res = gvox_get_result(gvox_ctx);
        // int error_count = 0;
        while (res != GVOX_RESULT_SUCCESS) {
            size_t size = 0;
            gvox_get_result_message(gvox_ctx, nullptr, &size);
            char *str = new char[size + 1];
            gvox_get_result_message(gvox_ctx, str, nullptr);
            str[size] = '\0';
            AppUi::Console::s_instance->add_log("ERROR loading model: {}", str);
            gvox_pop_result(gvox_ctx);
            delete[] str;
            res = gvox_get_result(gvox_ctx);
            // ++error_count;
        }
    }

    gvox_destroy_adapter_context(i_ctx);
    gvox_destroy_adapter_context(o_ctx);
    gvox_destroy_adapter_context(p_ctx);
    gvox_destroy_adapter_context(s_ctx);
    return result;
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
    gpu_input.time = std::chrono::duration<f32>(now - start).count();
    gpu_input.delta_time = std::chrono::duration<f32>(now - prev_time).count();
    prev_time = now;
    gpu_input.render_res_scl = ui.settings.render_res_scl;
    gpu_input.halton_jitter = halton_offsets[gpu_input.frame_index % halton_offsets.size()];
    gpu_input.fov = ui.settings.camera_fov * (std::numbers::pi_v<f32> / 180.0f);
    gpu_input.sensitivity = ui.settings.mouse_sensitivity;
    gpu_input.log2_chunks_per_axis = ui.settings.log2_chunks_per_axis;

    if (ui.should_hotload_shaders) {
        auto reload_result = main_pipeline_manager.reload_all();
        if (auto *reload_err = std::get_if<daxa::PipelineReloadError>(&reload_result)) {
            AppUi::Console::s_instance->add_log(reload_err->message);
        }
    }

    task_swapchain_image.set_images({.images = {&swapchain_image, 1}});
    if (swapchain_image.is_empty()) {
        return;
    }

    if (ui.should_upload_seed_data) {
        update_seeded_value_noise();
    }

    if (ui.should_upload_gvox_model) {
        if (!model_is_loading) {
            gvox_model_data_future = std::async(std::launch::async, &VoxelApp::load_gvox_data, this);
            model_is_loading = true;
        }
        if (model_is_loading && gvox_model_data_future.wait_for(0.01s) == std::future_status::ready) {
            model_is_ready = true;
            model_is_loading = false;

            gvox_model_data = gvox_model_data_future.get();
            gpu_app.prev_gvox_model_buffer = gpu_app.gpu_resources.gvox_model_buffer;
            gpu_app.gpu_resources.gvox_model_buffer = device.create_buffer({
                .size = static_cast<u32>(gvox_model_data.size),
                .name = "gvox_model_buffer",
            });
            gpu_app.task_gvox_model_buffer.set_buffers({.buffers = std::array{gpu_app.gpu_resources.gvox_model_buffer}});
        }
    }

    gpu_app.begin_frame(device, main_task_graph, ui);

    if (ui.should_run_startup || model_is_ready) {
        run_startup(main_task_graph);
    }
    if (model_is_ready) {
        upload_model(main_task_graph);
    }

    gpu_input.fif_index = gpu_input.frame_index % (FRAMES_IN_FLIGHT + 1);
    // condition_values[static_cast<usize>(Conditions::DYNAMIC_BUFFERS_REALLOC)] = should_realloc;
    // main_task_graph.execute({.permutation_condition_values = condition_values});
    main_task_graph.execute({});
    model_is_ready = false;

    gpu_input.resize_factor = 1.0f;
    gpu_input.mouse.pos_delta = {0.0f, 0.0f};
    gpu_input.mouse.scroll_delta = {0.0f, 0.0f};

    ui.debug_gpu_heap_usage = gpu_output.voxel_malloc_output.current_element_count * VOXEL_MALLOC_PAGE_SIZE_BYTES;
    ui.debug_player_pos = gpu_output.player_pos;
    ui.debug_player_rot = gpu_output.player_rot;
    ui.debug_chunk_offset = gpu_output.chunk_offset;
    ui.debug_page_count = gpu_app.voxel_world.buffers.voxel_malloc.current_element_count;
    ui.debug_total_jobs_ran = gpu_output.total_jobs_ran;

    // task_render_pos_image.swap_images(task_render_prev_pos_image);
    // task_render_col_image.swap_images(task_render_prev_col_image);

    gpu_app.end_frame();

    auto t1 = Clock::now();
    ui.update(gpu_input.delta_time, std::chrono::duration<f32>(t1 - t0).count());

    ++gpu_input.frame_index;
}
void VoxelApp::on_mouse_move(f32 x, f32 y) {
    f32vec2 const center = {static_cast<f32>(window_size.x / 2), static_cast<f32>(window_size.y / 2)};
    gpu_input.mouse.pos = f32vec2{x, y};
    auto offset = gpu_input.mouse.pos - center;
    gpu_input.mouse.pos = gpu_input.mouse.pos *f32vec2{static_cast<f32>(gpu_input.frame_dim.x), static_cast<f32>(gpu_input.frame_dim.y)} / f32vec2{static_cast<f32>(window_size.x), static_cast<f32>(window_size.y)};
    if (!ui.paused) {
        gpu_input.mouse.pos_delta = gpu_input.mouse.pos_delta + offset;
        set_mouse_pos(center.x, center.y);
    }
}
void VoxelApp::on_mouse_scroll(f32 dx, f32 dy) {
    auto &io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        return;
    }

    gpu_input.mouse.scroll_delta = gpu_input.mouse.scroll_delta + f32vec2{dx, dy};
}
void VoxelApp::on_mouse_button(i32 button_id, i32 action) {
    auto &io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        return;
    }
    if (ui.limbo_action_index != INVALID_GAME_ACTION) {
        return;
    }

    if (ui.settings.mouse_button_binds.contains(button_id)) {
        gpu_input.actions[ui.settings.mouse_button_binds.at(button_id)] = static_cast<u32>(action);
    }
}
void VoxelApp::on_key(i32 key_id, i32 action) {
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
        if (key_id == GLFW_KEY_F1 && action == GLFW_PRESS) {
            ui.toggle_help();
        }
        if (key_id == GLFW_KEY_GRAVE_ACCENT && action == GLFW_PRESS) {
            ui.toggle_console();
        }
    }

    if (key_id == GLFW_KEY_R && action == GLFW_PRESS) {
        ui.should_run_startup = true;
        start = Clock::now();
    }

    if (ui.settings.keybinds.contains(key_id)) {
        gpu_input.actions[ui.settings.keybinds.at(key_id)] = static_cast<u32>(action);
    }
}
void VoxelApp::on_resize(u32 sx, u32 sy) {
    minimized = (sx == 0 || sy == 0);
    auto resized = sx != window_size.x || sy != window_size.y || render_res_scl != ui.settings.render_res_scl;
    if (!minimized && resized) {
        swapchain.resize();
        window_size.x = swapchain.get_surface_extent().x;
        window_size.y = swapchain.get_surface_extent().y;
        render_res_scl = ui.settings.render_res_scl;
        {
            // resize render images
            // gpu_resources.render_images.size.x = static_cast<u32>(static_cast<f32>(window_size.x) * render_res_scl);
            // gpu_resources.render_images.size.y = static_cast<u32>(static_cast<f32>(window_size.y) * render_res_scl);
            device.wait_idle();
            gpu_app.needs_vram_calc = true;
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
    gpu_input.frame_dim.x = static_cast<u32>(static_cast<f32>(window_size.x) * render_res_scl);
    gpu_input.frame_dim.y = static_cast<u32>(static_cast<f32>(window_size.y) * render_res_scl);
    gpu_input.rounded_frame_dim = round_frame_dim(gpu_input.frame_dim);
    gpu_input.output_resolution = window_size;
}

void VoxelApp::update_seeded_value_noise() {
    daxa::TaskGraph temp_task_graph = daxa::TaskGraph({
        .device = device,
        .name = "temp_task_graph",
    });
    temp_task_graph.use_persistent_image(gpu_app.task_value_noise_image);
    temp_task_graph.add_task({
        .uses = {
            daxa::TaskImageUse<daxa::TaskImageAccess::TRANSFER_WRITE>{gpu_app.task_value_noise_image.view().view({.layer_count = 256})},
        },
        .task = [this](daxa::TaskInterface task_runtime) {
            auto staging_buffer = device.create_buffer({
                .size = static_cast<u32>(256 * 256 * 256 * 1),
                .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = "staging_buffer",
            });
            auto *buffer_ptr = device.get_host_address_as<u8>(staging_buffer);
            std::mt19937_64 rng(std::hash<std::string>{}(ui.settings.world_seed_str));
            std::uniform_int_distribution<std::mt19937::result_type> dist(0, 255);
            for (u32 i = 0; i < (256 * 256 * 256 * 1); ++i) {
                buffer_ptr[i] = dist(rng) & 0xff;
            }
            auto cmd_list = task_runtime.get_command_list();
            cmd_list.pipeline_barrier({
                .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
            });
            cmd_list.destroy_buffer_deferred(staging_buffer);
            for (u32 i = 0; i < 256; ++i) {
                cmd_list.copy_buffer_to_image({
                    .buffer = staging_buffer,
                    .buffer_offset = 256 * 256 * i,
                    .image = gpu_app.task_value_noise_image.get_state().images[0],
                    .image_slice{
                        .base_array_layer = i,
                        .layer_count = 1,
                    },
                    .image_extent = {256, 256, 1},
                });
            }
            gpu_app.needs_vram_calc = true;
        },
        .name = "upload_value_noise",
    });
    temp_task_graph.submit({});
    temp_task_graph.complete({});
    temp_task_graph.execute({});
    ui.should_upload_seed_data = false;
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
void VoxelApp::run_startup(daxa::TaskGraph &) {
    auto temp_task_graph = daxa::TaskGraph({
        .device = device,
        .name = "temp_task_graph",
    });

    auto record_ctx = RecordContext{
        .device = this->device,
        .task_graph = temp_task_graph,
    };

    gpu_app.startup(record_ctx);

    temp_task_graph.submit({});
    temp_task_graph.complete({});
    temp_task_graph.execute({});

    ui.should_run_startup = false;
}

void VoxelApp::upload_model(daxa::TaskGraph &) {
    auto temp_task_graph = daxa::TaskGraph({
        .device = device,
        .name = "temp_task_graph",
    });
    temp_task_graph.use_persistent_buffer(gpu_app.task_gvox_model_buffer);
    temp_task_graph.add_task({
        .uses = {
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{gpu_app.task_gvox_model_buffer},
        },
        .task = [this](daxa::TaskInterface task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            if (!gpu_app.prev_gvox_model_buffer.is_empty()) {
                cmd_list.destroy_buffer_deferred(gpu_app.prev_gvox_model_buffer);
            }
            cmd_list.pipeline_barrier({
                .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
            });
            auto staging_gvox_model_buffer = device.create_buffer({
                .size = static_cast<u32>(gvox_model_data.size),
                .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = "staging_gvox_model_buffer",
            });
            cmd_list.destroy_buffer_deferred(staging_gvox_model_buffer);
            char *buffer_ptr = device.get_host_address_as<char>(staging_gvox_model_buffer);
            std::copy(gvox_model_data.ptr, gvox_model_data.ptr + gvox_model_data.size, buffer_ptr);
            if (gvox_model_data.ptr != nullptr) {
                free(gvox_model_data.ptr);
            }
            cmd_list.copy_buffer_to_buffer({
                .src_buffer = staging_gvox_model_buffer,
                .dst_buffer = gpu_app.gpu_resources.gvox_model_buffer,
                .size = static_cast<u32>(gvox_model_data.size),
            });
            ui.should_upload_gvox_model = false;
            has_model = true;
            gpu_app.needs_vram_calc = true;
        },
        .name = "upload_model",
    });
    temp_task_graph.submit({});
    temp_task_graph.complete({});
    temp_task_graph.execute({});
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
    compute_image_sizes();

    daxa::TaskGraph result_task_graph = daxa::TaskGraph({
        .device = device,
        .swapchain = swapchain,
        .alias_transients = GVOX_ENGINE_INSTALL,
        .permutation_condition_count = static_cast<usize>(Conditions::COUNT),
        .name = "main_task_graph",
    });

    result_task_graph.use_persistent_image(task_swapchain_image);
    task_swapchain_image.set_images({.images = std::array{swapchain_image}});

    auto record_ctx = RecordContext{
        .device = this->device,
        .task_graph = result_task_graph,
        .render_resolution = gpu_input.rounded_frame_dim,
        .output_resolution = gpu_input.output_resolution,
        .task_swapchain_image = task_swapchain_image,
    };

    gpu_app.update(record_ctx);

    record_ctx.task_graph.add_task({
        .uses = {
            daxa::TaskImageUse<daxa::TaskImageAccess::COLOR_ATTACHMENT, daxa::ImageViewType::REGULAR_2D>{task_swapchain_image},
        },
        .task = [this](daxa::TaskInterface task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            imgui_renderer.record_commands(ImGui::GetDrawData(), cmd_list, swapchain_image, window_size.x, window_size.y);
        },
        .name = "ImGui draw",
    });

    result_task_graph.submit({});
    result_task_graph.present({});
    result_task_graph.complete({});

    return result_task_graph;
}
