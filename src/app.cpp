#include "app.hpp"

#include <fmt/format.h>
#include <fstream>
#include <numbers>

#include <nlohmann/json.hpp>
#include <sago/platform_folders.h>
#include <soundio/soundio.h>

#include <imgui_stdlib.h>
#include <imnodes.h>

static constexpr std::array<std::string_view, GAME_KEY_LAST + 1> control_strings{
    "Move Forward",
    "Strafe Left",
    "Move Backward",
    "Strafe Right",
    "Reload Chunks",
    "Toggle Fly",
    "Interact 1",
    "Interact 0",
    "Jump",
    "Crouch",
    "Sprint",
    "Walk",
    "Change Camera",
    "Toggle Brush Placement",
};

static constexpr std::array<std::string_view, GAME_TOOL_LAST + 1> tool_strings{
    "None",
    "Brush",
};

auto default_gpu_input() -> GpuInput {
    return {
        .settings{
            .fov = 90.0f,
            .jitter_scl = 1.0f,
            .frame_blending = 0.3f,
            .sensitivity = 1.0f,
            .daylight_cycle_time = std::numbers::pi_v<f32> * 0.6f,
        },
    };
}

#define ENABLE_THREAD_POOL 0

void ThreadPool::thread_loop() {
#if ENABLE_THREAD_POOL
    while (true) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            mutex_condition.wait(lock, [this] {
                return !jobs.empty() || should_terminate;
            });
            if (should_terminate) {
                return;
            }
            job = jobs.front();
            jobs.pop();
        }
        job();
    }
#endif
}
void ThreadPool::start() {
#if ENABLE_THREAD_POOL
    uint32_t const num_threads = std::thread::hardware_concurrency();
    threads.resize(num_threads);
    for (uint32_t i = 0; i < num_threads; i++) {
        threads.at(i) = std::thread(&ThreadPool::thread_loop, this);
    }
#endif
}
void ThreadPool::enqueue(std::function<void()> const &job) {
#if ENABLE_THREAD_POOL
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        jobs.push(job);
    }
    mutex_condition.notify_one();
#else
    job();
#endif
}
auto ThreadPool::busy() -> bool {
#if ENABLE_THREAD_POOL
    bool pool_busy;
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        pool_busy = !jobs.empty();
    }
    return pool_busy;
#else
    return false;
#endif
}
void ThreadPool::stop() {
#if ENABLE_THREAD_POOL
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        should_terminate = true;
    }
    mutex_condition.notify_all();
    for (std::thread &active_thread : threads) {
        active_thread.join();
    }
    threads.clear();
#endif
}

void Brush::cleanup(daxa::Device &device) {
    device.destroy_image(preview_thumbnail);
    custom_brush_settings.clear();
    if (custom_buffer_size != 0) {
        device.destroy_buffer(custom_brush_settings_buffer);
        delete[] custom_brush_settings_data;
    }
}

// using Clock = std::chrono::high_resolution_clock;
// struct Timer {
//     Clock::time_point start = Clock::now();
//     ~Timer() {
//         auto now = Clock::now();
//         std::cout << "elapsed: " << std::chrono::duration<float>(now - start).count() << std::endl;
//     }
// };

auto App::get_flag(u32 index) -> bool {
    return (gpu_input.settings.flags >> index) & 0x01;
}
void App::set_flag(u32 index, bool value) {
    gpu_input.settings.flags &= ~(0x01 << index);
    gpu_input.settings.flags |= static_cast<u32>(value) << index;
}

App::App()
    : BaseApp<App>(),
      // clang-format off
      gpu_input{default_gpu_input()},
      gpu_input_buffer{device.create_buffer({
          .size = sizeof(GpuInput),
          .debug_name = APPNAME_PREFIX("gpu_input_buffer"),
      })},
      render_image{device.create_image(daxa::ImageInfo{
          .format = daxa::Format::R16G16B16A16_SFLOAT,
          .size = {render_size_x, render_size_y, 1},
          .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::TRANSFER_SRC,
          .debug_name = APPNAME_PREFIX("render_image"),
      })},
      raytrace_output_image{device.create_image({
          .format = daxa::Format::R32_SFLOAT,
          .size = {render_size_x, render_size_y, 1},
          .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::TRANSFER_DST,
          .debug_name = APPNAME_PREFIX("raytrace_output_image"),
      })},
      gpu_globals_buffer{device.create_buffer({
          .size = sizeof(GpuGlobals),
          .debug_name = "gpu_globals_buffer",
      })},
      gpu_voxel_world_buffer{device.create_buffer({
          .size = sizeof(VoxelWorld),
          .debug_name = "gpu_voxel_world_buffer",
      })},
      gpu_voxel_brush_buffer{device.create_buffer({
          .size = sizeof(VoxelBrush),
          .debug_name = "gpu_voxel_brush_buffer",
      })},
      gpu_indirect_dispatch_buffer{device.create_buffer({
          .size = sizeof(GpuIndirectDispatch),
          .debug_name = "gpu_indirect_dispatch_buffer",
      })},
      optical_depth_image{device.create_image(daxa::ImageInfo{
          .format = daxa::Format::R32_SFLOAT,
          .size = {512, 512, 1},
          .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::SHADER_READ_ONLY,
          .debug_name = APPNAME_PREFIX("optical_depth_image"),
      })},
      optical_depth_sampler{device.create_sampler(daxa::SamplerInfo{
          .debug_name = APPNAME_PREFIX("optical_depth_sampler"),
      })},
      data_directory{std::filesystem::path(sago::getDataHome()) / "GabeVoxelGame"},
      brushes{load_brushes()},
      chunkgen_brush_key{absolute(std::filesystem::path("assets/brushes/terrain")).string()},
      current_brush_key{absolute(std::filesystem::path("assets/brushes/sphere")).string()},
      last_seen_brushes_folder_update{std::chrono::file_clock::now()},
      keys{},
      // clang-format on
      mouse_buttons{
          GLFW_MOUSE_BUTTON_1,
          GLFW_MOUSE_BUTTON_2,
          GLFW_MOUSE_BUTTON_3,
          GLFW_MOUSE_BUTTON_4,
          GLFW_MOUSE_BUTTON_5,
      },
      loop_task_list{record_loop_task_list()} {

    ImNodes::CreateContext();

    // gvox_model_path = "sponge.vox";
    // gvox_model_type = "magicavoxel";

    gvox_model_path = "phantom_mansion.gvox";
    gvox_model_type = "gvox";

    gvox_ctx = gvox_create_context();
    gvox_push_root_path(gvox_ctx, "assets");
    gvox_push_root_path(gvox_ctx, data_directory.string().c_str());

    // gvox_push_root_path(gvox_ctx, "C:/dev/projects/c++/gvox/tests/simple");

    thread_pool.start();

    // clang-format off
    startup_comp_pipeline = startup_pipeline_manager.add_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"startup.comp.glsl"}},
        .push_constant_size = sizeof(StartupCompPush),
        .debug_name = APPNAME_PREFIX("startup_comp_pipeline"),
    }).value();
    optical_depth_comp_pipeline = optical_depth_pipeline_manager.add_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"optical_depth.comp.glsl"}},
        .push_constant_size = sizeof(OpticalDepthCompPush),
        .debug_name = APPNAME_PREFIX("optical_depth_comp_pipeline"),
    }).value();
    subchunk_x2x4_comp_pipeline = chunkgen_pipeline_manager.add_compute_pipeline({
        .shader_info = {
            .source = daxa::ShaderFile{"chunk_opt.comp.glsl"}, 
            .compile_options = {
                .defines = {{.name = "SUBCHUNK_X2X4", .value = "1"}},
            },
        },
        .push_constant_size = sizeof(ChunkOptCompPush),
        .debug_name = APPNAME_PREFIX("subchunk_x2x4_comp_pipeline"),
    }).value();
    subchunk_x8up_comp_pipeline = chunkgen_pipeline_manager.add_compute_pipeline({
        .shader_info = {
            .source = daxa::ShaderFile{"chunk_opt.comp.glsl"}, 
            .compile_options = {
                .defines = {{.name = "SUBCHUNK_X8UP", .value = "1"}},
            },
        },
        .push_constant_size = sizeof(ChunkOptCompPush),
        .debug_name = APPNAME_PREFIX("subchunk_x8up_comp_pipeline"),
    }).value();
    subchunk_brush_x2x4_comp_pipeline = chunkgen_pipeline_manager.add_compute_pipeline({
        .shader_info = {
            .source = daxa::ShaderFile{"chunk_opt_brush.comp.glsl"}, 
            .compile_options = {
                .defines = {{.name = "SUBCHUNK_X2X4", .value = "1"}},
            },
        },
        .push_constant_size = sizeof(ChunkOptCompPush),
        .debug_name = APPNAME_PREFIX("subchunk_brush_x2x4_comp_pipeline"),
    }).value();
    subchunk_brush_x8up_comp_pipeline = basic_pipeline_manager.add_compute_pipeline({
        .shader_info = {
            .source = daxa::ShaderFile{"chunk_opt_brush.comp.glsl"}, 
            .compile_options = {
                .defines = {{.name = "SUBCHUNK_X8UP", .value = "1"}},
            },
        },
        .push_constant_size = sizeof(ChunkOptCompPush),
        .debug_name = APPNAME_PREFIX("subchunk_brush_x8up_comp_pipeline"),
    }).value();
    draw_comp_pipeline = basic_pipeline_manager.add_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"draw.comp.glsl"}},
        .push_constant_size = sizeof(DrawCompPush),
        .debug_name = APPNAME_PREFIX("draw_comp_pipeline"),
    }).value();
    raytrace_comp_pipeline = basic_pipeline_manager.add_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"raytrace.comp.glsl"}},
        .push_constant_size = sizeof(RaytraceCompPush),
        .debug_name = APPNAME_PREFIX("raytrace_comp_pipeline"),
    }).value();
    // clang-format on

    set_flag(GPU_INPUT_FLAG_INDEX_USE_PERSISTENT_THREAD_TRACE, false);

    if (!std::filesystem::exists(data_directory)) {
        std::filesystem::create_directory(data_directory);
    }
    load_settings();
}

void App::save_settings() {
    auto json = nlohmann::json{};
    for (i32 i = 0; i < GAME_KEY_LAST + 1; ++i) {
        auto str = fmt::format("key_{}", i);
        json[str] = keys[i];
    }
    auto f = std::ofstream(data_directory / "settings.json");
    f << std::setw(4) << json;
}
void App::load_settings() {
    if (!std::filesystem::exists(data_directory / "settings.json")) {
        std::filesystem::copy("assets/default_settings.json", data_directory / "settings.json");
    }
    auto json = nlohmann::json::parse(std::ifstream(data_directory / "settings.json"));
    for (i32 i = 0; i < GAME_KEY_LAST + 1; ++i) {
        auto str = fmt::format("key_{}", i);
        if (json.contains(str)) {
            keys[i] = json[str];
        }
    }
}
void App::reset_settings() {
    keys = {
        GLFW_KEY_W,
        GLFW_KEY_A,
        GLFW_KEY_S,
        GLFW_KEY_D,
        GLFW_KEY_R,
        GLFW_KEY_F,
        GLFW_KEY_Q,
        GLFW_KEY_E,
        GLFW_KEY_SPACE,
        GLFW_KEY_LEFT_CONTROL,
        GLFW_KEY_LEFT_SHIFT,
        GLFW_KEY_LEFT_ALT,
        GLFW_KEY_F5,
        GLFW_KEY_B,
    };
    gpu_input = default_gpu_input();

    save_settings();
}

auto App::load_brushes() -> std::unordered_map<std::string, Brush> {
    std::unordered_map<std::string, Brush> result;

    std::array<std::filesystem::path, 2> const brushes_roots = {
        "assets/brushes",
        data_directory / "brushes",
    };
    for (auto const &brushes_root : brushes_roots) {
        if (!std::filesystem::exists(data_directory)) {
            std::filesystem::create_directory(data_directory);
        }
        if (!std::filesystem::exists(brushes_root)) {
            std::filesystem::create_directory(brushes_root);
        }
        for (auto const &brushes_file : std::filesystem::directory_iterator{brushes_root}) {
            if (!brushes_file.is_directory())
                continue;
            auto path = absolute(brushes_file.path());
            auto name = path; // path.filename().string();
            if (result.contains(name.string())) {
                imgui_console.add_log("[error] Found 2 folders with the same name..?");
                continue;
            }
            auto display_name = name.string();
            if (!std::filesystem::exists(path / "config.json")) {
                imgui_console.add_log("[error] Failed to find the config.json file associated with brush '%s'", display_name.c_str());
                continue;
            }
            auto config_json = nlohmann::json::parse(std::ifstream(path / "config.json"));
            if (config_json.contains("display_name"))
                display_name = config_json["display_name"];
            auto custom_brush_settings = std::vector<CustomUIParameter>{};
            usize custom_buffer_size = 0;
            if (config_json.contains("input")) {
                auto const &input = config_json["input"];
                for (auto const &item : input) {
                    auto const &name_str = item["name"];
                    auto const type_str = (std::string)item["type"];
                    auto type_iter = std::find(ui_component_strings.begin(), ui_component_strings.end(), std::string_view(type_str));
                    auto id = static_cast<UiComponentID>(type_iter - ui_component_strings.begin());
                    auto type_size = ui_component_sizes[static_cast<usize>(id)];
                    custom_buffer_size += type_size;
                    CustomUIParameterTypeData type_data;
                    switch (id) {
                    case UiComponentID::COLOR: {
                        type_data = CustomUI_color{
                            .default_value = {
                                item["default_value"][0],
                                item["default_value"][1],
                                item["default_value"][2],
                            },
                        };
                    } break;
                    case UiComponentID::SLIDER_I32: {
                        type_data = CustomUI_slider_i32{
                            .default_value = item["default_value"],
                            .min = item["slider_range"][0],
                            .max = item["slider_range"][1],
                        };
                    } break;
                    case UiComponentID::SLIDER_U32: {
                        type_data = CustomUI_slider_u32{
                            .default_value = item["default_value"],
                            .min = item["slider_range"][0],
                            .max = item["slider_range"][1],
                        };
                    } break;
                    case UiComponentID::SLIDER_F32: {
                        type_data = CustomUI_slider_f32{
                            .default_value = item["default_value"],
                            .min = item["slider_range"][0],
                            .max = item["slider_range"][1],
                        };
                    } break;
                    case UiComponentID::SLIDER_F32VEC3: {
                        type_data = CustomUI_slider_f32vec3{
                            .default_value = {
                                item["default_value"][0],
                                item["default_value"][1],
                                item["default_value"][2],
                            },
                            .min = item["slider_range"][0],
                            .max = item["slider_range"][1],
                        };
                    } break;
                    case UiComponentID::INPUT_F32VEC3: {
                        type_data = CustomUI_input_f32vec3{
                            .default_value = {
                                item["default_value"][0],
                                item["default_value"][1],
                                item["default_value"][2],
                            },
                        };
                    } break;
                    }
                    custom_brush_settings.push_back(CustomUIParameter{
                        .id = id,
                        .name = name_str,
                        .type_data = type_data,
                    });
                }
            }

            auto custom_brush_settings_buffer = daxa::BufferId{};
            u8 *custom_brush_settings_data = nullptr;
            if (custom_buffer_size != 0) {
                custom_brush_settings_buffer = device.create_buffer({
                    .size = static_cast<u32>(custom_buffer_size),
                    .debug_name = "custom_brush_settings_buffer",
                });
                custom_brush_settings_data = new u8[custom_buffer_size];
                usize parameter_offset = 0;
                for (auto const &parameter : custom_brush_settings) {
                    switch (parameter.id) {
                    case UiComponentID::COLOR: {
                        f32vec3 &p_color = *reinterpret_cast<f32vec3 *>(custom_brush_settings_data + parameter_offset);
                        p_color = std::get<CustomUI_color>(parameter.type_data).default_value;
                    } break;
                    case UiComponentID::SLIDER_I32: {
                        i32 &p_i32 = *reinterpret_cast<i32 *>(custom_brush_settings_data + parameter_offset);
                        p_i32 = std::get<CustomUI_slider_i32>(parameter.type_data).default_value;
                    } break;
                    case UiComponentID::SLIDER_U32: {
                        u32 &p_u32 = *reinterpret_cast<u32 *>(custom_brush_settings_data + parameter_offset);
                        p_u32 = std::get<CustomUI_slider_u32>(parameter.type_data).default_value;
                    } break;
                    case UiComponentID::SLIDER_F32: {
                        f32 &p_f32 = *reinterpret_cast<f32 *>(custom_brush_settings_data + parameter_offset);
                        p_f32 = std::get<CustomUI_slider_f32>(parameter.type_data).default_value;
                    } break;
                    case UiComponentID::SLIDER_F32VEC3: {
                        f32vec3 &p_f32vec3 = *reinterpret_cast<f32vec3 *>(custom_brush_settings_data + parameter_offset);
                        p_f32vec3 = std::get<CustomUI_slider_f32vec3>(parameter.type_data).default_value;
                    } break;
                    case UiComponentID::INPUT_F32VEC3: {
                        f32vec3 &p_f32vec3 = *reinterpret_cast<f32vec3 *>(custom_brush_settings_data + parameter_offset);
                        p_f32vec3 = std::get<CustomUI_input_f32vec3>(parameter.type_data).default_value;
                    } break;
                    }
                    parameter_offset += ui_component_sizes[static_cast<usize>(parameter.id)];
                }
            }
            if (!std::filesystem::exists(path / "brush_info.glsl")) {
                imgui_console.add_log("[error] Failed to find the info.glsl file associated with brush '%s'", display_name.c_str());
                continue;
            }
            if (!std::filesystem::exists(path / "brush_kernel.glsl")) {
                imgui_console.add_log("[error] Failed to find the brush_kernel.glsl file associated with brush '%s'", display_name.c_str());
                continue;
            }

            auto thumbnail_path = std::filesystem::path("assets/brushes/default_thumbnail.png");
            if (config_json.contains("custom_image"))
                thumbnail_path = name / std::string(config_json["custom_image"]);

            auto image = device.create_image({
                .format = daxa::Format::R8G8B8A8_SRGB,
                .size = {1, 1, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_READ_ONLY | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .debug_name = path.string() + " thumbnail image",
            });

            result.emplace(
                name.string(),
                Brush{
                    .key = name,
                    .display_name = display_name,

                    .thumbnail_image_path = thumbnail_path,
                    .thumbnail_needs_updating = true,
                    .preview_thumbnail = image,
                    .task_preview_thumbnail = {},

                    .pipelines = {
                        .pipeline_manager = basic_pipeline_manager,
                        .perframe_comp_info = {
                            .shader_info = {
                                .source = daxa::ShaderFile{"perframe.comp.glsl"},
                                .compile_options = {.root_paths = {path}},
                            },
                            .push_constant_size = sizeof(PerframeCompPush),
                            .debug_name = APPNAME_PREFIX("perframe_comp_pipeline"),
                        },
                        .chunk_edit_comp_info = {
                            .shader_info = {
                                .source = daxa::ShaderFile{"chunk_edit.comp.glsl"},
                                .compile_options = {.root_paths = {path}},
                            },
                            .push_constant_size = sizeof(ChunkEditCompPush),
                            .debug_name = APPNAME_PREFIX("chunk_edit_comp_pipeline"),
                        },
                        .chunkgen_comp_info = {
                            .shader_info = {
                                .source = daxa::ShaderFile{"chunkgen.comp.glsl"},
                                .compile_options = {.root_paths = {path}},
                            },
                            .push_constant_size = sizeof(ChunkEditCompPush),
                            .debug_name = APPNAME_PREFIX("chunkgen_comp_pipeline"),
                        },
                        .brush_chunkgen_comp_info = {
                            .shader_info = {
                                .source = daxa::ShaderFile{"chunkgen_brush.comp.glsl"},
                                .compile_options = {.root_paths = {path}},
                            },
                            .push_constant_size = sizeof(ChunkEditCompPush),
                            .debug_name = APPNAME_PREFIX("brush_chunkgen_comp_pipeline"),
                        },
                    },
                    .settings = {
                        .limit_edit_rate = false,
                        .edit_rate = 0.5f,
                        .color = {1.0f, 0.2f, 0.2f},
                    },

                    .custom_brush_settings = std::move(custom_brush_settings),
                    .custom_brush_settings_buffer = custom_brush_settings_buffer,
                    .custom_buffer_size = custom_buffer_size,
                    .custom_brush_settings_data = custom_brush_settings_data,
                });
        }
    }

    return result;
}
void App::reload_brushes() {
    device.wait_idle();
    for (auto &[key, brush] : brushes) {
        brush.cleanup(device);
    }
    brushes.clear();
    brushes = std::move(load_brushes());
    loop_task_list = std::move(record_loop_task_list());
}

App::~App() {
    ImNodes::DestroyContext();
    gvox_destroy_context(gvox_ctx);
    while (thread_pool.busy()) {
        std::this_thread::sleep_for(1ms);
    }
    thread_pool.stop();
    device.wait_idle();
    device.collect_garbage();
    device.destroy_image(optical_depth_image);
    device.destroy_sampler(optical_depth_sampler);
    device.destroy_buffer(gvox_model_buffer);
    device.destroy_buffer(gpu_globals_buffer);
    device.destroy_buffer(gpu_voxel_world_buffer);
    device.destroy_buffer(gpu_voxel_brush_buffer);
    device.destroy_buffer(gpu_input_buffer);
    device.destroy_buffer(gpu_indirect_dispatch_buffer);
    device.destroy_image(render_image);
    device.destroy_image(raytrace_output_image);
    for (auto &[key, brush] : brushes) {
        brush.cleanup(device);
    }
}

void imgui_help_marker(const char *const desc) {
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 25.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
};

void imgui_align_centered_for_width(float width, float alignment = 0.5f) {
    ImGuiStyle &style = ImGui::GetStyle();
    float avail = ImGui::GetContentRegionAvail().x;
    float off = (avail - width) * alignment;
    if (off > 0.0f)
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + off);
}

void App::imgui_gpu_input_flag_checkbox(char const *const str, u32 flag_index) {
    auto flag = get_flag(flag_index);
    ImGui::Checkbox(str, &flag);
    set_flag(flag_index, flag);
}

void App::settings_ui() {
    imgui_gpu_input_flag_checkbox("Use Persistent Thread Trace", GPU_INPUT_FLAG_INDEX_USE_PERSISTENT_THREAD_TRACE);
    ImGui::Checkbox("Battery Saving Mode", &battery_saving_mode);
    // auto prev_vsync = use_vsync;
    // ImGui::Checkbox("VSYNC", &use_vsync);
    // if (prev_vsync != use_vsync) {
    //     device.wait_idle();
    //     swapchain.change_present_mode(use_vsync ? daxa::PresentMode::DOUBLE_BUFFER_WAIT_FOR_VBLANK : daxa::PresentMode::DO_NOT_WAIT_FOR_VBLANK);
    // }
    ImGui::SliderFloat("FOV", &gpu_input.settings.fov, 0.01f, 170.0f);
    ImGui::SliderFloat("Daylight Cycle Time", &gpu_input.settings.daylight_cycle_time, -1.5f * std::numbers::pi_v<f32>, 2.5f * std::numbers::pi_v<f32>);
    ImGui::InputFloat("Mouse Sensitivity", &gpu_input.settings.sensitivity);
    ImGui::SliderFloat("Jitter Scale", &gpu_input.settings.jitter_scl, 0.0f, 1.0f);
    ImGui::SliderFloat("Frame Blending", &gpu_input.settings.frame_blending, 0.0f, 0.99f);
    ImGui::Checkbox("Use Custom Resolution", &use_custom_resolution);
    if (use_custom_resolution) {
        i32 custom_res[2] = {static_cast<i32>(render_size_x), static_cast<i32>(render_size_y)};
        ImGui::InputInt2("Resolution", custom_res);
        if (custom_res[0] != render_size_x || custom_res[1] != render_size_y) {
            render_size_x = custom_res[0];
            render_size_y = custom_res[1];
            recreate_render_images();
        }
    } else {
        auto prev_scl = render_resolution_scl;
        ImGui::SliderFloat("Resolution Scale", &render_resolution_scl, 0.1f, 1.0f);
        render_resolution_scl = std::round(render_resolution_scl * 40.0f) / 40.0f;
        if (prev_scl != render_resolution_scl) {
            render_size_x = size_x * render_resolution_scl;
            render_size_y = size_y * render_resolution_scl;
            recreate_render_images();
        }
    }
    if (ImGui::Button("Help")) {
        show_help_menu = !show_help_menu;
    }
    if (ImGui::Button("Reset Settings")) {
        reset_settings();
    }

    if (ImGui::TreeNode("Controls")) {
        if (ImGui::BeginTable("controls_table", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY, ImVec2(0, 250))) {
            ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 0.0f, 0);
            ImGui::TableSetupColumn("Key", ImGuiTableColumnFlags_WidthStretch, 0.0f, 1);
            ImGui::TableHeadersRow();
            for (usize i = 0; i < keys.size(); ++i) {
                ImGui::TableNextRow(ImGuiTableRowFlags_None);
                if (ImGui::TableSetColumnIndex(0)) {
                    ImGui::Text("%s", control_strings[i].data());
                }

                if (ImGui::TableSetColumnIndex(1)) {
                    if (i == new_key_index) {
                        ImGui::Button("<press any key>", ImVec2(-FLT_MIN, 0.0f));
                        if (ImGui::IsKeyDown(ImGuiKey_Escape)) {
                            new_key_index = GAME_KEY_LAST + 1;
                        } else {
                            for (i32 key_i = 0; key_i < 512; ++key_i) {
                                auto key_state = glfwGetKey(glfw_window_ptr, key_i);
                                if (key_state != GLFW_RELEASE && !controls_popup_is_open) {
                                    new_key_id = key_i;
                                    auto key_find_iter = std::find(keys.begin(), keys.end(), key_i);
                                    if (key_find_iter != keys.end()) {
                                        prev_key_id = keys[new_key_index];
                                        old_key_index = key_find_iter - keys.begin();
                                        if (old_key_index != new_key_index) {
                                            // new key to set, but already in bindings
                                            ImGui::OpenPopup("controls_popup_id");
                                            controls_popup_is_open = true;
                                        } else {
                                            // same key was pressed
                                            keys[new_key_index] = new_key_id;
                                            new_key_index = GAME_KEY_LAST + 1;
                                            save_settings();
                                        }
                                    } else {
                                        // new key to set
                                        keys[new_key_index] = new_key_id;
                                        new_key_index = GAME_KEY_LAST + 1;
                                        save_settings();
                                    }
                                    break;
                                }
                            }
                        }
                    } else {
                        auto key_name = get_key_string(keys[i]);
                        if (ImGui::Button(key_name, ImVec2(-FLT_MIN, 0.0f))) {
                            if (new_key_index == GAME_KEY_LAST + 1)
                                new_key_index = i;
                        }
                    }
                }
            }

            if (ImGui::BeginPopupModal("controls_popup_id", nullptr, ImGuiWindowFlags_Modal | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration)) {
                ImGui::Text("You're about to overwrite the binding of another key, Would you like to swap these keys?");
                if (ImGui::Button("YES") || ImGui::IsKeyDown(ImGuiKey_Enter)) {
                    keys[old_key_index] = prev_key_id;
                    keys[new_key_index] = new_key_id;
                    save_settings();
                    new_key_index = GAME_KEY_LAST + 1;
                    ImGui::CloseCurrentPopup();
                    controls_popup_is_open = false;
                }
                ImGui::SameLine();
                if (ImGui::Button("CANCEL") || ImGui::IsKeyDown(ImGuiKey_Escape)) {
                    new_key_index = GAME_KEY_LAST + 1;
                    ImGui::CloseCurrentPopup();
                    controls_popup_is_open = false;
                }
                ImGui::EndPopup();
            }

            ImGui::EndTable();
        }

        ImGui::TreePop();
    }
}

void App::brush_tool_ui() {
    imgui_align_centered_for_width(ImGui::CalcTextSize("Reload Brushes").x);
    if (ImGui::Button("Reload Brushes")) {
        reload_brushes();
    }

    i32 temp_int = static_cast<i32>(gpu_input.settings.brush_chunk_update_n);
    ImGui::InputInt("Update Rate", &temp_int);
    imgui_help_marker("The number of chunks in the brush region to update per frame. A value of 0 will update all");
    gpu_input.settings.brush_chunk_update_n = static_cast<u32>(std::min(std::max(temp_int, 0), BRUSH_CHUNK_N));
    imgui_gpu_input_flag_checkbox("Use Striped Brush Preview", GPU_INPUT_FLAG_INDEX_BRUSH_PREVIEW_OVERLAY);
    imgui_gpu_input_flag_checkbox("Show Brush Bounding Box", GPU_INPUT_FLAG_INDEX_SHOW_BRUSH_BOUNDING_BOX);

    {
        auto &current_brush = brushes.at(current_brush_key);
        imgui_align_centered_for_width(128.0f);
        if (ImGui::ImageButton(*reinterpret_cast<ImTextureID const *>(&current_brush.preview_thumbnail), ImVec2(128, 128)))
            ImGui::OpenPopup("brush_selection_popup");
    }

    ImGui::SetNextWindowSize(ImVec2(420, 0));
    if (ImGui::BeginPopup("brush_selection_popup")) {
        float window_visible_x2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
        ImGuiStyle &style = ImGui::GetStyle();
        auto brush_n = brushes.size();
        usize n = 0;
        for (auto const &[key, brush] : brushes) {
            auto button_sz = ImVec2(64, 64);
            ImGui::PushID(n);
            if (ImGui::ImageButton(*reinterpret_cast<ImTextureID const *>(&brush.preview_thumbnail), button_sz)) {
                auto &new_brush = brushes.at(key);
                new_brush.pipelines.compile();
                if (new_brush.pipelines.compiled) {
                    current_brush_key = key;
                }
                ImGui::CloseCurrentPopup();
            }
            imgui_help_marker(brush.display_name.c_str());
            float last_button_x2 = ImGui::GetItemRectMax().x;
            float next_button_x2 = last_button_x2 + style.ItemSpacing.x + button_sz.x;
            if (n + 1 < brush_n && next_button_x2 < window_visible_x2)
                ImGui::SameLine();
            ImGui::PopID();
            ++n;
        }
        ImGui::EndPopup();
    }

    auto &current_brush = brushes.at(current_brush_key);

    imgui_align_centered_for_width(ImGui::CalcTextSize("Use for chunkgen").x);
    bool use_for_chunkgen_disabled = chunkgen_brush_key == current_brush_key;
    if (use_for_chunkgen_disabled)
        ImGui::BeginDisabled();
    if (ImGui::Button("Use for chunkgen"))
        chunkgen_brush_key = current_brush_key;
    if (use_for_chunkgen_disabled)
        ImGui::EndDisabled();

    ImGui::Text("Brush Settings");
    ImGui::Checkbox("Limit Edit Rate", &current_brush.settings.limit_edit_rate);
    set_flag(GPU_INPUT_FLAG_INDEX_LIMIT_EDIT_RATE, current_brush.settings.limit_edit_rate);
    if (current_brush.settings.limit_edit_rate)
        ImGui::SliderFloat("Edit Rate", &current_brush.settings.edit_rate, 0.01f, 1.0f);
    gpu_input.settings.edit_rate = current_brush.settings.edit_rate;

    if (current_brush.key.filename() == "model") {
        ImGui::Text("%s Settings", current_brush.display_name.c_str());
        ImGui::InputText("File", &gvox_model_path);
        ImGui::InputText("File Type", &gvox_model_type);
        if (ImGui::Button("Reload Model"))
            should_upload_gvox_model = true;
    } else if (current_brush.custom_brush_settings.size() > 0) {
        ImGui::Text("%s Settings", current_brush.display_name.c_str());
    }

    usize parameter_offset = 0;
    for (auto const &parameter : current_brush.custom_brush_settings) {
        switch (parameter.id) {
        case UiComponentID::COLOR: {
            f32vec3 &p_color = *reinterpret_cast<f32vec3 *>(current_brush.custom_brush_settings_data + parameter_offset);
            ImGui::ColorEdit3(parameter.name.c_str(), reinterpret_cast<f32 *>(&p_color));
        } break;
        case UiComponentID::SLIDER_I32: {
            i32 &p_i32 = *reinterpret_cast<i32 *>(current_brush.custom_brush_settings_data + parameter_offset);
            auto &type_data = std::get<CustomUI_slider_i32>(parameter.type_data);
            ImGui::SliderInt(parameter.name.c_str(), &p_i32, type_data.min, type_data.max);
        } break;
        case UiComponentID::SLIDER_U32: {
            u32 &p_u32 = *reinterpret_cast<u32 *>(current_brush.custom_brush_settings_data + parameter_offset);
            auto &type_data = std::get<CustomUI_slider_u32>(parameter.type_data);
            i32 temp = static_cast<i32>(p_u32);
            ImGui::SliderInt(parameter.name.c_str(), &temp, type_data.min, type_data.max);
            p_u32 = static_cast<u32>(temp);
        } break;
        case UiComponentID::SLIDER_F32: {
            f32 &p_f32 = *reinterpret_cast<f32 *>(current_brush.custom_brush_settings_data + parameter_offset);
            auto &type_data = std::get<CustomUI_slider_f32>(parameter.type_data);
            ImGui::SliderFloat(parameter.name.c_str(), &p_f32, type_data.min, type_data.max);
        } break;
        case UiComponentID::SLIDER_F32VEC3: {
            f32vec3 &p_f32vec3 = *reinterpret_cast<f32vec3 *>(current_brush.custom_brush_settings_data + parameter_offset);
            auto &type_data = std::get<CustomUI_slider_f32vec3>(parameter.type_data);
            ImGui::SliderFloat3(parameter.name.c_str(), reinterpret_cast<f32 *>(&p_f32vec3), type_data.min, type_data.max);
        } break;
        case UiComponentID::INPUT_F32VEC3: {
            f32vec3 &p_f32vec3 = *reinterpret_cast<f32vec3 *>(current_brush.custom_brush_settings_data + parameter_offset);
            ImGui::InputFloat3(parameter.name.c_str(), reinterpret_cast<f32 *>(&p_f32vec3));
        } break;
        }
        parameter_offset += ui_component_sizes[static_cast<usize>(parameter.id)];
    }
}

void App::ui_update() {
    frametimes[frametime_rotation_index] = gpu_input.delta_time;
    frametime_rotation_index = (frametime_rotation_index + 1) % frametimes.size();

    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::PushFont(menu_font);

    // ImGui::ShowDemoWindow();

    if (show_menus) {
        ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoBackground;
        const ImGuiViewport *viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
        window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
        ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
        ImGui::Begin("DockSpace Demo", nullptr, window_flags);
        ImGui::PopStyleVar(3);
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("Settings")) {
                settings_ui();
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }
        ImGui::End();

        if (show_console)
            imgui_console.draw("Console", &show_console);

        if (show_tool_menu) {
            ImGui::Begin("Tools");
            for (u32 tool_i = 0; tool_i < GAME_TOOL_LAST + 1; ++tool_i) {
                if (ImGui::Button(tool_strings[tool_i].data())) {
                    current_tool = tool_i;
                }
            }
            ImGui::End();

            // ImGui::Begin("Node Editor");
            // ImNodes::BeginNodeEditor();

            // const int hardcoded_node_id = 1;
            // const int output_attr_id = 2;
            // static float value = 0.0f;

            // ImNodes::BeginNode(hardcoded_node_id);
            // ImNodes::BeginNodeTitleBar();
            // ImGui::TextUnformatted("My node");
            // ImNodes::EndNodeTitleBar();
            // ImNodes::BeginOutputAttribute(output_attr_id);
            // ImGui::Text("My output pin");
            // ImNodes::EndOutputAttribute();
            // ImNodes::EndNode();

            // ImNodes::EndNodeEditor();
            // ImGui::End();
        }

        if (show_tool_settings_menu) {
            ImGui::Begin("Tool Settings");
            if (current_tool == GAME_TOOL_BRUSH)
                brush_tool_ui();
            ImGui::End();
        }
    }

    if (show_debug_menu) {
        ImGui::PushFont(mono_font);
        const ImGuiViewport *viewport = ImGui::GetMainViewport();
        auto pos = viewport->WorkPos;
        pos.x += viewport->WorkSize.x - 220.0f;
        ImGui::SetNextWindowPos(pos);
        ImGui::Begin("Debug Menu", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDecoration);
        float average = 0.0f;
        for (auto frametime : frametimes)
            average += frametime;
        average /= static_cast<float>(frametimes.size());
        fmt_str.clear();
        fmt::format_to(std::back_inserter(fmt_str), "avg {:.2f} ms ({:.2f} fps)", average * 1000, 1.0f / average);
        ImGui::PlotLines("", frametimes.data(), static_cast<int>(frametimes.size()), static_cast<int>(frametime_rotation_index), fmt_str.c_str(), 0, 0.05f, ImVec2(0, 120.0f));
        auto device_props = device.properties();
        ImGui::Text("GPU: %s", device_props.device_name);
        ImGui::End();
        ImGui::PopFont();
    }

    if (show_help_menu) {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, {300.f, 360.f});
        ImGui::Begin("Help", &show_help_menu);
        ImGui::Text(R"(Controls:
ESCAPE to toggle pause (lock/unlock camera)
F1 for help
F3 to see debug info
CTRL+R to reload the game

* Brush Controls:
E Place Edit Origin (important for some brushes, like the Castle Wall)
SCROLL to increase/decrease brush size
LEFT MOUSE BUTTON to destroy voxels
RIGHT MOUSE BUTTON to place voxels
)");
        ImGui::End();
        ImGui::PopStyleVar();
    }

    ImGui::PopFont();
    ImGui::Render();
}

void App::on_update() {
    auto now = Clock::now();
    gpu_input.time = std::chrono::duration<f32>(now - start).count();
    gpu_input.delta_time = std::chrono::duration<f32>(now - prev_time).count();
    prev_time = now;

    gpu_input.frame_dim = {render_size_x, render_size_y};
    gpu_input.settings.tool_id = current_tool;
    set_flag(GPU_INPUT_FLAG_INDEX_PAUSED, show_menus);

    if (battery_saving_mode) {
        std::this_thread::sleep_for(10ms);
    }

    auto handle_reload_result = [this](daxa::Result<bool> const &result) {
        if (result.v.value_or(true)) {
            // std::cout << reload_result.to_string() << std::endl;
            if (result.is_err()) {
                imgui_console.add_log("[error] Failed to recompile a pipeline:");
                imgui_console.add_log("[error]   - %s", result.message().c_str());
            }
        }
    };

    {
        auto reload_result = basic_pipeline_manager.reload_all();
        handle_reload_result(reload_result);
    }
    {
        auto reload_result = chunkgen_pipeline_manager.reload_all();
        handle_reload_result(reload_result);
        if (reload_result.v.value_or(false))
            should_regenerate = true;
    }
    {
        auto reload_result = startup_pipeline_manager.reload_all();
        handle_reload_result(reload_result);
        if (reload_result.v.value_or(false))
            should_run_startup = true;
    }
    {
        auto reload_result = optical_depth_pipeline_manager.reload_all();
        handle_reload_result(reload_result);
        if (reload_result.v.value_or(false))
            should_regen_optical_depth = true;
    }

    ui_update();
    submit_task_list();

    gpu_input.mouse.pos_delta = {0.0f, 0.0f};
    gpu_input.mouse.scroll_delta = {0.0f, 0.0f};
}

void App::on_mouse_move(f32 x, f32 y) {
    f32vec2 center = {static_cast<f32>(size_x / 2), static_cast<f32>(size_y / 2)};
    gpu_input.mouse.pos = f32vec2{x, y};
    auto offset = gpu_input.mouse.pos - center;
    gpu_input.mouse.pos = gpu_input.mouse.pos * f32vec2{static_cast<f32>(render_size_x), static_cast<f32>(render_size_y)} / f32vec2{static_cast<f32>(size_x), static_cast<f32>(size_y)};
    if (!show_menus) {
        gpu_input.mouse.pos_delta = gpu_input.mouse.pos_delta + offset;
        set_mouse_pos(center.x, center.y);
    }
}
void App::on_mouse_scroll(f32 dx, f32 dy) {
    auto &io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        return;

    gpu_input.mouse.scroll_delta = gpu_input.mouse.scroll_delta + f32vec2{dx, dy};
}
void App::on_mouse_button(i32 button_id, i32 action) {
    auto &io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        return;

    auto mb_find_iter = std::find(mouse_buttons.begin(), mouse_buttons.end(), button_id);
    if (mb_find_iter != mouse_buttons.end()) {
        auto index = mb_find_iter - mouse_buttons.begin();
        gpu_input.mouse.buttons[index] = action;
    }
}
void App::on_key(i32 key_id, i32 action) {
    auto &io = ImGui::GetIO();
    if (io.WantCaptureKeyboard || new_key_index != GAME_KEY_LAST + 1)
        return;

    if (key_id == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        toggle_menus();
    if (key_id == GLFW_KEY_R && action == GLFW_PRESS) {
        if (glfwGetKey(glfw_window_ptr, GLFW_KEY_LEFT_CONTROL) != GLFW_RELEASE) {
            should_run_startup = true;
            start = Clock::now();
        } else {
            should_regenerate = true;
        }
    }
    if (key_id == GLFW_KEY_F1 && action == GLFW_PRESS)
        show_help_menu = !show_help_menu;
    if (key_id == GLFW_KEY_F3 && action == GLFW_PRESS)
        show_debug_menu = !show_debug_menu;
    if (key_id == GLFW_KEY_T && action == GLFW_PRESS)
        show_tool_menu = !show_tool_menu;
    if (key_id == GLFW_KEY_N && action == GLFW_PRESS)
        show_tool_settings_menu = !show_tool_settings_menu;
    if (key_id == GLFW_KEY_GRAVE_ACCENT && action == GLFW_PRESS)
        show_console = !show_console;
    if (key_id == GLFW_KEY_P && action == GLFW_PRESS)
        set_flag(GPU_INPUT_FLAG_INDEX_USE_PERSISTENT_THREAD_TRACE, !get_flag(GPU_INPUT_FLAG_INDEX_USE_PERSISTENT_THREAD_TRACE));

    // imgui_console.add_log("key_id = %d", key_id);

    auto key_find_iter = std::find(keys.begin(), keys.end(), key_id);
    if (key_find_iter != keys.end()) {
        auto index = key_find_iter - keys.begin();
        gpu_input.keyboard.keys[index] = action;
    }
}
void App::on_resize(u32 sx, u32 sy) {
    minimized = (sx == 0 || sy == 0);
    if (!minimized) {
        swapchain.resize();
        size_x = swapchain.get_surface_extent().x;
        size_y = swapchain.get_surface_extent().y;
        if (!use_custom_resolution) {
            render_size_x = size_x * render_resolution_scl;
            render_size_y = size_y * render_resolution_scl;
            recreate_render_images();
        }
        on_update();
    }
}
void App::recreate_render_images() {
    loop_task_list.remove_runtime_image(task_render_image, render_image);
    device.destroy_image(render_image);
    render_image = device.create_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {render_size_x, render_size_y, 1},
        .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::TRANSFER_SRC,
        .debug_name = APPNAME_PREFIX("render_image"),
    });
    loop_task_list.add_runtime_image(task_render_image, render_image);

    loop_task_list.remove_runtime_image(task_raytrace_output_image, raytrace_output_image);
    device.destroy_image(raytrace_output_image);
    raytrace_output_image = device.create_image({
        .format = daxa::Format::R32_SFLOAT,
        .size = {render_size_x, render_size_y, 1},
        .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE,
        .debug_name = APPNAME_PREFIX("raytrace_output_image"),
    });
    loop_task_list.add_runtime_image(task_raytrace_output_image, raytrace_output_image);
}
void App::toggle_menus() {
    set_mouse_capture(show_menus);
    gpu_input.mouse = {};
    gpu_input.keyboard = {};
    show_menus = !show_menus;
}

void App::submit_task_list() {
    loop_task_list.remove_runtime_image(task_swapchain_image, swapchain_image);
    swapchain_image = swapchain.acquire_next_image();
    loop_task_list.add_runtime_image(task_swapchain_image, swapchain_image);
    if (swapchain_image.is_empty())
        return;
    loop_task_list.execute();
}

void App::record_tasks(daxa::TaskList &new_task_list) {
    task_render_image = new_task_list.create_task_image({.debug_name = APPNAME_PREFIX("task_render_image")});
    new_task_list.add_runtime_image(task_render_image, render_image);
    task_gpu_input_buffer = new_task_list.create_task_buffer({.debug_name = APPNAME_PREFIX("task_gpu_input_buffer")});
    new_task_list.add_runtime_buffer(task_gpu_input_buffer, gpu_input_buffer);
    task_gpu_globals_buffer = new_task_list.create_task_buffer({.debug_name = APPNAME_PREFIX("task_gpu_globals_buffer")});
    new_task_list.add_runtime_buffer(task_gpu_globals_buffer, gpu_globals_buffer);
    task_gpu_voxel_world_buffer = new_task_list.create_task_buffer({.debug_name = APPNAME_PREFIX("task_gpu_voxel_world_buffer")});
    new_task_list.add_runtime_buffer(task_gpu_voxel_world_buffer, gpu_voxel_world_buffer);
    task_gpu_voxel_brush_buffer = new_task_list.create_task_buffer({.debug_name = APPNAME_PREFIX("task_gpu_voxel_brush_buffer")});
    new_task_list.add_runtime_buffer(task_gpu_voxel_brush_buffer, gpu_voxel_brush_buffer);
    task_gpu_indirect_dispatch_buffer = new_task_list.create_task_buffer({.debug_name = APPNAME_PREFIX("task_gpu_indirect_dispatch_buffer")});
    new_task_list.add_runtime_buffer(task_gpu_indirect_dispatch_buffer, gpu_indirect_dispatch_buffer);
    task_optical_depth_image = new_task_list.create_task_image({.debug_name = APPNAME_PREFIX("task_optical_depth_image")});
    new_task_list.add_runtime_image(task_optical_depth_image, optical_depth_image);
    task_gvox_model_buffer = new_task_list.create_task_buffer({.debug_name = APPNAME_PREFIX("task_gvox_model_buffer")});

    task_raytrace_output_image = new_task_list.create_task_image({.debug_name = APPNAME_PREFIX("task_raytrace_output_image")});
    new_task_list.add_runtime_image(task_raytrace_output_image, raytrace_output_image);

    daxa::UsedTaskImages thumbnail_upload_task_usages;
    daxa::UsedTaskImages imgui_task_usages;
    task_gpu_brush_settings_buffer = new_task_list.create_task_buffer({.debug_name = APPNAME_PREFIX("task_gpu_brush_settings_buffer")});
    for (auto &[key, brush] : brushes) {
        brush.task_preview_thumbnail = new_task_list.create_task_image({.debug_name = APPNAME_PREFIX("brush.task_preview_thumbnail")}),
        new_task_list.add_runtime_image(brush.task_preview_thumbnail, brush.preview_thumbnail);
        thumbnail_upload_task_usages.push_back({brush.task_preview_thumbnail, daxa::TaskImageAccess::TRANSFER_WRITE, {}});
        imgui_task_usages.push_back({brush.task_preview_thumbnail, daxa::TaskImageAccess::SHADER_READ_ONLY, {}});
        new_task_list.add_runtime_buffer(task_gpu_brush_settings_buffer, brush.custom_brush_settings_buffer);
    }

    new_task_list.add_task({
        .used_buffers = {
            {task_gvox_model_buffer, daxa::TaskBufferAccess::TRANSFER_WRITE},
        },
        .task = [=, this](daxa::TaskRuntime runtime) {
            if (should_upload_gvox_model) {
                auto cmd_list = runtime.get_command_list();
                if (!gvox_model_buffer.is_empty()) {
                    runtime.remove_runtime_buffer(task_gvox_model_buffer, gvox_model_buffer);
                    cmd_list.destroy_buffer_deferred(gvox_model_buffer);
                }
                gvox_destroy_scene(gvox_model);
                if (gvox_model_type == "gvox") {
                    gvox_model = gvox_load(gvox_ctx, gvox_model_path.c_str());
                } else {
                    gvox_model = gvox_load_raw(gvox_ctx, gvox_model_path.c_str(), gvox_model_type.c_str());
                }

                gvox_model_size = static_cast<u32>(sizeof(GVoxVoxel) * gvox_model.nodes[0].size_x * gvox_model.nodes[0].size_y * gvox_model.nodes[0].size_z + sizeof(u32) * 3);
                gvox_model_buffer = device.create_buffer({
                    .size = gvox_model_size,
                    .debug_name = APPNAME_PREFIX("gvox_model_buffer"),
                });
                runtime.add_runtime_buffer(task_gvox_model_buffer, gvox_model_buffer);
                cmd_list.pipeline_barrier({
                    .waiting_pipeline_access = daxa::AccessConsts::TRANSFER_WRITE,
                });
                auto staging_gvox_model_buffer = device.create_buffer({
                    .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .size = gvox_model_size,
                    .debug_name = APPNAME_PREFIX("staging_gvox_model_buffer"),
                });
                cmd_list.destroy_buffer_deferred(staging_gvox_model_buffer);
                GpuGVoxModel *buffer_ptr = device.get_host_address_as<GpuGVoxModel>(staging_gvox_model_buffer);
                buffer_ptr->size_x = static_cast<u32>(gvox_model.nodes[0].size_x);
                buffer_ptr->size_y = static_cast<u32>(gvox_model.nodes[0].size_y);
                buffer_ptr->size_z = static_cast<u32>(gvox_model.nodes[0].size_z);
                for (usize i = 0; i < buffer_ptr->size_x * buffer_ptr->size_y * buffer_ptr->size_z; ++i) {
                    auto &i_vox = gvox_model.nodes[0].voxels[i];
                    auto &o_vox = buffer_ptr->voxels[i];
                    o_vox.col.x = i_vox.color.x;
                    o_vox.col.y = i_vox.color.y;
                    o_vox.col.z = i_vox.color.z;
                    o_vox.id = i_vox.id;
                }
                cmd_list.copy_buffer_to_buffer({
                    .src_buffer = staging_gvox_model_buffer,
                    .dst_buffer = gvox_model_buffer,
                    .size = gvox_model_size,
                });
                should_upload_gvox_model = false;
            }
        },
        .debug_name = APPNAME_PREFIX("Upload GVOX Model"),
    });

    new_task_list.add_task({
        .used_buffers = {
            {task_gpu_brush_settings_buffer, daxa::TaskBufferAccess::TRANSFER_WRITE},
        },
        .used_images = thumbnail_upload_task_usages,
        .task = [=, this](daxa::TaskRuntime runtime) {
            auto cmd_list = runtime.get_command_list();
            u32 total_settings_size = 0;
            for (auto &[key, brush] : brushes) {
                total_settings_size += brush.custom_buffer_size;
            }
            auto image_staging_buffer = device.create_buffer({
                .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .size = static_cast<u32>((4 * 512 * 512) * brushes.size()),
            });
            auto settings_staging_buffer = device.create_buffer({
                .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .size = total_settings_size,
            });
            cmd_list.destroy_buffer_deferred(image_staging_buffer);
            cmd_list.destroy_buffer_deferred(settings_staging_buffer);
            auto *buffer_ptr = device.get_host_address_as<u8>(image_staging_buffer);
            auto *settings_buffer_ptr = device.get_host_address_as<u8>(settings_staging_buffer);
            u32 offset = 0;
            u32 settings_offset = 0;
            for (auto &[key, brush] : brushes) {
                if (!brush.thumbnail_needs_updating)
                    continue;
                i32 thumbnail_sx, thumbnail_sy;
                u8 *thumbnail_data = stbi_load(brush.thumbnail_image_path.string().c_str(), &thumbnail_sx, &thumbnail_sy, 0, 4);
                if (thumbnail_sx * thumbnail_sy > 512 * 512) {
                    imgui_console.add_log("[error] Image '%s' was too big! skipping...", brush.thumbnail_image_path.string().c_str());
                    brush.thumbnail_needs_updating = false;
                    stbi_image_free(thumbnail_data);
                    continue;
                }
                runtime.remove_runtime_image(brush.task_preview_thumbnail, brush.preview_thumbnail);
                if (!brush.preview_thumbnail.is_empty()) {
                    cmd_list.destroy_image_deferred(brush.preview_thumbnail);
                }
                brush.preview_thumbnail = device.create_image({
                    .format = daxa::Format::R8G8B8A8_SRGB,
                    .size = {static_cast<u32>(thumbnail_sx), static_cast<u32>(thumbnail_sy), 1},
                    .usage = daxa::ImageUsageFlagBits::SHADER_READ_ONLY | daxa::ImageUsageFlagBits::TRANSFER_DST,
                    .debug_name = brush.key.string() + " thumbnail image",
                });
                runtime.add_runtime_image(brush.task_preview_thumbnail, brush.preview_thumbnail);
                cmd_list.pipeline_barrier_image_transition({
                    .waiting_pipeline_access = daxa::AccessConsts::TRANSFER_WRITE,
                    .before_layout = daxa::ImageLayout::UNDEFINED,
                    .after_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                    .image_slice = {},
                    .image_id = brush.preview_thumbnail,
                });
                auto data_size = thumbnail_sx * thumbnail_sy * 4;
                memcpy(buffer_ptr + offset, thumbnail_data, data_size);
                stbi_image_free(thumbnail_data);
                cmd_list.copy_buffer_to_image({
                    .buffer = image_staging_buffer,
                    .buffer_offset = offset,
                    .image = brush.preview_thumbnail,
                    .image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                    .image_extent = {static_cast<u32>(thumbnail_sx), static_cast<u32>(thumbnail_sy), 1},
                });
                offset += data_size;
                brush.thumbnail_needs_updating = false;

                if (brush.custom_buffer_size > 0) {
                    memcpy(settings_buffer_ptr + settings_offset, brush.custom_brush_settings_data, brush.custom_buffer_size);
                    cmd_list.copy_buffer_to_buffer({
                        .src_buffer = settings_staging_buffer,
                        .src_offset = settings_offset,
                        .dst_buffer = brush.custom_brush_settings_buffer,
                        .dst_offset = 0,
                        .size = static_cast<u32>(brush.custom_buffer_size),
                    });
                    settings_offset += brush.custom_buffer_size;
                }
            }
        },
        .debug_name = APPNAME_PREFIX("Upload brush thumbnails and default settings"),
    });

    new_task_list.add_task({
        .used_buffers = {
            {task_gpu_input_buffer, daxa::TaskBufferAccess::TRANSFER_WRITE},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();

            auto staging_gpu_input_buffer = device.create_buffer({
                .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .size = sizeof(GpuInput),
                .debug_name = APPNAME_PREFIX("staging_gpu_input_buffer"),
            });
            cmd_list.destroy_buffer_deferred(staging_gpu_input_buffer);

            GpuInput *buffer_ptr = device.get_host_address_as<GpuInput>(staging_gpu_input_buffer);
            *buffer_ptr = this->gpu_input;

            cmd_list.copy_buffer_to_buffer({
                .src_buffer = staging_gpu_input_buffer,
                .dst_buffer = gpu_input_buffer,
                .size = sizeof(GpuInput),
            });
        },
        .debug_name = APPNAME_PREFIX("Input Transfer"),
    });
    new_task_list.add_task({
        .used_buffers = {
            {task_gpu_brush_settings_buffer, daxa::TaskBufferAccess::TRANSFER_WRITE},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            auto const &current_brush = brushes.at(current_brush_key);
            if (current_brush.custom_buffer_size > 0) {
                auto staging_gpu_brush_settings_buffer = device.create_buffer({
                    .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .size = static_cast<u32>(current_brush.custom_buffer_size),
                    .debug_name = APPNAME_PREFIX("staging_gpu_brush_settings_buffer"),
                });
                cmd_list.destroy_buffer_deferred(staging_gpu_brush_settings_buffer);
                GpuInput *buffer_ptr = device.get_host_address_as<GpuInput>(staging_gpu_brush_settings_buffer);
                memcpy(buffer_ptr, current_brush.custom_brush_settings_data, current_brush.custom_buffer_size);
                cmd_list.copy_buffer_to_buffer({
                    .src_buffer = staging_gpu_brush_settings_buffer,
                    .dst_buffer = current_brush.custom_brush_settings_buffer,
                    .size = static_cast<u32>(current_brush.custom_buffer_size),
                });
            }
        },
        .debug_name = APPNAME_PREFIX("Brush Settings Transfer"),
    });

    new_task_list.add_task({
        .used_buffers = {
            {task_gpu_globals_buffer, daxa::TaskBufferAccess::HOST_TRANSFER_WRITE},
            {task_gpu_voxel_world_buffer, daxa::TaskBufferAccess::HOST_TRANSFER_WRITE},
            {task_gpu_voxel_brush_buffer, daxa::TaskBufferAccess::HOST_TRANSFER_WRITE},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            if (!should_run_startup && should_regenerate) {
                auto cmd_list = task_runtime.get_command_list();
                cmd_list.clear_buffer({
                    .buffer = gpu_voxel_world_buffer,
                    .offset = offsetof(VoxelWorld, chunk_update_indices),
                    .size = offsetof(VoxelWorld, voxel_chunks) - offsetof(VoxelWorld, chunk_update_indices),
                    .clear_value = 0,
                });
                should_regenerate = false;
            }
            if (should_run_startup) {
                auto cmd_list = task_runtime.get_command_list();
                cmd_list.clear_buffer({
                    .buffer = gpu_globals_buffer,
                    .offset = 0,
                    .size = sizeof(GpuGlobals),
                    .clear_value = 0,
                });
                cmd_list.clear_buffer({
                    .buffer = gpu_voxel_brush_buffer,
                    .offset = 0,
                    .size = sizeof(VoxelBrush),
                    .clear_value = 0,
                });
            }
        },
        .debug_name = "Startup (Globals Clear)",
    });
    new_task_list.add_task({
        .used_buffers = {
            {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_voxel_world_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_voxel_brush_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            if (should_run_startup) {
                should_run_startup = false;
                auto cmd_list = task_runtime.get_command_list();
                cmd_list.set_pipeline(*startup_comp_pipeline);
                auto push = StartupCompPush{
                    .gpu_globals = this->device.get_device_address(gpu_globals_buffer),
                    .voxel_world = this->device.get_device_address(gpu_voxel_world_buffer),
                    .voxel_brush = this->device.get_device_address(gpu_voxel_brush_buffer),
                };
                cmd_list.push_constant(push);
                cmd_list.dispatch(1, 1, 1);
            }
        },
        .debug_name = "Startup (Compute)",
    });

    new_task_list.add_task({
        .used_images = {
            {task_optical_depth_image, daxa::TaskImageAccess::COMPUTE_SHADER_WRITE_ONLY, daxa::ImageMipArraySlice{}},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            if (should_regen_optical_depth) {
                should_regen_optical_depth = false;
                auto cmd_list = task_runtime.get_command_list();
                cmd_list.set_pipeline(*optical_depth_comp_pipeline);
                auto push = OpticalDepthCompPush{
                    .image_id = optical_depth_image.default_view(),
                };
                cmd_list.push_constant(push);
                cmd_list.dispatch(64, 64, 1);
            }
        },
        .debug_name = "OpticalDepth (Compute)",
    });

    new_task_list.add_task({
        .used_buffers = {
            {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_gpu_brush_settings_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_gpu_voxel_world_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_voxel_brush_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_indirect_dispatch_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            auto &current_brush = brushes.at(current_brush_key);
            // cmd_list.set_pipeline(*perframe_comp_pipeline);
            cmd_list.set_pipeline(*current_brush.pipelines.get_perframe_comp());
            u64 brush_settings_id = 0;
            if (current_brush.custom_buffer_size > 0)
                brush_settings_id = this->device.get_device_address(current_brush.custom_brush_settings_buffer);
            auto push = PerframeCompPush{
                .gpu_globals = this->device.get_device_address(gpu_globals_buffer),
                .gpu_input = this->device.get_device_address(gpu_input_buffer),
                .brush_settings = brush_settings_id,
                .voxel_world = this->device.get_device_address(gpu_voxel_world_buffer),
                .voxel_brush = this->device.get_device_address(gpu_voxel_brush_buffer),
                .gpu_indirect_dispatch = this->device.get_device_address(gpu_indirect_dispatch_buffer),
            };
            cmd_list.push_constant(push);
            cmd_list.dispatch(1, 1, 1);
        },
        .debug_name = "Perframe (Compute)",
    });

#if 1
    new_task_list.add_task({
        .used_buffers = {
            {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_gpu_brush_settings_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_gpu_voxel_world_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_voxel_brush_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gvox_model_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_gpu_indirect_dispatch_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            auto &current_brush = brushes.at(chunkgen_brush_key);
            cmd_list.set_pipeline(*current_brush.pipelines.get_chunkgen_comp());
            u64 brush_settings_id = 0;
            if (current_brush.custom_buffer_size > 0)
                brush_settings_id = this->device.get_device_address(current_brush.custom_brush_settings_buffer);
            cmd_list.push_constant(ChunkEditCompPush{
                .gpu_globals = device.get_device_address(gpu_globals_buffer),
                .gpu_input = this->device.get_device_address(gpu_input_buffer),
                .brush_settings = brush_settings_id,
                .voxel_world = this->device.get_device_address(gpu_voxel_world_buffer),
                .voxel_brush = this->device.get_device_address(gpu_voxel_brush_buffer),
                .gpu_gvox_model = device.get_device_address(gvox_model_buffer),
            });
            cmd_list.dispatch((CHUNK_SIZE + 7) / 8, (CHUNK_SIZE + 7) / 8, (CHUNK_SIZE + 7) / 8);
        },
        .debug_name = APPNAME_PREFIX("Chunkgen (Compute)"),
    });
    new_task_list.add_task({
        .used_buffers = {
            {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_gpu_brush_settings_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_gpu_voxel_world_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_voxel_brush_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_indirect_dispatch_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            auto &current_brush = brushes.at(current_brush_key);
            // cmd_list.set_pipeline(*brush_chunkgen_comp_pipeline);
            cmd_list.set_pipeline(*current_brush.pipelines.get_brush_chunkgen_comp());
            u64 brush_settings_id = 0;
            if (current_brush.custom_buffer_size > 0)
                brush_settings_id = this->device.get_device_address(current_brush.custom_brush_settings_buffer);
            cmd_list.push_constant(ChunkEditCompPush{
                .gpu_globals = device.get_device_address(gpu_globals_buffer),
                .gpu_input = this->device.get_device_address(gpu_input_buffer),
                .brush_settings = brush_settings_id,
                .voxel_world = this->device.get_device_address(gpu_voxel_world_buffer),
                .voxel_brush = this->device.get_device_address(gpu_voxel_brush_buffer),
                .gpu_gvox_model = device.get_device_address(gvox_model_buffer),
            });
            cmd_list.dispatch_indirect({.indirect_buffer = gpu_indirect_dispatch_buffer, .offset = offsetof(GpuIndirectDispatch, brush_chunk_dispatch)});
        },
        .debug_name = APPNAME_PREFIX("Brush Chunkgen (Compute)"),
    });
    new_task_list.add_task({
        .used_buffers = {
            {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_gpu_brush_settings_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_gpu_voxel_world_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_voxel_brush_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_gpu_indirect_dispatch_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_gvox_model_buffer, daxa::TaskBufferAccess::TRANSFER_WRITE},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            auto &current_brush = brushes.at(current_brush_key);
            // cmd_list.set_pipeline(*chunk_edit_comp_pipeline);
            cmd_list.set_pipeline(*current_brush.pipelines.get_chunk_edit_comp());
            u64 brush_settings_id = 0;
            if (current_brush.custom_buffer_size > 0)
                brush_settings_id = this->device.get_device_address(current_brush.custom_brush_settings_buffer);
            cmd_list.push_constant(ChunkEditCompPush{
                .gpu_globals = device.get_device_address(gpu_globals_buffer),
                .gpu_input = device.get_device_address(gpu_input_buffer),
                .brush_settings = brush_settings_id,
                .voxel_world = this->device.get_device_address(gpu_voxel_world_buffer),
                .voxel_brush = this->device.get_device_address(gpu_voxel_brush_buffer),
                .gpu_gvox_model = device.get_device_address(gvox_model_buffer),
            });
            cmd_list.dispatch_indirect({.indirect_buffer = gpu_indirect_dispatch_buffer, .offset = offsetof(GpuIndirectDispatch, chunk_edit_dispatch)});
        },
        .debug_name = APPNAME_PREFIX("Chunk Edit (Compute)"),
    });
    new_task_list.add_task({
        .used_buffers = {
            {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_indirect_dispatch_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_gpu_voxel_world_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_voxel_brush_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            cmd_list.set_pipeline(*subchunk_x2x4_comp_pipeline);
            cmd_list.push_constant(ChunkOptCompPush{
                .gpu_globals = device.get_device_address(gpu_globals_buffer),
                .voxel_world = this->device.get_device_address(gpu_voxel_world_buffer),
                .voxel_brush = this->device.get_device_address(gpu_voxel_brush_buffer),
            });
            cmd_list.dispatch_indirect({.indirect_buffer = gpu_indirect_dispatch_buffer, .offset = offsetof(GpuIndirectDispatch, subchunk_x2x4_dispatch)});
        },
        .debug_name = APPNAME_PREFIX("Subchunk x2x4 (Compute)"),
    });
    new_task_list.add_task({
        .used_buffers = {
            {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_indirect_dispatch_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_gpu_voxel_world_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_voxel_brush_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            cmd_list.set_pipeline(*subchunk_x8up_comp_pipeline);
            cmd_list.push_constant(ChunkOptCompPush{
                .gpu_globals = device.get_device_address(gpu_globals_buffer),
                .voxel_world = this->device.get_device_address(gpu_voxel_world_buffer),
                .voxel_brush = this->device.get_device_address(gpu_voxel_brush_buffer),
            });
            cmd_list.dispatch_indirect({.indirect_buffer = gpu_indirect_dispatch_buffer, .offset = offsetof(GpuIndirectDispatch, subchunk_x8up_dispatch)});
        },
        .debug_name = APPNAME_PREFIX("Subchunk x8up (Compute)"),
    });
    new_task_list.add_task({
        .used_buffers = {
            {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_voxel_world_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_voxel_brush_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            cmd_list.set_pipeline(*subchunk_brush_x2x4_comp_pipeline);
            cmd_list.push_constant(ChunkOptCompPush{
                .gpu_globals = device.get_device_address(gpu_globals_buffer),
                .voxel_world = this->device.get_device_address(gpu_voxel_world_buffer),
                .voxel_brush = this->device.get_device_address(gpu_voxel_brush_buffer),
            });
            cmd_list.dispatch_indirect({.indirect_buffer = gpu_indirect_dispatch_buffer, .offset = offsetof(GpuIndirectDispatch, brush_subchunk_x2x4_dispatch)});
        },
        .debug_name = APPNAME_PREFIX("Brush Subchunk x2x4 (Compute)"),
    });
    new_task_list.add_task({
        .used_buffers = {
            {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_voxel_world_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_voxel_brush_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            cmd_list.set_pipeline(*subchunk_brush_x8up_comp_pipeline);
            cmd_list.push_constant(ChunkOptCompPush{
                .gpu_globals = device.get_device_address(gpu_globals_buffer),
                .voxel_world = this->device.get_device_address(gpu_voxel_world_buffer),
                .voxel_brush = this->device.get_device_address(gpu_voxel_brush_buffer),
            });
            cmd_list.dispatch_indirect({.indirect_buffer = gpu_indirect_dispatch_buffer, .offset = offsetof(GpuIndirectDispatch, brush_subchunk_x8up_dispatch)});
        },
        .debug_name = APPNAME_PREFIX("Brush Subchunk x8up (Compute)"),
    });

#if USE_PERSISTENT_THREAD_TRACE
    // new_task_list.add_task({
    //     .used_images = {
    //         {task_raytrace_output_image, daxa::TaskImageAccess::TRANSFER_WRITE, daxa::ImageMipArraySlice{}},
    //     },
    //     .task = [this](daxa::TaskRuntime task_runtime) {
    //         auto cmd_list = task_runtime.get_command_list();
    //         cmd_list.clear_image({
    //             .dst_image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
    //             .clear_value = {std::array{0.0f, 0.0f, 0.0f, 0.0f}},
    //             .dst_image = raytrace_output_image,
    //         });
    //     },
    // });
    new_task_list.add_task({
        .used_buffers = {
            {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_gpu_voxel_world_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_gpu_voxel_brush_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
        },
        .used_images = {
            {task_raytrace_output_image, daxa::TaskImageAccess::COMPUTE_SHADER_WRITE_ONLY, daxa::ImageMipArraySlice{}},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            if (get_flag(GPU_INPUT_FLAG_INDEX_USE_PERSISTENT_THREAD_TRACE)) {
                auto cmd_list = task_runtime.get_command_list();
                cmd_list.set_pipeline(*raytrace_comp_pipeline);
                cmd_list.push_constant(RaytraceCompPush{
                    .gpu_globals = device.get_device_address(gpu_globals_buffer),
                    .gpu_input = device.get_device_address(gpu_input_buffer),
                    .voxel_world = this->device.get_device_address(gpu_voxel_world_buffer),
                    .voxel_brush = this->device.get_device_address(gpu_voxel_brush_buffer),
                    .raytrace_output_image_id = raytrace_output_image.default_view(),
                });
                cmd_list.dispatch_indirect({.indirect_buffer = gpu_indirect_dispatch_buffer, .offset = offsetof(GpuIndirectDispatch, raytrace_dispatch)});
                // cmd_list.dispatch(PERSISTENT_THREAD_TRACE_DISPATCH_SIZE, 1, 1);
            }
        },
        .debug_name = APPNAME_PREFIX("Raytrace (Compute)"),
    });
#endif

    new_task_list.add_task({
        .used_buffers = {
            {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_gpu_voxel_world_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_gpu_voxel_brush_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
        },
        .used_images = {
            {task_raytrace_output_image, daxa::TaskImageAccess::COMPUTE_SHADER_READ_ONLY, daxa::ImageMipArraySlice{}},
            {task_render_image, daxa::TaskImageAccess::COMPUTE_SHADER_WRITE_ONLY, daxa::ImageMipArraySlice{}},
            {task_optical_depth_image, daxa::TaskImageAccess::COMPUTE_SHADER_READ_ONLY, daxa::ImageMipArraySlice{}},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            cmd_list.set_pipeline(*draw_comp_pipeline);
            cmd_list.push_constant(DrawCompPush{
                .gpu_globals = device.get_device_address(gpu_globals_buffer),
                .gpu_input = device.get_device_address(gpu_input_buffer),
                .voxel_world = this->device.get_device_address(gpu_voxel_world_buffer),
                .voxel_brush = this->device.get_device_address(gpu_voxel_brush_buffer),
                .raytrace_output_image_id = raytrace_output_image.default_view(),
                .image_id = render_image.default_view(),
                .optical_depth_image_id = optical_depth_image.default_view(),
                .optical_depth_sampler_id = optical_depth_sampler,
            });
            cmd_list.dispatch((render_size_x + 7) / 8, (render_size_y + 7) / 8);
        },
        .debug_name = APPNAME_PREFIX("Draw (Compute)"),
    });
#endif

    new_task_list.add_task({
        .used_buffers = {
            {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
        },
        .used_images = {
            {task_render_image, daxa::TaskImageAccess::TRANSFER_READ, daxa::ImageMipArraySlice{}},
            {task_swapchain_image, daxa::TaskImageAccess::TRANSFER_WRITE, daxa::ImageMipArraySlice{}},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            cmd_list.blit_image_to_image({
                .src_image = render_image,
                .src_image_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
                .dst_image = swapchain_image,
                .dst_image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                .src_slice = {.image_aspect = daxa::ImageAspectFlagBits::COLOR},
                .src_offsets = {{{0, 0, 0}, {static_cast<i32>(render_size_x), static_cast<i32>(render_size_y), 1}}},
                .dst_slice = {.image_aspect = daxa::ImageAspectFlagBits::COLOR},
                .dst_offsets = {{{0, 0, 0}, {static_cast<i32>(size_x), static_cast<i32>(size_y), 1}}},
            });
        },
        .debug_name = APPNAME_PREFIX("Blit (render to swapchain)"),
    });

    imgui_task_usages.push_back({task_swapchain_image, daxa::TaskImageAccess::COLOR_ATTACHMENT, daxa::ImageMipArraySlice{}});

    new_task_list.add_task({
        .used_images = imgui_task_usages,
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            imgui_renderer.record_commands(ImGui::GetDrawData(), cmd_list, swapchain_image, size_x, size_y);
        },
        .debug_name = APPNAME_PREFIX("ImGui Task"),
    });
}
