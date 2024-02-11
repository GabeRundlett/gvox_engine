#include "voxel_app.hpp"

#include <thread>
#include <numbers>
#include <fstream>
#include <random>
#include <unordered_map>

#include <gvox/adapters/input/byte_buffer.h>
#include <gvox/adapters/output/byte_buffer.h>
#include <gvox/adapters/parse/voxlap.h>

// #include <voxels/gvox_model.inl>
#include <minizip/unzip.h>

static_assert(IsVoxelWorld<VoxelWorld>);

#define APPNAME "Voxel App"

using namespace std::chrono_literals;

#include <iostream>

void GpuResources::create(daxa::Device &device) {
    value_noise_image = device.create_image({
        .dimensions = 2,
        .format = daxa::Format::R8_UNORM,
        .size = {256, 256, 1},
        .array_layer_count = 256,
        .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
        .name = "value_noise_image",
    });
    blue_noise_vec2_image = device.create_image({
        .dimensions = 3,
        .format = daxa::Format::R8G8B8A8_UNORM,
        .size = {128, 128, 64},
        .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
        .name = "blue_noise_vec2_image",
    });
    input_buffer = device.create_buffer({
        .size = sizeof(GpuInput),
        .name = "input_buffer",
    });
    output_buffer = device.create_buffer({
        .size = sizeof(GpuOutput) * (FRAMES_IN_FLIGHT + 1),
        .name = "output_buffer",
    });
    staging_output_buffer = device.create_buffer({
        .size = sizeof(GpuOutput) * (FRAMES_IN_FLIGHT + 1),
        .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
        .name = "staging_output_buffer",
    });
    globals_buffer = device.create_buffer({
        .size = sizeof(GpuGlobals),
        .name = "globals_buffer",
    });
    gvox_model_buffer = device.create_buffer({
        .size = static_cast<daxa_u32>(offsetof(GpuGvoxModel, data)),
        .name = "gvox_model_buffer",
    });
    sampler_nnc = device.create_sampler({
        .magnification_filter = daxa::Filter::NEAREST,
        .minification_filter = daxa::Filter::NEAREST,
        .max_lod = 0.0f,
    });
    sampler_lnc = device.create_sampler({
        .magnification_filter = daxa::Filter::LINEAR,
        .minification_filter = daxa::Filter::NEAREST,
        .max_lod = 0.0f,
    });
    sampler_llc = device.create_sampler({
        .magnification_filter = daxa::Filter::LINEAR,
        .minification_filter = daxa::Filter::LINEAR,
        .max_lod = 0.0f,
    });
    sampler_llr = device.create_sampler({
        .magnification_filter = daxa::Filter::LINEAR,
        .minification_filter = daxa::Filter::LINEAR,
        .address_mode_u = daxa::SamplerAddressMode::REPEAT,
        .address_mode_v = daxa::SamplerAddressMode::REPEAT,
        .address_mode_w = daxa::SamplerAddressMode::REPEAT,
        .max_lod = 0.0f,
    });
}
void GpuResources::destroy(daxa::Device &device) const {
    device.destroy_image(value_noise_image);
    device.destroy_image(blue_noise_vec2_image);
    if (!debug_texture.is_empty()) {
        device.destroy_image(debug_texture);
    }
    device.destroy_buffer(input_buffer);
    device.destroy_buffer(output_buffer);
    device.destroy_buffer(staging_output_buffer);
    device.destroy_buffer(globals_buffer);
    if (!gvox_model_buffer.is_empty()) {
        device.destroy_buffer(gvox_model_buffer);
    }
    device.destroy_sampler(sampler_nnc);
    device.destroy_sampler(sampler_lnc);
    device.destroy_sampler(sampler_llc);
    device.destroy_sampler(sampler_llr);
}

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
      main_pipeline_manager{daxa::PipelineManagerInfo{
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
      }},
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
    gvox_ctx = gvox_create_context();

    renderer.create(device);
    gpu_resources.create(device);
    voxel_world.create(device);
    particles.create(device);

    task_input_buffer.set_buffers({.buffers = std::array{gpu_resources.input_buffer}});
    task_output_buffer.set_buffers({.buffers = std::array{gpu_resources.output_buffer}});
    task_staging_output_buffer.set_buffers({.buffers = std::array{gpu_resources.staging_output_buffer}});
    task_globals_buffer.set_buffers({.buffers = std::array{gpu_resources.globals_buffer}});
    task_gvox_model_buffer.set_buffers({.buffers = std::array{gpu_resources.gvox_model_buffer}});

    task_value_noise_image.set_images({.images = std::array{gpu_resources.value_noise_image}});
    task_blue_noise_vec2_image.set_images({.images = std::array{gpu_resources.blue_noise_vec2_image}});

    {
        daxa::TaskGraph temp_task_graph = daxa::TaskGraph({
            .device = device,
            .name = "temp_task_graph",
        });
        temp_task_graph.use_persistent_image(task_blue_noise_vec2_image);
        temp_task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, daxa::ImageViewType::REGULAR_2D, task_blue_noise_vec2_image),
            },
            .task = [this](daxa::TaskInterface const &ti) {
                auto staging_buffer = ti.device.create_buffer({
                    .size = static_cast<daxa_u32>(128 * 128 * 4 * 64 * 1),
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "staging_buffer",
                });
                auto *buffer_ptr = ti.device.get_host_address_as<uint8_t>(staging_buffer).value();
                auto *stbn_zip = unzOpen("assets/STBN.zip");
                for (auto i = 0; i < 64; ++i) {
                    [[maybe_unused]] int err = 0;
                    daxa_i32 size_x = 0;
                    daxa_i32 size_y = 0;
                    auto load_image = [&](char const *path, uint8_t *buffer_out_ptr) {
                        err = unzLocateFile(stbn_zip, path, 1);
                        assert(err == UNZ_OK);
                        auto file_info = unz_file_info{};
                        err = unzGetCurrentFileInfo(stbn_zip, &file_info, nullptr, 0, nullptr, 0, nullptr, 0);
                        assert(err == UNZ_OK);
                        auto file_data = std::vector<uint8_t>{};
                        file_data.resize(file_info.uncompressed_size);
                        err = unzOpenCurrentFile(stbn_zip);
                        assert(err == UNZ_OK);
                        err = unzReadCurrentFile(stbn_zip, file_data.data(), static_cast<uint32_t>(file_data.size()));
                        assert(err == file_data.size());

                        auto fi_mem = FreeImage_OpenMemory(file_data.data(), static_cast<DWORD>(file_data.size()));
                        auto fi_file_desc = FreeImage_GetFileTypeFromMemory(fi_mem, 0);
                        FIBITMAP *fi_bitmap = FreeImage_LoadFromMemory(fi_file_desc, fi_mem);
                        FreeImage_CloseMemory(fi_mem);
                        size_x = static_cast<int32_t>(FreeImage_GetWidth(fi_bitmap));
                        size_y = static_cast<int32_t>(FreeImage_GetHeight(fi_bitmap));
                        auto *temp_data = FreeImage_GetBits(fi_bitmap);
                        assert(temp_data != nullptr && "Failed to load image");
                        auto pixel_size = FreeImage_GetBPP(fi_bitmap);
                        if (pixel_size != 32) {
                            auto *temp = FreeImage_ConvertTo32Bits(fi_bitmap);
                            FreeImage_Unload(fi_bitmap);
                            fi_bitmap = temp;
                        }

                        if (temp_data != nullptr) {
                            assert(size_x == 128 && size_y == 128);
                            std::copy(temp_data + 0, temp_data + 128 * 128 * 4, buffer_out_ptr);
                        }
                        FreeImage_Unload(fi_bitmap);
                    };
                    auto vec2_name = std::string{"STBN/stbn_vec2_2Dx1D_128x128x64_"} + std::to_string(i) + ".png";
                    load_image(vec2_name.c_str(), buffer_ptr + (128 * 128 * 4) * i + (128 * 128 * 4 * 64) * 0);
                }

                ti.recorder.pipeline_barrier({
                    .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
                });
                ti.recorder.destroy_buffer_deferred(staging_buffer);
                ti.recorder.copy_buffer_to_image({
                    .buffer = staging_buffer,
                    .buffer_offset = (size_t{128} * 128 * 4 * 64) * 0,
                    .image = task_blue_noise_vec2_image.get_state().images[0],
                    .image_extent = {128, 128, 64},
                });
                needs_vram_calc = true;
            },
            .name = "upload_blue_noise",
        });
        temp_task_graph.submit({});
        temp_task_graph.complete({});
        temp_task_graph.execute({});
    }

    {
        daxa::TaskGraph temp_task_graph = daxa::TaskGraph({
            .device = device,
            .name = "temp_task_graph",
        });

        auto texture_path = "assets/debug.png";
        auto fi_file_desc = FreeImage_GetFileType(texture_path, 0);
        FIBITMAP *fi_bitmap = FreeImage_Load(fi_file_desc, texture_path);
        auto size_x = static_cast<uint32_t>(FreeImage_GetWidth(fi_bitmap));
        auto size_y = static_cast<uint32_t>(FreeImage_GetHeight(fi_bitmap));
        auto *temp_data = FreeImage_GetBits(fi_bitmap);
        assert(temp_data != nullptr && "Failed to load image");
        auto pixel_size = FreeImage_GetBPP(fi_bitmap);
        if (pixel_size != 32) {
            auto *temp = FreeImage_ConvertTo32Bits(fi_bitmap);
            FreeImage_Unload(fi_bitmap);
            fi_bitmap = temp;
        }
        auto size = static_cast<daxa_u32>(size_x) * static_cast<daxa_u32>(size_y) * 4 * 1;

        gpu_resources.debug_texture = device.create_image({
            .dimensions = 2,
            .format = daxa::Format::R8G8B8A8_UNORM,
            .size = {static_cast<daxa_u32>(size_x), static_cast<daxa_u32>(size_y), 1},
            .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
            .name = "debug_texture",
        });

        task_debug_texture.set_images({.images = std::array{gpu_resources.debug_texture}});
        temp_task_graph.use_persistent_image(task_debug_texture);
        temp_task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, daxa::ImageViewType::REGULAR_2D, task_debug_texture),
            },
            .task = [&, this](daxa::TaskInterface const &ti) {
                auto staging_buffer = ti.device.create_buffer({
                    .size = size,
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "staging_buffer",
                });
                auto *buffer_ptr = ti.device.get_host_address_as<uint8_t>(staging_buffer).value();
                std::copy(temp_data + 0, temp_data + size, buffer_ptr);
                FreeImage_Unload(fi_bitmap);

                ti.recorder.pipeline_barrier({
                    .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
                });
                ti.recorder.destroy_buffer_deferred(staging_buffer);
                ti.recorder.copy_buffer_to_image({
                    .buffer = staging_buffer,
                    .image = task_debug_texture.get_state().images[0],
                    .image_extent = {static_cast<daxa_u32>(size_x), static_cast<daxa_u32>(size_y), 1},
                });
                needs_vram_calc = true;
            },
            .name = "upload_debug_texture",
        });
        temp_task_graph.submit({});
        temp_task_graph.complete({});
        temp_task_graph.execute({});
    }

    constexpr auto IMMEDIATE_LOAD_MODEL_FROM_GABES_DRIVE = false;
    if constexpr (IMMEDIATE_LOAD_MODEL_FROM_GABES_DRIVE) {
        // ui.gvox_model_path = "C:/Users/gabe/AppData/Roaming/GabeVoxelGame/models/building.vox";
        // ui.gvox_model_path = "C:/dev/models/half-life/test.dae";
        ui.gvox_model_path = "C:/dev/models/Bistro_v5_2/BistroExterior.fbx";
        gvox_model_data = load_gvox_data();
        if (gvox_model_data.size != 0) {
            model_is_ready = true;
            prev_gvox_model_buffer = gpu_resources.gvox_model_buffer;
            gpu_resources.gvox_model_buffer = device.create_buffer({
                .size = static_cast<daxa_u32>(gvox_model_data.size),
                .name = "gvox_model_buffer",
            });
            task_gvox_model_buffer.set_buffers({.buffers = std::array{gpu_resources.gvox_model_buffer}});
        }
    }

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

    main_pipeline_manager.add_virtual_file({
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

    // main_pipeline_manager.add_virtual_file({
    //     .name = "gpu_resources",
    //     .contents = gpu_resources.get_shader_string(),
    // });

    main_task_graph = record_main_task_graph();
    main_pipeline_manager.wait();
    debug_utils::Console::add_log(std::format("startup: {} s\n", std::chrono::duration<float>(Clock::now() - start).count()));
}
VoxelApp::~VoxelApp() {
    gvox_destroy_context(gvox_ctx);
    device.wait_idle();
    device.collect_garbage();

    renderer.destroy(device);
    gpu_resources.destroy(device);
    voxel_world.destroy(device);
    particles.destroy(device);

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

auto VoxelApp::load_gvox_data_from_parser(GvoxAdapterContext *i_ctx, GvoxAdapterContext *p_ctx, GvoxRegionRange const *region_range) -> GvoxModelData {
    auto result = GvoxModelData{};
    GvoxByteBufferOutputAdapterConfig o_config = {
        .out_size = &result.size,
        .out_byte_buffer_ptr = &result.ptr,
        .allocate = nullptr,
    };
    GvoxAdapterContext *o_ctx = gvox_create_adapter_context(gvox_ctx, gvox_get_output_adapter(gvox_ctx, "byte_buffer"), &o_config);
    GvoxAdapterContext *s_ctx = gvox_create_adapter_context(gvox_ctx, gvox_get_serialize_adapter(gvox_ctx, "gvox_palette"), nullptr);

    {
        // time_t start = clock();
        gvox_blit_region(
            i_ctx, o_ctx, p_ctx, s_ctx,
            region_range,
            // GVOX_CHANNEL_BIT_COLOR | GVOX_CHANNEL_BIT_MATERIAL_ID | GVOX_CHANNEL_BIT_EMISSIVITY);
            GVOX_CHANNEL_BIT_COLOR);
        // time_t end = clock();
        // double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        // debug_utils::Console::add_log("{}s, new size: {} bytes", cpu_time_used, result.size);

        GvoxResult res = gvox_get_result(gvox_ctx);
        // int error_count = 0;
        while (res != GVOX_RESULT_SUCCESS) {
            size_t size = 0;
            gvox_get_result_message(gvox_ctx, nullptr, &size);
            char *str = new char[size + 1];
            gvox_get_result_message(gvox_ctx, str, nullptr);
            str[size] = '\0';
            debug_utils::Console::add_log(fmt::format("ERROR loading model: {}", str));
            gvox_pop_result(gvox_ctx);
            delete[] str;
            res = gvox_get_result(gvox_ctx);
            // ++error_count;
            return {};
        }
    }

    gvox_destroy_adapter_context(o_ctx);
    gvox_destroy_adapter_context(s_ctx);
    return result;
}

auto VoxelApp::load_gvox_data() -> GvoxModelData {
    auto result = GvoxModelData{};
    auto file = std::ifstream(ui.gvox_model_path, std::ios::binary);
    if (!file.is_open()) {
        debug_utils::Console::add_log("[error] Failed to load the model");
        ui.should_upload_gvox_model = false;
        return result;
    }
    file.seekg(0, std::ios_base::end);
    auto temp_gvox_model_size = static_cast<daxa_u32>(file.tellg());
    auto temp_gvox_model = std::vector<uint8_t>{};
    temp_gvox_model.resize(temp_gvox_model_size);
    {
        // time_t start = clock();
        file.seekg(0, std::ios_base::beg);
        file.read(reinterpret_cast<char *>(temp_gvox_model.data()), static_cast<std::streamsize>(temp_gvox_model_size));
        file.close();
        // time_t end = clock();
        // double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        // debug_utils::Console::add_log("(pulling file into memory: {}s)", cpu_time_used);
    }
    GvoxByteBufferInputAdapterConfig i_config = {
        .data = temp_gvox_model.data(),
        .size = temp_gvox_model_size,
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
        } else if (ext == ".rle") {
            gvox_model_type = "gvox_run_length_encoding";
        } else if (ext == ".oct") {
            gvox_model_type = "gvox_octree";
        } else if (ext == ".glp") {
            gvox_model_type = "gvox_global_palette";
        } else if (ext == ".brk") {
            gvox_model_type = "gvox_brickmap";
        } else if (ext == ".gvr") {
            gvox_model_type = "gvox_raw";
        } else if (ext == ".vxl") {
            i_config_ptr = &voxlap_config;
            gvox_model_type = "voxlap";
        } else {
            return open_mesh_model();
        }
    } else {
        return open_mesh_model();
    }
    GvoxAdapterContext *i_ctx = gvox_create_adapter_context(gvox_ctx, gvox_get_input_adapter(gvox_ctx, "byte_buffer"), &i_config);
    GvoxAdapterContext *p_ctx = gvox_create_adapter_context(gvox_ctx, gvox_get_parse_adapter(gvox_ctx, gvox_model_type), i_config_ptr);
    result = load_gvox_data_from_parser(i_ctx, p_ctx, nullptr);
    gvox_destroy_adapter_context(i_ctx);
    gvox_destroy_adapter_context(p_ctx);
    return result;
}

auto VoxelApp::open_mesh_model() -> GvoxModelData {
    MeshModel mesh_model;
    ::open_mesh_model(this->device, mesh_model, ui.gvox_model_path, "test");
    if (mesh_model.meshes.size() == 0) {
        debug_utils::Console::add_log("[error] Failed to load the mesh model");
        ui.should_upload_gvox_model = false;
        return {};
    }

    auto mesh_gpu_input = MeshGpuInput{};
    mesh_gpu_input.size = {768, 768, 768};
    mesh_gpu_input.bound_min = mesh_model.bound_min;
    mesh_gpu_input.bound_max = mesh_model.bound_max;

    daxa::SamplerId texture_sampler = device.create_sampler({
        .magnification_filter = daxa::Filter::LINEAR,
        .minification_filter = daxa::Filter::LINEAR,
        .address_mode_u = daxa::SamplerAddressMode::REPEAT,
        .address_mode_v = daxa::SamplerAddressMode::REPEAT,
        .address_mode_w = daxa::SamplerAddressMode::REPEAT,
        .name = "texture_sampler",
    });
    daxa::BufferId mesh_gpu_input_buffer = device.create_buffer(daxa::BufferInfo{
        .size = sizeof(MeshGpuInput),
        .name = "mesh_gpu_input_buffer",
    });
    daxa::BufferId voxel_buffer = device.create_buffer(daxa::BufferInfo{
        .size = static_cast<u32>(sizeof(u32) * mesh_gpu_input.size.x * mesh_gpu_input.size.y * mesh_gpu_input.size.z),
        .name = "voxel_buffer",
    });
    daxa::BufferId staging_voxel_buffer = device.create_buffer({
        .size = static_cast<u32>(sizeof(u32) * mesh_gpu_input.size.x * mesh_gpu_input.size.y * mesh_gpu_input.size.z),
        .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
        .name = "staging_voxel_buffer",
    });
    auto preprocess_pipeline = main_pipeline_manager.add_compute_pipeline({
        .shader_info = {
            .source = daxa::ShaderFile{"mesh/preprocess.comp.glsl"},
        },
        .push_constant_size = sizeof(MeshPreprocessPush),
        .name = "preprocess_pipeline",
    });
    auto raster_pipeline = main_pipeline_manager.add_raster_pipeline({
        .vertex_shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"mesh/voxelize.raster.glsl"},
            .compile_options = {.defines = {{"RASTER_VERT", "1"}}},
        },
        .fragment_shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"mesh/voxelize.raster.glsl"},
            .compile_options = {.defines = {{"RASTER_FRAG", "1"}}},
        },
        .raster = {
            // .samples = 8,
            .conservative_raster_info = daxa::ConservativeRasterInfo{
                .mode = daxa::ConservativeRasterizationMode::OVERESTIMATE,
                .size = 0.0f,
            },
        },
        .push_constant_size = sizeof(MeshRasterPush),
        .name = "raster_pipeline",
    });
    daxa::CommandSubmitInfo submit_info;
    daxa::TaskGraph task_list = daxa::TaskGraph({
        .device = device,
        .name = "mesh voxel conv",
    });
    auto task_gpu_input_buffer = daxa::TaskBuffer(daxa::TaskBufferInfo{.initial_buffers = {.buffers = std::array{mesh_gpu_input_buffer}}, .name = "task_gpu_input_buffer"});
    task_list.use_persistent_buffer(task_gpu_input_buffer);
    auto task_vertex_buffer_ids = std::vector<daxa::BufferId>{};
    auto task_image_ids = std::vector<daxa::ImageId>{};
    for (auto const &mesh : mesh_model.meshes) {
        task_vertex_buffer_ids.push_back(mesh.vertex_buffer);
        task_image_ids.push_back(mesh.textures[0]->image_id);
        // task_list.add_runtime_buffer(task_vertex_buffer, mesh.vertex_buffer);
        // task_list.add_runtime_image(task_image_id, mesh.textures[0]->image_id);
    }
    auto task_vertex_buffer = daxa::TaskBuffer(daxa::TaskBufferInfo{.initial_buffers = {.buffers = task_vertex_buffer_ids}, .name = "task_vertex_buffer"});
    auto task_image_id = daxa::TaskImage(daxa::TaskImageInfo{.initial_images = {.images = task_image_ids}, .name = "task_image_id"});
    auto task_voxel_buffer = daxa::TaskBuffer(daxa::TaskBufferInfo{.initial_buffers = {.buffers = std::array{voxel_buffer}}, .name = "task_voxel_buffer"});
    auto task_staging_voxel_buffer = daxa::TaskBuffer(daxa::TaskBufferInfo{.initial_buffers = {.buffers = std::array{staging_voxel_buffer}}, .name = "task_staging_voxel_buffer"});
    task_list.use_persistent_buffer(task_vertex_buffer);
    task_list.use_persistent_image(task_image_id);
    task_list.use_persistent_buffer(task_voxel_buffer);
    task_list.use_persistent_buffer(task_staging_voxel_buffer);
    task_list.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, task_gpu_input_buffer),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, task_vertex_buffer),
        },
        .task = [&](daxa::TaskInterface const &ti) {
            {
                auto staging_gpu_input_buffer = device.create_buffer({
                    .size = sizeof(MeshGpuInput),
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "staging_gpu_input_buffer",
                });
                ti.recorder.destroy_buffer_deferred(staging_gpu_input_buffer);
                auto *buffer_ptr = device.get_host_address_as<MeshGpuInput>(staging_gpu_input_buffer).value();
                *buffer_ptr = mesh_gpu_input;
                ti.recorder.copy_buffer_to_buffer({
                    .src_buffer = staging_gpu_input_buffer,
                    .dst_buffer = mesh_gpu_input_buffer,
                    .size = sizeof(MeshGpuInput),
                });
            }
            {
                usize vert_n = 0;
                for (auto const &mesh : mesh_model.meshes) {
                    vert_n += mesh.verts.size();
                }
                auto staging_vertex_buffer = device.create_buffer({
                    .size = static_cast<u32>(sizeof(MeshVertex) * vert_n),
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "staging_vertex_buffer",
                });
                ti.recorder.destroy_buffer_deferred(staging_vertex_buffer);
                auto *buffer_ptr = device.get_host_address_as<MeshVertex>(staging_vertex_buffer).value();
                usize vert_offset = 0;
                for (auto const &mesh : mesh_model.meshes) {
                    std::memcpy(buffer_ptr + vert_offset, mesh.verts.data(), sizeof(MeshVertex) * mesh.verts.size());
                    ti.recorder.copy_buffer_to_buffer({
                        .src_buffer = staging_vertex_buffer,
                        .dst_buffer = mesh.vertex_buffer,
                        .src_offset = sizeof(MeshVertex) * vert_offset,
                        .size = sizeof(MeshVertex) * mesh.verts.size(),
                    });
                    vert_offset += mesh.verts.size();
                }
            }
        },
        .name = "Input Transfer",
    });
    task_list.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, task_voxel_buffer),
        },
        .task = [&](daxa::TaskInterface const &ti) {
            ti.recorder.clear_buffer({
                .buffer = voxel_buffer,
                .offset = 0,
                .size = sizeof(u32) * mesh_gpu_input.size.x * mesh_gpu_input.size.y * mesh_gpu_input.size.z,
                .clear_value = {},
            });
        },
        .name = "Clear Output",
    });
    task_list.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::COMPUTE_SHADER_READ, task_gpu_input_buffer),
            daxa::inl_attachment(daxa::TaskBufferAccess::COMPUTE_SHADER_WRITE, task_vertex_buffer),
        },
        .task = [&](daxa::TaskInterface const &ti) {
            for (auto const &mesh : mesh_model.meshes) {
                if (!preprocess_pipeline.is_valid()) {
                    return;
                }
                ti.recorder.set_pipeline(preprocess_pipeline.get());
                set_push_constant(
                    ti,
                    MeshPreprocessPush{
                        .modl_mat = mesh.modl_mat,
                        .gpu_input = device.get_device_address(mesh_gpu_input_buffer).value(),
                        .vertex_buffer = device.get_device_address(mesh.vertex_buffer).value(),
                        .normal_buffer = device.get_device_address(mesh.normal_buffer).value(),
                        .triangle_count = static_cast<u32>(mesh.verts.size() / 3),
                    });
                ti.recorder.dispatch({.x = static_cast<u32>(mesh.verts.size() / 3)});
            }
        },
        .name = "Preprocess Verts",
    });
    task_list.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::FRAGMENT_SHADER_READ, task_gpu_input_buffer),
            daxa::inl_attachment(daxa::TaskBufferAccess::FRAGMENT_SHADER_READ, task_vertex_buffer),
            daxa::inl_attachment(daxa::TaskBufferAccess::FRAGMENT_SHADER_WRITE, task_voxel_buffer),
            daxa::inl_attachment(daxa::TaskImageAccess::FRAGMENT_SHADER_SAMPLED, daxa::ImageViewType::REGULAR_2D, task_image_id.view().view({.base_mip_level = 0, .level_count = 4})),
        },
        .task = [&](daxa::TaskInterface const &ti) {
            auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({.render_area = {.width = mesh_gpu_input.size.x, .height = mesh_gpu_input.size.y}});
            if (!raster_pipeline.is_valid()) {
                return;
            }
            renderpass_recorder.set_pipeline(raster_pipeline.get());
            for (auto const &mesh : mesh_model.meshes) {
                set_push_constant(
                    ti, renderpass_recorder,
                    MeshRasterPush{
                        .gpu_input = device.get_device_address(mesh_gpu_input_buffer).value(),
                        .vertex_buffer = device.get_device_address(mesh.vertex_buffer).value(),
                        .normal_buffer = device.get_device_address(mesh.normal_buffer).value(),
                        .voxel_buffer = device.get_device_address(voxel_buffer).value(),
                        .texture_id = {.value = static_cast<uint32_t>(mesh.textures[0]->image_id.default_view().index)},
                        .texture_sampler = texture_sampler,
                    });
                renderpass_recorder.draw({.vertex_count = static_cast<u32>(mesh.verts.size())});
            }
            ti.recorder = std::move(renderpass_recorder).end_renderpass();
        },
        .name = "Raster to Voxels",
    });
    task_list.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_READ, task_voxel_buffer),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, task_staging_voxel_buffer),
        },
        .task = [&](daxa::TaskInterface const &ti) {
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer = voxel_buffer,
                .dst_buffer = staging_voxel_buffer,
                .size = sizeof(u32) * mesh_gpu_input.size.x * mesh_gpu_input.size.y * mesh_gpu_input.size.z,
            });
        },
        .name = "Output Transfer",
    });
    task_list.submit({});
    task_list.complete({});
    task_list.execute({});
    device.wait_idle();
    auto cleanup = [&]() {
        device.wait_idle();
        device.destroy_buffer(mesh_gpu_input_buffer);
        for (auto &mesh : mesh_model.meshes) {
            device.destroy_buffer(mesh.vertex_buffer);
            device.destroy_buffer(mesh.normal_buffer);
        }
        for (auto &[key, value] : mesh_model.textures) {
            device.destroy_image(value->image_id);
        }
        device.destroy_buffer(voxel_buffer);
        device.destroy_buffer(staging_voxel_buffer);
        device.destroy_sampler(texture_sampler);
    };
    auto *buffer_ptr = device.get_host_address_as<u32>(staging_voxel_buffer).value();
    if (buffer_ptr == nullptr) {
        cleanup();
        debug_utils::Console::add_log("[error] Failed to voxelize the mesh model");
        ui.should_upload_gvox_model = false;
        return {};
    }

    struct GpuOutputState {
        uint32_t *buffer_ptr;
        daxa_u32vec3 size;
    };

    auto gpu_output_state = GpuOutputState{
        .buffer_ptr = buffer_ptr,
        .size = mesh_gpu_input.size,
    };

    GvoxParseAdapterInfo procedural_adapter_info = {
        .base_info = {
            .name_str = "gpu_result",
            .create = [](GvoxAdapterContext *ctx, void const *user_state_ptr) -> void {
                gvox_adapter_set_user_pointer(ctx, const_cast<void *>(user_state_ptr));
            },
            .destroy = [](GvoxAdapterContext *) -> void {},
            .blit_begin = [](GvoxBlitContext * /*unused*/, GvoxAdapterContext * /*unused*/, GvoxRegionRange const * /*unused*/, uint32_t /*unused*/) {},
            .blit_end = [](GvoxBlitContext * /*unused*/, GvoxAdapterContext * /*unused*/) {},
        },
        .query_details = []() -> GvoxParseAdapterDetails { return {.preferred_blit_mode = GVOX_BLIT_MODE_SERIALIZE_DRIVEN}; },
        .query_parsable_range = [](GvoxBlitContext * /*unused*/, GvoxAdapterContext * /*unused*/) -> GvoxRegionRange { return {{0, 0, 0}, {0, 0, 0}}; },
        .sample_region = [](GvoxBlitContext * /*unused*/, GvoxAdapterContext *ctx, GvoxRegion const * /*unused*/, GvoxOffset3D const *offset, uint32_t channel_id) -> GvoxSample {
            auto &[buffer_ptr_, size] = *static_cast<GpuOutputState *>(gvox_adapter_get_user_pointer(ctx));
            auto voxel_i = static_cast<size_t>(offset->x) + size.x * offset->y + size.x * size.y * offset->z;
            auto u32_voxel = buffer_ptr_[voxel_i];
            // float r = static_cast<float>((u32_voxel >> 0x00) & 0xff) / 255.0f;
            // float g = static_cast<float>((u32_voxel >> 0x08) & 0xff) / 255.0f;
            // float b = static_cast<float>((u32_voxel >> 0x10) & 0xff) / 255.0f;
            // uint32_t const id = (u32_voxel >> 0x18) & 0xff;
            switch (channel_id) {
            case GVOX_CHANNEL_ID_COLOR: return {u32_voxel, 1u};
            // case GVOX_CHANNEL_ID_NORMAL: return {0x0, 1u};
            // case GVOX_CHANNEL_ID_MATERIAL_ID: return {id, 1u};
            default:
                gvox_adapter_push_error(ctx, GVOX_RESULT_ERROR_PARSE_ADAPTER_INVALID_INPUT, "Tried sampling something other than color or normal");
                return {0u, 0u};
            }
            return {};
        },
        .query_region_flags = [](GvoxBlitContext * /*unused*/, GvoxAdapterContext * /*unused*/, GvoxRegionRange const * /*unused*/, uint32_t /*unused*/) -> uint32_t { return 0; },
        .load_region = [](GvoxBlitContext * /*unused*/, GvoxAdapterContext *ctx, GvoxRegionRange const *range, uint32_t channel_flags) -> GvoxRegion {
            GvoxRegion const region = {.range = *range, .channels = channel_flags, .flags = 0u, .data = nullptr};
            return region;
        },
        .unload_region = [](GvoxBlitContext * /*unused*/, GvoxAdapterContext * /*unused*/, GvoxRegion * /*unused*/) {},
        .parse_region = [](GvoxBlitContext *blit_ctx, GvoxAdapterContext *ctx, GvoxRegionRange const *range, uint32_t channel_flags) -> void {
            GvoxRegion const region = {.range = *range, .channels = channel_flags, .flags = 0u, .data = nullptr};
            gvox_emit_region(blit_ctx, &region);
        },
    };

    static auto parse_adapter = gvox_register_parse_adapter(gvox_ctx, &procedural_adapter_info);

    GvoxAdapterContext *p_ctx = gvox_create_adapter_context(gvox_ctx, parse_adapter, &gpu_output_state);
    GvoxRegionRange region_range = {
        .offset = {0, 0, 0},
        .extent = {
            mesh_gpu_input.size.x,
            mesh_gpu_input.size.y,
            mesh_gpu_input.size.z,
        },
    };
    auto result = load_gvox_data_from_parser(nullptr, p_ctx, &region_range);
    gvox_destroy_adapter_context(p_ctx);
    cleanup();
    if (result.size == 0) {
        ui.should_upload_gvox_model = false;
    }
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
        auto reload_result = main_pipeline_manager.reload_all();
        if (auto *reload_err = daxa::get_if<daxa::PipelineReloadError>(&reload_result)) {
            debug_utils::Console::add_log(reload_err->message);
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
        if (false) {
            // async model loading.. broken with mesh import
            if (!model_is_loading) {
                gvox_model_data_future = std::async(std::launch::async, &VoxelApp::load_gvox_data, this);
                model_is_loading = true;
            }
            if (model_is_loading && gvox_model_data_future.wait_for(0.01s) == std::future_status::ready) {
                model_is_ready = true;
                model_is_loading = false;
                gvox_model_data = gvox_model_data_future.get();
                prev_gvox_model_buffer = gpu_resources.gvox_model_buffer;
                gpu_resources.gvox_model_buffer = device.create_buffer({
                    .size = static_cast<daxa_u32>(gvox_model_data.size),
                    .name = "gvox_model_buffer",
                });
                task_gvox_model_buffer.set_buffers({.buffers = std::array{gpu_resources.gvox_model_buffer}});
            }
        } else {
            if (!model_is_loading) {
                model_is_loading = true;
            }
            if (model_is_loading) {
                model_is_loading = false;
                gvox_model_data = load_gvox_data();
                if (gvox_model_data.size != 0) {
                    model_is_ready = true;
                    prev_gvox_model_buffer = gpu_resources.gvox_model_buffer;
                    gpu_resources.gvox_model_buffer = device.create_buffer({
                        .size = static_cast<daxa_u32>(gvox_model_data.size),
                        .name = "gvox_model_buffer",
                    });
                    task_gvox_model_buffer.set_buffers({.buffers = std::array{gpu_resources.gvox_model_buffer}});
                }
            }
        }
    }

#if !IMMEDIATE_SKY
    if (ui.should_regenerate_sky) {
        auto sky_task_graph = daxa::TaskGraph({
            .device = device,
            .alias_transients = GVOX_ENGINE_INSTALL,
            .name = "sky_task_graph",
        });

        auto record_ctx = RecordContext{
            .device = this->device,
            .task_graph = sky_task_graph,
            .pipeline_manager = &main_pipeline_manager,
            .render_resolution = gpu_input.rounded_frame_dim,
            .output_resolution = gpu_input.output_resolution,
            .task_swapchain_image = task_swapchain_image,
            .compute_pipelines = &this->compute_pipelines,
            .raster_pipelines = &this->raster_pipelines,
        };

        record_ctx.task_input_buffer = task_input_buffer;
        sky_task_graph.use_persistent_buffer(task_input_buffer);

        auto sky_cube = generate_procedural_sky(record_ctx);
        sky.use_images(record_ctx);
        sky.render(record_ctx, sky_cube);

        sky_task_graph.submit({});
        sky_task_graph.complete({});
        sky_task_graph.execute({});

        ui.should_regenerate_sky = false;
    }
#endif

    // gpu_app_draw_ui();

    if (ui.should_run_startup || model_is_ready) {
        run_startup(main_task_graph);
    }
    if (model_is_ready) {
        upload_model(main_task_graph);
    }

    if (ui.should_record_task_graph) {
        device.wait_idle();
        main_task_graph = record_main_task_graph();
    }

    gpu_app_begin_frame(main_task_graph);

    gpu_input.fif_index = gpu_input.frame_index % (FRAMES_IN_FLIGHT + 1);
    // condition_values[static_cast<size_t>(Conditions::DYNAMIC_BUFFERS_REALLOC)] = should_realloc;
    // main_task_graph.execute({.permutation_condition_values = condition_values});
    main_task_graph.execute({});
    model_is_ready = false;

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

void VoxelApp::update_seeded_value_noise() {
    daxa::TaskGraph temp_task_graph = daxa::TaskGraph({
        .device = device,
        .name = "temp_task_graph",
    });
    temp_task_graph.use_persistent_image(task_value_noise_image);
    temp_task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, daxa::ImageViewType::REGULAR_2D, task_value_noise_image.view().view({.layer_count = 256})),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            auto staging_buffer = device.create_buffer({
                .size = static_cast<daxa_u32>(256 * 256 * 256 * 1),
                .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = "staging_buffer",
            });
            auto *buffer_ptr = device.get_host_address_as<uint8_t>(staging_buffer).value();
            std::mt19937_64 rng(std::hash<std::string>{}(ui.settings.world_seed_str));
            std::uniform_int_distribution<std::mt19937::result_type> dist(0, 255);
            for (daxa_u32 i = 0; i < (256 * 256 * 256 * 1); ++i) {
                buffer_ptr[i] = dist(rng) & 0xff;
            }
            ti.recorder.pipeline_barrier({
                .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
            });
            ti.recorder.destroy_buffer_deferred(staging_buffer);
            for (daxa_u32 i = 0; i < 256; ++i) {
                ti.recorder.copy_buffer_to_image({
                    .buffer = staging_buffer,
                    .buffer_offset = 256 * 256 * i,
                    .image = task_value_noise_image.get_state().images[0],
                    .image_slice{
                        .base_array_layer = i,
                        .layer_count = 1,
                    },
                    .image_extent = {256, 256, 1},
                });
            }
            needs_vram_calc = true;
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
void VoxelApp::run_startup(daxa::TaskGraph & /*unused*/) {
    auto temp_task_graph = daxa::TaskGraph({
        .device = device,
        .name = "temp_task_graph",
    });

    auto record_ctx = RecordContext{
        .device = this->device,
        .task_graph = temp_task_graph,
        .pipeline_manager = &main_pipeline_manager,
        .compute_pipelines = &this->compute_pipelines,
        .raster_pipelines = &this->raster_pipelines,
    };

    gpu_app_record_startup(record_ctx);

    temp_task_graph.submit({});
    temp_task_graph.complete({});
    temp_task_graph.execute({});

    ui.should_run_startup = false;
}

void VoxelApp::upload_model(daxa::TaskGraph & /*unused*/) {
    auto temp_task_graph = daxa::TaskGraph({
        .device = device,
        .name = "temp_task_graph",
    });
    temp_task_graph.use_persistent_buffer(task_gvox_model_buffer);
    temp_task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, task_gvox_model_buffer),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            if (!prev_gvox_model_buffer.is_empty()) {
                ti.recorder.destroy_buffer_deferred(prev_gvox_model_buffer);
            }
            ti.recorder.pipeline_barrier({
                .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
            });
            auto staging_gvox_model_buffer = device.create_buffer({
                .size = static_cast<daxa_u32>(gvox_model_data.size),
                .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = "staging_gvox_model_buffer",
            });
            ti.recorder.destroy_buffer_deferred(staging_gvox_model_buffer);
            char *buffer_ptr = device.get_host_address_as<char>(staging_gvox_model_buffer).value();
            std::copy(gvox_model_data.ptr, gvox_model_data.ptr + gvox_model_data.size, buffer_ptr);
            if (gvox_model_data.ptr != nullptr) {
                free(gvox_model_data.ptr);
            }
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer = staging_gvox_model_buffer,
                .dst_buffer = gpu_resources.gvox_model_buffer,
                .size = static_cast<daxa_u32>(gvox_model_data.size),
            });
            ui.should_upload_gvox_model = false;
            has_model = true;
            needs_vram_calc = true;
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
        .pipeline_manager = &main_pipeline_manager,
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

    buffer_size(gpu_resources.gvox_model_buffer);
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

    bool should_realloc = voxel_world.check_for_realloc(device, gpu_output.voxel_world);

    if (should_realloc) {
        gpu_app_dynamic_buffers_realloc();
    }
}

void VoxelApp::gpu_app_dynamic_buffers_realloc() {
    auto temp_task_graph = daxa::TaskGraph({
        .device = device,
        .name = "temp_task_graph",
    });

    voxel_world.dynamic_buffers_realloc(temp_task_graph, needs_vram_calc);

    temp_task_graph.submit({});
    temp_task_graph.complete({});
    temp_task_graph.execute({});
}

void VoxelApp::gpu_app_record_startup(RecordContext &record_ctx) {
    record_ctx.task_graph.use_persistent_buffer(task_input_buffer);
    record_ctx.task_graph.use_persistent_buffer(task_globals_buffer);

    voxel_world.use_buffers(record_ctx);

    voxel_world.record_startup(record_ctx);
    record_ctx.task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, task_globals_buffer),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            ti.recorder.clear_buffer({
                .buffer = task_globals_buffer.get_state().buffers[0],
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
            daxa::TaskViewVariant{std::pair{StartupCompute::gpu_input, task_input_buffer}},
            daxa::TaskViewVariant{std::pair{StartupCompute::globals, task_globals_buffer}},
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
    record_ctx.task_graph.use_persistent_image(task_value_noise_image);
    record_ctx.task_graph.use_persistent_image(task_blue_noise_vec2_image);
    record_ctx.task_graph.use_persistent_image(task_debug_texture);

    record_ctx.task_graph.use_persistent_buffer(task_input_buffer);
    record_ctx.task_graph.use_persistent_buffer(task_output_buffer);
    record_ctx.task_graph.use_persistent_buffer(task_staging_output_buffer);
    record_ctx.task_graph.use_persistent_buffer(task_globals_buffer);
    record_ctx.task_graph.use_persistent_buffer(task_gvox_model_buffer);

    voxel_world.use_buffers(record_ctx);
    particles.use_buffers(record_ctx);

    record_ctx.task_blue_noise_vec2_image = task_blue_noise_vec2_image;
    record_ctx.task_debug_texture = task_debug_texture;
    record_ctx.task_input_buffer = task_input_buffer;
    record_ctx.task_globals_buffer = task_globals_buffer;

    record_ctx.task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, task_input_buffer),
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
                .dst_buffer = task_input_buffer.get_state().buffers[0],
                .size = sizeof(GpuInput),
            });
        },
        .name = "GpuInputUploadTransferTask",
    });

    // TODO: Refactor this. I hate that due to these tasks, I have to call this "use_buffers" thing above.
    record_ctx.add(ComputeTask<PerframeCompute, PerframeComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"app.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{PerframeCompute::gpu_input, task_input_buffer}},
            daxa::TaskViewVariant{std::pair{PerframeCompute::gpu_output, task_output_buffer}},
            daxa::TaskViewVariant{std::pair{PerframeCompute::globals, task_globals_buffer}},
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
    voxel_world.record_frame(record_ctx, task_gvox_model_buffer, task_value_noise_image);

    renderer.render(record_ctx, voxel_world.buffers, particles, record_ctx.task_swapchain_image, swapchain.get_format());

    record_ctx.task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_READ, task_output_buffer),
            daxa::inl_attachment(daxa::TaskBufferAccess::HOST_TRANSFER_WRITE, task_staging_output_buffer),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            auto output_buffer = task_output_buffer.get_state().buffers[0];
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
