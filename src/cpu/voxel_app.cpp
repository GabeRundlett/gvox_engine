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

#include <minizip/unzip.h>

constexpr auto round_frame_dim(u32vec2 size) {
    constexpr auto over_estimation = u32vec2{32, 32};
    auto result = (size + u32vec2{over_estimation.x - 1u, over_estimation.y - 1u}) / over_estimation * over_estimation;
    // not necessary, since it rounds up!
    // result = {std::max(result.x, over_estimation.x), std::max(result.y, over_estimation.y)};
    return result;
}

void VoxelChunks::create(daxa::Device &device, u32 log2_chunks_per_axis) {
    auto chunk_n = (1u << log2_chunks_per_axis);
    chunk_n = chunk_n * chunk_n * chunk_n;
    buffer = device.create_buffer({
        .size = static_cast<u32>(sizeof(VoxelLeafChunk)) * chunk_n,
        .name = "voxel_chunks_buffer",
    });
}
void VoxelChunks::destroy(daxa::Device &device) const {
    if (!buffer.is_empty()) {
        device.destroy_buffer(buffer);
    }
}

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
    temp_voxel_chunks_buffer = device.create_buffer({
        .size = sizeof(TempVoxelChunk) * MAX_CHUNK_UPDATES_PER_FRAME,
        .name = "temp_voxel_chunks_buffer",
    });
    gvox_model_buffer = device.create_buffer({
        .size = static_cast<u32>(offsetof(GpuGvoxModel, data)),
        .name = "gvox_model_buffer",
    });
    simulated_voxel_particles_buffer = device.create_buffer({
        .size = sizeof(SimulatedVoxelParticle) * std::max<u32>(MAX_SIMULATED_VOXEL_PARTICLES, 1),
        .name = "simulated_voxel_particles_buffer",
    });
    rendered_voxel_particles_buffer = device.create_buffer({
        .size = sizeof(u32) * std::max<u32>(MAX_RENDERED_VOXEL_PARTICLES, 1),
        .name = "rendered_voxel_particles_buffer",
    });
    placed_voxel_particles_buffer = device.create_buffer({
        .size = sizeof(u32) * std::max<u32>(MAX_SIMULATED_VOXEL_PARTICLES, 1),
        .name = "placed_voxel_particles_buffer",
    });
    final_image_sampler = device.create_sampler({
        .magnification_filter = daxa::Filter::LINEAR,
        .minification_filter = daxa::Filter::NEAREST,
        .max_lod = 0.0f,
    });
    value_noise_sampler = device.create_sampler({
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
    device.destroy_buffer(input_buffer);
    device.destroy_buffer(output_buffer);
    device.destroy_buffer(staging_output_buffer);
    device.destroy_buffer(globals_buffer);
    device.destroy_buffer(temp_voxel_chunks_buffer);
    voxel_chunks.destroy(device);
    voxel_malloc.destroy(device);
    if (!gvox_model_buffer.is_empty()) {
        device.destroy_buffer(gvox_model_buffer);
    }
    voxel_leaf_chunk_malloc.destroy(device);
    voxel_parent_chunk_malloc.destroy(device);
    device.destroy_buffer(simulated_voxel_particles_buffer);
    device.destroy_buffer(rendered_voxel_particles_buffer);
    device.destroy_buffer(placed_voxel_particles_buffer);
    device.destroy_sampler(final_image_sampler);
    device.destroy_sampler(value_noise_sampler);
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
      daxa_instance{daxa::create_instance({.enable_validation = false})},
      device{daxa_instance.create_device({
          // .enable_buffer_device_address_capture_replay = false,
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
      gbuffer_renderer{main_pipeline_manager},
      reprojection_renderer{main_pipeline_manager},
      ssao_renderer{main_pipeline_manager, gpu_resources.final_image_sampler},
      shadow_renderer{main_pipeline_manager},
      gpu_resources{},
      // clang-format off
      startup_task_state{main_pipeline_manager},
      perframe_task_state{main_pipeline_manager},
      chunk_edit_task_state{main_pipeline_manager},
      chunk_opt_x2x4_task_state{main_pipeline_manager},
      chunk_opt_x8up_task_state{main_pipeline_manager},
      chunk_alloc_task_state{main_pipeline_manager},
      postprocessing_task_state{main_pipeline_manager, gpu_resources.final_image_sampler, swapchain.get_format()},
      per_chunk_task_state{main_pipeline_manager},
      voxel_particle_sim_task_state{main_pipeline_manager},
      voxel_particle_raster_task_state{main_pipeline_manager},
      // clang-format on
      main_task_graph{[this]() {
          gpu_resources.create(device);
          gpu_resources.voxel_chunks.create(device, ui.settings.log2_chunks_per_axis);
          gpu_resources.voxel_malloc.create(device);
          gpu_resources.voxel_leaf_chunk_malloc.create(device);
          gpu_resources.voxel_parent_chunk_malloc.create(device);
          return record_main_task_graph();
      }()},
      gvox_ctx(gvox_create_context()) {

    start = Clock::now();

    constexpr auto IMMEDIATE_LOAD_MODEL_FROM_GABES_DRIVE = false;
    if constexpr (IMMEDIATE_LOAD_MODEL_FROM_GABES_DRIVE) {
        ui.gvox_model_path = "C:/Users/gabe/AppData/Roaming/GabeVoxelGame/models/building.vox";
        gvox_model_data = load_gvox_data();
        model_is_ready = true;

        prev_gvox_model_buffer = gpu_resources.gvox_model_buffer;
        gpu_resources.gvox_model_buffer = device.create_buffer({
            .size = static_cast<u32>(gvox_model_data.size),
            .name = "gvox_model_buffer",
        });
        task_gvox_model_buffer.set_buffers({.buffers = std::array{gpu_resources.gvox_model_buffer}});
    }

    {
        daxa::TaskGraph temp_task_graph = daxa::TaskGraph({
            .device = device,
            .name = "temp_task_graph",
        });
        temp_task_graph.use_persistent_image(task_blue_noise_vec2_image);
        temp_task_graph.add_task({
            .uses = {
                daxa::TaskImageUse<daxa::TaskImageAccess::TRANSFER_WRITE>{task_blue_noise_vec2_image},
            },
            .task = [this](daxa::TaskInterface task_runtime) {
                auto staging_buffer = device.create_buffer({
                    .size = static_cast<u32>(128 * 128 * 4 * 64 * 1),
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "staging_buffer",
                });
                auto *buffer_ptr = device.get_host_address_as<u8>(staging_buffer);
                auto *stbn_zip = unzOpen("assets/STBN.zip");
                for (auto i = 0; i < 64; ++i) {
                    [[maybe_unused]] int err = 0;
                    i32 size_x = 0;
                    i32 size_y = 0;
                    i32 channel_n = 0;
                    auto load_image = [&](char const *path, u8 *buffer_out_ptr) {
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
                        auto *temp_data = stbi_load_from_memory(file_data.data(), static_cast<int>(file_data.size()), &size_x, &size_y, &channel_n, 4);
                        if (temp_data != nullptr) {
                            assert(size_x == 128 && size_y == 128);
                            std::copy(temp_data + 0, temp_data + 128 * 128 * 4, buffer_out_ptr);
                        }
                    };
                    auto vec2_name = std::string{"STBN/stbn_vec2_2Dx1D_128x128x64_"} + std::to_string(i) + ".png";
                    load_image(vec2_name.c_str(), buffer_ptr + (128 * 128 * 4) * i + (128 * 128 * 4 * 64) * 0);
                }

                auto cmd_list = task_runtime.get_command_list();
                cmd_list.pipeline_barrier({
                    .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
                });
                cmd_list.destroy_buffer_deferred(staging_buffer);
                cmd_list.copy_buffer_to_image({
                    .buffer = staging_buffer,
                    .buffer_offset = (usize{128} * 128 * 4 * 64) * 0,
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
    gpu_resources.destroy(device);
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

void VoxelApp::calc_vram_usage() {
    auto &result_size = ui.debug_vram_usage;
    ui.debug_gpu_resource_infos.clear();
    result_size = 0;

    auto format_to_pixel_size = [](daxa::Format format) -> u32 {
        switch (format) {
        case daxa::Format::R16G16B16_SFLOAT: return 3 * 2;
        case daxa::Format::R16G16B16A16_SFLOAT: return 4 * 2;
        case daxa::Format::R32G32B32_SFLOAT: return 3 * 4;
        default:
        case daxa::Format::R32G32B32A32_SFLOAT: return 4 * 4;
        }
    };

    auto image_size = [this, &format_to_pixel_size, &result_size](daxa::ImageId image) {
        if (image.is_empty()) {
            return;
        }
        auto image_info = device.info_image(image);
        auto size = format_to_pixel_size(image_info.format) * image_info.size.x * image_info.size.y * image_info.size.z;
        ui.debug_gpu_resource_infos.push_back({
            .type = "image",
            .name = image_info.name,
            .size = size,
        });
        result_size += size;
    };
    auto buffer_size = [this, &result_size](daxa::BufferId buffer) {
        if (buffer.is_empty()) {
            return;
        }
        auto buffer_info = device.info_buffer(buffer);
        ui.debug_gpu_resource_infos.push_back({
            .type = "buffer",
            .name = buffer_info.name,
            .size = buffer_info.size,
        });
        result_size += buffer_info.size;
    };

    buffer_size(gpu_resources.input_buffer);
    buffer_size(gpu_resources.globals_buffer);
    buffer_size(gpu_resources.temp_voxel_chunks_buffer);
    buffer_size(gpu_resources.voxel_chunks.buffer);

    gpu_resources.voxel_malloc.for_each_buffer(buffer_size);
    gpu_resources.voxel_leaf_chunk_malloc.for_each_buffer(buffer_size);
    gpu_resources.voxel_parent_chunk_malloc.for_each_buffer(buffer_size);

    buffer_size(gpu_resources.gvox_model_buffer);
    buffer_size(gpu_resources.simulated_voxel_particles_buffer);
    buffer_size(gpu_resources.rendered_voxel_particles_buffer);
    buffer_size(gpu_resources.placed_voxel_particles_buffer);

    needs_vram_calc = false;
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

    {
        auto reload_result = main_pipeline_manager.reload_all();
        if (auto *reload_err = std::get_if<daxa::PipelineReloadError>(&reload_result)) {
            AppUi::Console::s_instance->add_log(reload_err->message);
        }
    }

    task_swapchain_image.set_images({.images = {&swapchain_image, 1}});
    if (swapchain_image.is_empty()) {
        return;
    }

    if (ui.should_recreate_voxel_buffers) {
        recreate_voxel_chunks();
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
            prev_gvox_model_buffer = gpu_resources.gvox_model_buffer;
            gpu_resources.gvox_model_buffer = device.create_buffer({
                .size = static_cast<u32>(gvox_model_data.size),
                .name = "gvox_model_buffer",
            });
            task_gvox_model_buffer.set_buffers({.buffers = std::array{gpu_resources.gvox_model_buffer}});
        }
    }

    if (needs_vram_calc) {
        calc_vram_usage();
    }

    gpu_resources.voxel_malloc.check_for_realloc(device, gpu_output.voxel_malloc_output.current_element_count);
    gpu_resources.voxel_leaf_chunk_malloc.check_for_realloc(device, gpu_output.voxel_leaf_chunk_output.current_element_count);
    gpu_resources.voxel_parent_chunk_malloc.check_for_realloc(device, gpu_output.voxel_parent_chunk_output.current_element_count);

    auto should_realloc =
        gpu_resources.voxel_malloc.needs_realloc() ||
        gpu_resources.voxel_leaf_chunk_malloc.needs_realloc() ||
        gpu_resources.voxel_parent_chunk_malloc.needs_realloc();

    if (ui.should_run_startup || model_is_ready) {
        run_startup(main_task_graph);
    }
    if (model_is_ready) {
        upload_model(main_task_graph);
    }
    if (should_realloc) {
        dynamic_buffers_realloc(main_task_graph);
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
    ui.debug_page_count = gpu_resources.voxel_malloc.current_element_count;
    ui.debug_job_counters = std::bit_cast<ChunkHierarchyJobCounters>(gpu_output.job_counters_packed);
    ui.debug_total_jobs_ran = gpu_output.total_jobs_ran;

    // task_render_pos_image.swap_images(task_render_prev_pos_image);
    // task_render_col_image.swap_images(task_render_prev_col_image);

    gbuffer_renderer.next_frame();
    ssao_renderer.next_frame();
    shadow_renderer.next_frame();

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
            recreate_render_images();
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

void VoxelApp::recreate_render_images() {
    device.wait_idle();
    needs_vram_calc = true;
}
void VoxelApp::recreate_voxel_chunks() {
    gpu_resources.voxel_chunks.destroy(device);
    gpu_resources.voxel_chunks.create(device, ui.settings.log2_chunks_per_axis);
    task_voxel_chunks_buffer.set_buffers({.buffers = {&gpu_resources.voxel_chunks.buffer, 1}});
    ui.should_recreate_voxel_buffers = false;
    needs_vram_calc = true;
}

void VoxelApp::update_seeded_value_noise() {
    daxa::TaskGraph temp_task_graph = daxa::TaskGraph({
        .device = device,
        .name = "temp_task_graph",
    });
    temp_task_graph.use_persistent_image(task_value_noise_image);
    temp_task_graph.add_task({
        .uses = {
            daxa::TaskImageUse<daxa::TaskImageAccess::TRANSFER_WRITE>{task_value_noise_image.view().view({.layer_count = 256})},
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
void VoxelApp::run_startup(daxa::TaskGraph &) {
    auto temp_task_graph = daxa::TaskGraph({
        .device = device,
        .name = "temp_task_graph",
    });
    temp_task_graph.use_persistent_buffer(task_input_buffer);
    temp_task_graph.use_persistent_buffer(task_globals_buffer);
    temp_task_graph.use_persistent_buffer(task_temp_voxel_chunks_buffer);
    temp_task_graph.use_persistent_buffer(task_voxel_chunks_buffer);
    gpu_resources.voxel_malloc.for_each_task_buffer([&temp_task_graph](auto &task_buffer) { temp_task_graph.use_persistent_buffer(task_buffer); });
    gpu_resources.voxel_leaf_chunk_malloc.for_each_task_buffer([&temp_task_graph](auto &task_buffer) { temp_task_graph.use_persistent_buffer(task_buffer); });
    gpu_resources.voxel_parent_chunk_malloc.for_each_task_buffer([&temp_task_graph](auto &task_buffer) { temp_task_graph.use_persistent_buffer(task_buffer); });
    temp_task_graph.add_task({
        .uses = {
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{task_globals_buffer},
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{task_temp_voxel_chunks_buffer},
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{task_voxel_chunks_buffer},
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{gpu_resources.voxel_malloc.task_element_buffer},
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{gpu_resources.voxel_leaf_chunk_malloc.task_element_buffer},
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{gpu_resources.voxel_parent_chunk_malloc.task_element_buffer},
        },
        .task = [this](daxa::TaskInterface task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            cmd_list.clear_buffer({
                .buffer = task_globals_buffer.get_state().buffers[0],
                .offset = 0,
                .size = sizeof(GpuGlobals),
                .clear_value = 0,
            });
            cmd_list.clear_buffer({
                .buffer = task_temp_voxel_chunks_buffer.get_state().buffers[0],
                .offset = 0,
                .size = sizeof(TempVoxelChunk) * MAX_CHUNK_UPDATES_PER_FRAME,
                .clear_value = 0,
            });
            auto chunk_n = (1u << ui.settings.log2_chunks_per_axis);
            chunk_n = chunk_n * chunk_n * chunk_n;
            cmd_list.clear_buffer({
                .buffer = task_voxel_chunks_buffer.get_state().buffers[0],
                .offset = 0,
                .size = sizeof(VoxelLeafChunk) * chunk_n,
                .clear_value = 0,
            });
            gpu_resources.voxel_malloc.clear_buffers(cmd_list);
            gpu_resources.voxel_leaf_chunk_malloc.clear_buffers(cmd_list);
            gpu_resources.voxel_parent_chunk_malloc.clear_buffers(cmd_list);
        },
        .name = "StartupTask (Globals Clear)",
    });
    temp_task_graph.add_task(StartupComputeTask{
        {
            .uses = {
                .gpu_input = task_input_buffer,
                .globals = task_globals_buffer,
                .voxel_chunks = task_voxel_chunks_buffer,
            },
        },
        &startup_task_state,
    });

    temp_task_graph.add_task({
        .uses = {
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{gpu_resources.voxel_malloc.task_allocator_buffer},
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{gpu_resources.voxel_leaf_chunk_malloc.task_allocator_buffer},
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{gpu_resources.voxel_parent_chunk_malloc.task_allocator_buffer},
        },
        .task = [this](daxa::TaskInterface task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            gpu_resources.voxel_malloc.init(device, cmd_list);
            gpu_resources.voxel_leaf_chunk_malloc.init(device, cmd_list);
            gpu_resources.voxel_parent_chunk_malloc.init(device, cmd_list);
            ui.should_run_startup = false;
        },
        .name = "Initialize",
    });
    temp_task_graph.submit({});
    temp_task_graph.complete({});
    temp_task_graph.execute({});
}

void VoxelApp::upload_model(daxa::TaskGraph &) {
    auto temp_task_graph = daxa::TaskGraph({
        .device = device,
        .name = "temp_task_graph",
    });
    temp_task_graph.use_persistent_buffer(task_gvox_model_buffer);
    temp_task_graph.add_task({
        .uses = {
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{task_gvox_model_buffer},
        },
        .task = [this](daxa::TaskInterface task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            if (!prev_gvox_model_buffer.is_empty()) {
                cmd_list.destroy_buffer_deferred(prev_gvox_model_buffer);
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
                .dst_buffer = gpu_resources.gvox_model_buffer,
                .size = static_cast<u32>(gvox_model_data.size),
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

void VoxelApp::dynamic_buffers_realloc(daxa::TaskGraph &) {
    auto temp_task_graph = daxa::TaskGraph({
        .device = device,
        .name = "temp_task_graph",
    });
    gpu_resources.voxel_malloc.for_each_task_buffer([&temp_task_graph](auto &task_buffer) { temp_task_graph.use_persistent_buffer(task_buffer); });
    gpu_resources.voxel_leaf_chunk_malloc.for_each_task_buffer([&temp_task_graph](auto &task_buffer) { temp_task_graph.use_persistent_buffer(task_buffer); });
    gpu_resources.voxel_parent_chunk_malloc.for_each_task_buffer([&temp_task_graph](auto &task_buffer) { temp_task_graph.use_persistent_buffer(task_buffer); });
    temp_task_graph.add_task({
        .uses = {
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_READ>{gpu_resources.voxel_malloc.task_old_element_buffer},
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{gpu_resources.voxel_malloc.task_element_buffer},
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_READ>{gpu_resources.voxel_leaf_chunk_malloc.task_old_element_buffer},
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{gpu_resources.voxel_leaf_chunk_malloc.task_element_buffer},
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_READ>{gpu_resources.voxel_parent_chunk_malloc.task_old_element_buffer},
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{gpu_resources.voxel_parent_chunk_malloc.task_element_buffer},
        },
        .task = [&](daxa::TaskInterface task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            if (gpu_resources.voxel_malloc.needs_realloc()) {
                gpu_resources.voxel_malloc.realloc(device, cmd_list);
                needs_vram_calc = true;
            }
            if (gpu_resources.voxel_leaf_chunk_malloc.needs_realloc()) {
                gpu_resources.voxel_leaf_chunk_malloc.realloc(device, cmd_list);
                needs_vram_calc = true;
            }
            if (gpu_resources.voxel_parent_chunk_malloc.needs_realloc()) {
                gpu_resources.voxel_parent_chunk_malloc.realloc(device, cmd_list);
                needs_vram_calc = true;
            }
        },
        .name = "Transfer Task",
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

// ChunkHierarchy (chunk_hierarchy.comp.glsl) (x2)
// -> Creates hierarchical structure / chunk work items to be generated and processed by Chunk Edit

// ChunkEdit (chunk_edit.comp.glsl)
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
    gpu_input.frame_dim.x = static_cast<u32>(static_cast<f32>(window_size.x) * render_res_scl);
    gpu_input.frame_dim.y = static_cast<u32>(static_cast<f32>(window_size.y) * render_res_scl);
    gpu_input.rounded_frame_dim = round_frame_dim(gpu_input.frame_dim);

    daxa::TaskGraph result_task_graph = daxa::TaskGraph({
        .device = device,
        .swapchain = swapchain,
        .permutation_condition_count = static_cast<usize>(Conditions::COUNT),
        .name = "main_task_graph",
    });

    result_task_graph.use_persistent_image(task_value_noise_image);
    result_task_graph.use_persistent_image(task_blue_noise_vec2_image);
    task_value_noise_image.set_images({.images = std::array{gpu_resources.value_noise_image}});
    task_blue_noise_vec2_image.set_images({.images = std::array{gpu_resources.blue_noise_vec2_image}});

    result_task_graph.use_persistent_buffer(task_input_buffer);
    result_task_graph.use_persistent_buffer(task_output_buffer);
    result_task_graph.use_persistent_buffer(task_staging_output_buffer);
    result_task_graph.use_persistent_buffer(task_globals_buffer);
    result_task_graph.use_persistent_buffer(task_temp_voxel_chunks_buffer);
    result_task_graph.use_persistent_buffer(task_voxel_chunks_buffer);
    result_task_graph.use_persistent_buffer(task_gvox_model_buffer);
    gpu_resources.voxel_malloc.for_each_task_buffer([&result_task_graph](auto &task_buffer) { result_task_graph.use_persistent_buffer(task_buffer); });
    gpu_resources.voxel_leaf_chunk_malloc.for_each_task_buffer([&result_task_graph](auto &task_buffer) { result_task_graph.use_persistent_buffer(task_buffer); });
    gpu_resources.voxel_parent_chunk_malloc.for_each_task_buffer([&result_task_graph](auto &task_buffer) { result_task_graph.use_persistent_buffer(task_buffer); });

    task_input_buffer.set_buffers({.buffers = std::array{gpu_resources.input_buffer}});
    task_output_buffer.set_buffers({.buffers = std::array{gpu_resources.output_buffer}});
    task_staging_output_buffer.set_buffers({.buffers = std::array{gpu_resources.staging_output_buffer}});
    task_globals_buffer.set_buffers({.buffers = std::array{gpu_resources.globals_buffer}});
    task_temp_voxel_chunks_buffer.set_buffers({.buffers = std::array{gpu_resources.temp_voxel_chunks_buffer}});
    task_voxel_chunks_buffer.set_buffers({.buffers = std::array{gpu_resources.voxel_chunks.buffer}});
    task_gvox_model_buffer.set_buffers({.buffers = std::array{gpu_resources.gvox_model_buffer}});

    result_task_graph.use_persistent_buffer(task_simulated_voxel_particles_buffer);
    result_task_graph.use_persistent_buffer(task_rendered_voxel_particles_buffer);
    result_task_graph.use_persistent_buffer(task_placed_voxel_particles_buffer);
    task_simulated_voxel_particles_buffer.set_buffers({.buffers = std::array{gpu_resources.simulated_voxel_particles_buffer}});
    task_rendered_voxel_particles_buffer.set_buffers({.buffers = std::array{gpu_resources.rendered_voxel_particles_buffer}});
    task_placed_voxel_particles_buffer.set_buffers({.buffers = std::array{gpu_resources.placed_voxel_particles_buffer}});

    result_task_graph.use_persistent_image(task_swapchain_image);
    task_swapchain_image.set_images({.images = std::array{swapchain_image}});

    result_task_graph.add_task({
        .uses = {
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{task_input_buffer},
        },
        .task = [this](daxa::TaskInterface task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            auto staging_input_buffer = device.create_buffer({
                .size = sizeof(GpuInput),
                .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = "staging_input_buffer",
            });
            cmd_list.destroy_buffer_deferred(staging_input_buffer);
            auto *buffer_ptr = device.get_host_address_as<GpuInput>(staging_input_buffer);
            *buffer_ptr = gpu_input;
            cmd_list.copy_buffer_to_buffer({
                .src_buffer = staging_input_buffer,
                .dst_buffer = task_input_buffer.get_state().buffers[0],
                .size = sizeof(GpuInput),
            });
        },
        .name = "GpuInputUploadTransferTask",
    });

    result_task_graph.add_task(PerframeComputeTask{
        {
            .uses = {
                .gpu_input = task_input_buffer,
                .gpu_output = task_output_buffer,
                .globals = task_globals_buffer,
                .simulated_voxel_particles = task_simulated_voxel_particles_buffer,
                .voxel_malloc_page_allocator = gpu_resources.voxel_malloc.task_allocator_buffer,
                .voxel_leaf_chunk_allocator = gpu_resources.voxel_leaf_chunk_malloc.task_allocator_buffer,
                .voxel_parent_chunk_allocator = gpu_resources.voxel_parent_chunk_malloc.task_allocator_buffer,
                .voxel_chunks = task_voxel_chunks_buffer,
            },
        },
        &perframe_task_state,
    });

#if MAX_RENDERED_VOXEL_PARTICLES > 0
    result_task_graph.add_task(VoxelParticleSimComputeTask{
        {
            .uses = {
                .settings = task_settings_buffer,
                .gpu_input = task_input_buffer,
                .globals = task_globals_buffer,
                .voxel_malloc_page_allocator = gpu_resources.voxel_malloc.task_allocator_buffer,
                .voxel_chunks = task_voxel_chunks_buffer,
                .simulated_voxel_particles = task_simulated_voxel_particles_buffer,
                .rendered_voxel_particles = task_rendered_voxel_particles_buffer,
                .placed_voxel_particles = task_placed_voxel_particles_buffer,
            },
        },
        &voxel_particle_sim_task_state,
    });
#endif

    result_task_graph.add_task(PerChunkComputeTask{
        {
            .uses = {
                .gpu_input = task_input_buffer,
                .gvox_model = task_gvox_model_buffer,
                .globals = task_globals_buffer,
                .voxel_chunks = task_voxel_chunks_buffer,
                .value_noise_texture = task_value_noise_image.view().view({.layer_count = 256}),
            },
        },
        &per_chunk_task_state,
        &gpu_resources.value_noise_sampler,
    });

    result_task_graph.add_task(ChunkEditComputeTask{
        {
            .uses = {
                .gpu_input = task_input_buffer,
                .globals = task_globals_buffer,
                .gvox_model = task_gvox_model_buffer,
                .voxel_chunks = task_voxel_chunks_buffer,
                .temp_voxel_chunks = task_temp_voxel_chunks_buffer,
                .voxel_malloc_page_allocator = gpu_resources.voxel_malloc.task_allocator_buffer,
                .simulated_voxel_particles = task_simulated_voxel_particles_buffer,
                .placed_voxel_particles = task_placed_voxel_particles_buffer,
                .value_noise_texture = task_value_noise_image.view().view({.layer_count = 256}),
            },
        },
        &chunk_edit_task_state,
        &gpu_resources.value_noise_sampler,
    });

    result_task_graph.add_task(ChunkOpt_x2x4_ComputeTask{
        {
            .uses = {
                .gpu_input = task_input_buffer,
                .globals = task_globals_buffer,
                .temp_voxel_chunks = task_temp_voxel_chunks_buffer,
                .voxel_chunks = task_voxel_chunks_buffer,
            },
        },
        &chunk_opt_x2x4_task_state,
    });

    result_task_graph.add_task(ChunkOpt_x8up_ComputeTask{
        {
            .uses = {
                .gpu_input = task_input_buffer,
                .globals = task_globals_buffer,
                .temp_voxel_chunks = task_temp_voxel_chunks_buffer,
                .voxel_chunks = task_voxel_chunks_buffer,
            },
        },
        &chunk_opt_x8up_task_state,
    });

    result_task_graph.add_task(ChunkAllocComputeTask{
        {
            .uses = {
                .gpu_input = task_input_buffer,
                .globals = task_globals_buffer,
                .temp_voxel_chunks = task_temp_voxel_chunks_buffer,
                .voxel_chunks = task_voxel_chunks_buffer,
                .voxel_malloc_page_allocator = gpu_resources.voxel_malloc.task_allocator_buffer,
            },
        },
        &chunk_alloc_task_state,
    });

#if MAX_RENDERED_VOXEL_PARTICLES > 0
    result_task_graph.add_task(VoxelParticleRasterTask{
        {
            .uses = {
                .gpu_input = task_input_buffer,
                .globals = task_globals_buffer,
                .simulated_voxel_particles = task_simulated_voxel_particles_buffer,
                .rendered_voxel_particles = task_rendered_voxel_particles_buffer,
                .render_image = task_render_raster_color_image,
                .depth_image_id = task_render_raster_depth_image,
            },
        },
        &voxel_particle_raster_task_state,
    });
#endif

    auto record_ctx = RecordContext{
        .device = this->device,
        .task_graph = result_task_graph,
        .render_resolution = gpu_input.rounded_frame_dim,
        .task_blue_noise_vec2_image = task_blue_noise_vec2_image,
        .task_input_buffer = task_input_buffer,
        .task_globals_buffer = task_globals_buffer,
    };

    auto [gbuffer_depth, velocity_image] = gbuffer_renderer.render(record_ctx, gpu_resources.voxel_malloc.task_allocator_buffer, task_voxel_chunks_buffer);
    auto reprojection_map = reprojection_renderer.calculate_reprojection_map(record_ctx, gbuffer_depth, velocity_image);
    auto ssao_image = ssao_renderer.render(record_ctx, gbuffer_depth, reprojection_map);
    auto shading_image = shadow_renderer.render(record_ctx, gbuffer_depth, reprojection_map, gpu_resources.voxel_malloc.task_allocator_buffer, task_voxel_chunks_buffer);

    // auto composited_image = [](RecordContext &record_ctx) {
    //     record_ctx.task_graph.add_task(PostprocessingRasterTask{
    //         {
    //             .uses = {
    //                 .gpu_input = task_input_buffer,
    //                 .g_buffer_image_id = gbuffer_depth.gbuffer,
    //                 .ssao_image_id = ssao_image,
    //                 .shading_image_id = shading_image,
    //                 .render_image = task_swapchain_image,
    //             },
    //         },
    //         &postprocessing_task_state,
    //     });
    // }(record_ctx);

    result_task_graph.add_task(PostprocessingRasterTask{
        {
            .uses = {
                .gpu_input = task_input_buffer,
                .g_buffer_image_id = gbuffer_depth.gbuffer,
                .ssao_image_id = ssao_image,
                .shading_image_id = shading_image,
                .render_image = task_swapchain_image,
            },
        },
        &postprocessing_task_state,
    });

    result_task_graph.add_task({
        .uses = {
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_READ>{task_output_buffer},
            daxa::TaskBufferUse<daxa::TaskBufferAccess::HOST_TRANSFER_WRITE>{task_staging_output_buffer},
        },
        .task = [this](daxa::TaskInterface task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            auto output_buffer = task_output_buffer.get_state().buffers[0];
            auto staging_output_buffer = gpu_resources.staging_output_buffer;
            auto frame_index = gpu_input.frame_index + 1;
            auto *buffer_ptr = device.get_host_address_as<std::array<GpuOutput, (FRAMES_IN_FLIGHT + 1)>>(staging_output_buffer);
            u32 const offset = frame_index % (FRAMES_IN_FLIGHT + 1);
            gpu_output = (*buffer_ptr)[offset];
            cmd_list.copy_buffer_to_buffer({
                .src_buffer = output_buffer,
                .dst_buffer = staging_output_buffer,
                .size = sizeof(GpuOutput) * (FRAMES_IN_FLIGHT + 1),
            });
        },
        .name = "GpuOutputDownloadTransferTask",
    });

    result_task_graph.add_task({
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
