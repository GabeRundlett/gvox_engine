#include "voxel_app.hpp"

#include <thread>
#include <numbers>
#include <fstream>

#include <gvox/adapters/input/byte_buffer.h>
#include <gvox/adapters/output/byte_buffer.h>
#include <gvox/adapters/parse/voxlap.h>

#define APPNAME "Voxel App"

using namespace std::chrono_literals;

#include <iostream>

#include <minizip/unzip.h>

void RenderImages::create(daxa::Device &device) {
    pos_images[0] = device.create_image({
        .format = daxa::Format::R32G32B32A32_SFLOAT,
        .size = {size.x, size.y, 1},
        .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::SHADER_READ_ONLY,
        .name = "pos_image",
    });
    pos_images[1] = device.create_image({
        .format = daxa::Format::R32G32B32A32_SFLOAT,
        .size = {size.x, size.y, 1},
        .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::SHADER_READ_ONLY,
        .name = "pos_image",
    });
    col_images[0] = device.create_image({
        .format = daxa::Format::R32G32B32A32_SFLOAT,
        .size = {size.x, size.y, 1},
        .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::SHADER_READ_ONLY,
        .name = "col_image",
    });
    col_images[1] = device.create_image({
        .format = daxa::Format::R32G32B32A32_SFLOAT,
        .size = {size.x, size.y, 1},
        .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::SHADER_READ_ONLY,
        .name = "col_image",
    });
    final_image = device.create_image({
        .format = daxa::Format::R32G32B32A32_SFLOAT,
        .size = {size.x, size.y, 1},
        .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::SHADER_READ_ONLY | daxa::ImageUsageFlagBits::TRANSFER_SRC,
        .name = "final_image",
    });

    raster_color_image = device.create_image({
        .format = daxa::Format::R32G32B32A32_SFLOAT,
        .size = {size.x, size.y, 1},
        .usage = daxa::ImageUsageFlagBits::COLOR_ATTACHMENT | daxa::ImageUsageFlagBits::SHADER_READ_ONLY,
        .name = "raster_color_image",
    });
    raster_depth_image = device.create_image({
        .format = daxa::Format::D32_SFLOAT,
        .size = {size.x, size.y, 1},
        .usage = daxa::ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT | daxa::ImageUsageFlagBits::SHADER_READ_ONLY,
        .name = "raster_depth_image",
    });
}
void RenderImages::destroy(daxa::Device &device) const {
    device.destroy_image(pos_images[0]);
    device.destroy_image(pos_images[1]);
    device.destroy_image(col_images[0]);
    device.destroy_image(col_images[1]);
    device.destroy_image(final_image);
    device.destroy_image(raster_color_image);
    device.destroy_image(raster_depth_image);
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

void VoxelMalloc::create(daxa::Device &device, u32 page_count) {
    current_page_count = page_count;
    global_allocator_buffer = device.create_buffer({
        .size = sizeof(VoxelMalloc_GlobalAllocator),
        .name = "voxel_malloc_global_allocator_buffer",
    });
    pages_buffer = device.create_buffer({
        .size = VOXEL_MALLOC_PAGE_SIZE_BYTES * current_page_count,
        .name = "voxel_malloc_pages_buffer",
    });
    available_pages_stack_buffer = device.create_buffer({
        .size = static_cast<u32>(sizeof(VoxelMalloc_PageIndex)) * current_page_count,
        .name = "available_pages_stack_buffer",
    });
    released_pages_stack_buffer = device.create_buffer({
        .size = static_cast<u32>(sizeof(VoxelMalloc_PageIndex)) * current_page_count,
        .name = "released_pages_stack_buffer",
    });
}
void VoxelMalloc::destroy(daxa::Device &device) const {
    if (!pages_buffer.is_empty()) {
        device.destroy_buffer(pages_buffer);
    }
    if (!available_pages_stack_buffer.is_empty()) {
        device.destroy_buffer(available_pages_stack_buffer);
    }
    if (!released_pages_stack_buffer.is_empty()) {
        device.destroy_buffer(released_pages_stack_buffer);
    }
    device.destroy_buffer(global_allocator_buffer);
}

#if USE_OLD_ALLOC
void GpuHeap::create(daxa::Device &device, u32 size) {
    buffer = device.create_buffer({
        .size = size,
        .name = "gpu_heap_buffer",
    });
}
void GpuHeap::destroy(daxa::Device &device) const {
    if (!buffer.is_empty()) {
        device.destroy_buffer(buffer);
    }
}
#endif

void GpuResources::create(daxa::Device &device) {
    render_images.create(device);
    blue_noise_vec1_image = device.create_image({
        .dimensions = 3,
        .format = daxa::Format::R8G8B8A8_UNORM,
        .size = {128, 128, 64},
        .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::SHADER_READ_ONLY,
        .name = "blue_noise_vec1_image",
    });
    blue_noise_vec2_image = device.create_image({
        .dimensions = 3,
        .format = daxa::Format::R8G8B8A8_UNORM,
        .size = {128, 128, 64},
        .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::SHADER_READ_ONLY,
        .name = "blue_noise_vec2_image",
    });
    blue_noise_unit_vec3_image = device.create_image({
        .dimensions = 3,
        .format = daxa::Format::R8G8B8A8_UNORM,
        .size = {128, 128, 64},
        .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::SHADER_READ_ONLY,
        .name = "blue_noise_unit_vec3_image",
    });
    blue_noise_cosine_vec3_image = device.create_image({
        .dimensions = 3,
        .format = daxa::Format::R8G8B8A8_UNORM,
        .size = {128, 128, 64},
        .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::SHADER_READ_ONLY,
        .name = "blue_noise_cosine_vec3_image",
    });
    settings_buffer = device.create_buffer({
        .size = sizeof(GpuSettings),
        .name = "settings_buffer",
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
#if 0
    test_data_buffer = device.create_buffer({
        .size = test_data_size,
        .name = "test_data_buffer",
    });
#endif
    temp_voxel_chunks_buffer = device.create_buffer({
        .size = sizeof(TempVoxelChunk) * MAX_CHUNK_UPDATES_PER_FRAME,
        .name = "temp_voxel_chunks_buffer",
    });
    gvox_model_buffer = device.create_buffer({
        .size = offsetof(GpuGvoxModel, data),
        .name = "gvox_model_buffer",
    });
    simulated_voxel_particles_buffer = device.create_buffer({
        .size = sizeof(SimulatedVoxelParticle) * MAX_SIMULATED_VOXEL_PARTICLES,
        .name = "simulated_voxel_particles_buffer",
    });
    rendered_voxel_particles_buffer = device.create_buffer({
        .size = sizeof(u32) * MAX_RENDERED_VOXEL_PARTICLES,
        .name = "rendered_voxel_particles_buffer",
    });
}
void GpuResources::destroy(daxa::Device &device) const {
    render_images.destroy(device);
    device.destroy_image(blue_noise_vec1_image);
    device.destroy_image(blue_noise_vec2_image);
    device.destroy_image(blue_noise_unit_vec3_image);
    device.destroy_image(blue_noise_cosine_vec3_image);
    device.destroy_buffer(settings_buffer);
    device.destroy_buffer(input_buffer);
    device.destroy_buffer(output_buffer);
    device.destroy_buffer(staging_output_buffer);
    device.destroy_buffer(globals_buffer);
#if 0
    device.destroy_buffer(test_data_buffer);
#endif
    device.destroy_buffer(temp_voxel_chunks_buffer);
    voxel_chunks.destroy(device);
    voxel_malloc.destroy(device);
#if USE_OLD_ALLOC
    gpu_heap.destroy(device);
#endif
    if (!gvox_model_buffer.is_empty()) {
        device.destroy_buffer(gvox_model_buffer);
    }
    device.destroy_buffer(simulated_voxel_particles_buffer);
    device.destroy_buffer(rendered_voxel_particles_buffer);
}

VoxelApp::VoxelApp()
    : AppWindow(APPNAME, {1400, 1200}),
      daxa_ctx{daxa::create_context({.enable_validation = false})},
      device{daxa_ctx.create_device({
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
      main_pipeline_manager{daxa::PipelineManager({
          .device = device,
          .shader_compile_options = {
              .root_paths = {
                  DAXA_SHADER_INCLUDE_DIR,
                  "assets",
                  "src",
                  "gpu",
                  "src/gpu",
              },
              .language = daxa::ShaderLanguage::GLSL,
              .enable_debug_info = true,
          },
          .name = "pipeline_manager",
      })},
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
      gpu_resources{
          .render_images{.size{window_size}},
      },
      // clang-format off
      startup_task_state{main_pipeline_manager, ui},
      perframe_task_state{main_pipeline_manager, ui},
      chunk_edit_task_state{main_pipeline_manager, ui},
      chunk_opt_x2x4_task_state{main_pipeline_manager, ui},
      chunk_opt_x8up_task_state{main_pipeline_manager, ui},
      chunk_alloc_task_state{main_pipeline_manager, ui},
      trace_primary_task_state{main_pipeline_manager, ui, gpu_resources.render_images.size},
      color_scene_task_state{main_pipeline_manager, ui, gpu_resources.render_images.size},
      postprocessing_task_state{main_pipeline_manager, ui, gpu_resources.render_images.size},
      chunk_hierarchy_task_state{main_pipeline_manager, ui},
      voxel_particle_sim_task_state{main_pipeline_manager, ui},
      voxel_particle_raster_task_state{main_pipeline_manager, ui},
      // clang-format on
      main_task_list{[this]() {
          gpu_resources.create(device);
          gpu_resources.voxel_chunks.create(device, ui.settings.log2_chunks_per_axis);
#if USE_OLD_ALLOC
          gpu_resources.gpu_heap.create(device, ui.settings.gpu_heap_size);
#endif

          // Full size
          // u32 pages = 1 << ui.settings.log2_chunks_per_axis;
          // pages = pages * pages * pages;
          // pages = VOXEL_MALLOC_MAX_ALLOCATIONS_PER_CHUNK * pages;

          // Min size
          u32 pages = (FRAMES_IN_FLIGHT + 1) * VOXEL_MALLOC_MAX_PAGE_ALLOCATIONS_PER_FRAME;

          // 1GB
          // u32 pages = (1 << 30) / VOXEL_MALLOC_PAGE_SIZE_BYTES;

          gpu_resources.voxel_malloc.create(device, pages);
          return record_main_task_list();
      }()} {
    gvox_ctx = gvox_create_context();
    start = Clock::now();

#if 0
    {
        daxa::TaskList temp_task_list = daxa::TaskList({
            .device = device,
            .name = "temp_task_list",
        });
        temp_task_list.use_persistent_buffer(task_test_data_buffer);
        temp_task_list.add_task({
            .uses = {
                daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{task_test_data_buffer},
            },
            .task = [this](daxa::TaskInterface task_runtime) {
                auto file_handle = std::ifstream{"assets/mag1.bin", std::ios::binary};
                auto cmd_list = task_runtime.get_command_list();
                cmd_list.pipeline_barrier({
                    .waiting_pipeline_access = daxa::AccessConsts::TRANSFER_WRITE,
                });
                auto staging_buffer = device.create_buffer({
                    .size = static_cast<u32>(test_data_size),
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "staging_buffer",
                });
                cmd_list.destroy_buffer_deferred(staging_buffer);
                char *buffer_ptr = device.get_host_address_as<char>(staging_buffer);
                file_handle.read(reinterpret_cast<char *>(buffer_ptr), static_cast<std::streamsize>(test_data_size));
                cmd_list.copy_buffer_to_buffer({
                    .src_buffer = staging_buffer,
                    .dst_buffer = gpu_resources.test_data_buffer,
                    .size = static_cast<u32>(test_data_size),
                });
                needs_vram_calc = true;
            },
            .name = "upload_model",
        });
        temp_task_list.submit({});
        temp_task_list.complete({});
        temp_task_list.execute({});
    }
#endif

    {
        daxa::TaskList temp_task_list = daxa::TaskList({
            .device = device,
            .name = "temp_task_list",
        });
        temp_task_list.use_persistent_image(task_blue_noise_vec1_image);
        temp_task_list.use_persistent_image(task_blue_noise_vec2_image);
        temp_task_list.use_persistent_image(task_blue_noise_unit_vec3_image);
        temp_task_list.use_persistent_image(task_blue_noise_cosine_vec3_image);
        temp_task_list.add_task({
            .uses = {
                daxa::TaskImageUse<daxa::TaskImageAccess::TRANSFER_WRITE>{task_blue_noise_vec1_image},
                daxa::TaskImageUse<daxa::TaskImageAccess::TRANSFER_WRITE>{task_blue_noise_vec2_image},
                daxa::TaskImageUse<daxa::TaskImageAccess::TRANSFER_WRITE>{task_blue_noise_unit_vec3_image},
                daxa::TaskImageUse<daxa::TaskImageAccess::TRANSFER_WRITE>{task_blue_noise_cosine_vec3_image},
            },
            .task = [this](daxa::TaskInterface task_runtime) {
                auto staging_buffer = device.create_buffer({
                    .size = static_cast<u32>(128 * 128 * 4 * 64 * 4),
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "staging_buffer",
                });
                auto buffer_ptr = device.get_host_address_as<u8>(staging_buffer);
                auto *stbn_zip = unzOpen("assets/STBN.zip");
                for (auto i = 0; i < 64; ++i) {
                    [[maybe_unused]] int err = 0;
                    i32 size_x = 0, size_y = 0, channel_n = 0;
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
                        auto temp_data = stbi_load_from_memory(file_data.data(), static_cast<int>(file_data.size()), &size_x, &size_y, &channel_n, 4);
                        if (temp_data) {
                            assert(size_x == 128 && size_y == 128);
                            std::copy(temp_data + 0, temp_data + 128 * 128 * 4, buffer_out_ptr);
                        }
                    };
                    auto vec1_name = std::string{"STBN/stbn_vec1_2Dx1D_128x128x64_"} + std::to_string(i) + ".png";
                    auto vec2_name = std::string{"STBN/stbn_vec2_2Dx1D_128x128x64_"} + std::to_string(i) + ".png";
                    auto vec3_name = std::string{"STBN/stbn_unitvec3_2Dx1D_128x128x64_"} + std::to_string(i) + ".png";
                    auto cosine_vec3_name = std::string{"STBN/stbn_unitvec3_cosine_2Dx1D_128x128x64_"} + std::to_string(i) + ".png";
                    load_image(vec1_name.c_str(), buffer_ptr + (128 * 128 * 4) * i + (128 * 128 * 4 * 64) * 0);
                    load_image(vec2_name.c_str(), buffer_ptr + (128 * 128 * 4) * i + (128 * 128 * 4 * 64) * 1);
                    load_image(vec3_name.c_str(), buffer_ptr + (128 * 128 * 4) * i + (128 * 128 * 4 * 64) * 2);
                    load_image(cosine_vec3_name.c_str(), buffer_ptr + (128 * 128 * 4) * i + (128 * 128 * 4 * 64) * 3);
                }

                auto cmd_list = task_runtime.get_command_list();
                cmd_list.pipeline_barrier({
                    .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
                });
                cmd_list.destroy_buffer_deferred(staging_buffer);
                cmd_list.copy_buffer_to_image({
                    .buffer = staging_buffer,
                    .buffer_offset = (128 * 128 * 4 * 64) * 0,
                    .image = task_blue_noise_vec1_image.get_state().images[0],
                    .image_extent = {128, 128, 64},
                });
                cmd_list.copy_buffer_to_image({
                    .buffer = staging_buffer,
                    .buffer_offset = (128 * 128 * 4 * 64) * 1,
                    .image = task_blue_noise_vec2_image.get_state().images[0],
                    .image_extent = {128, 128, 64},
                });
                cmd_list.copy_buffer_to_image({
                    .buffer = staging_buffer,
                    .buffer_offset = (128 * 128 * 4 * 64) * 2,
                    .image = task_blue_noise_unit_vec3_image.get_state().images[0],
                    .image_extent = {128, 128, 64},
                });
                cmd_list.copy_buffer_to_image({
                    .buffer = staging_buffer,
                    .buffer_offset = (128 * 128 * 4 * 64) * 3,
                    .image = task_blue_noise_cosine_vec3_image.get_state().images[0],
                    .image_extent = {128, 128, 64},
                });
                needs_vram_calc = true;
            },
            .name = "upload_blue_noise",
        });
        temp_task_list.submit({});
        temp_task_list.complete({});
        temp_task_list.execute({});
    }
}
VoxelApp::~VoxelApp() {
    gvox_destroy_context(gvox_ctx);
    device.wait_idle();
    device.collect_garbage();
    gpu_resources.destroy(device);
}

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
        if (image.is_empty())
            return;
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
        if (buffer.is_empty())
            return;
        auto buffer_info = device.info_buffer(buffer);
        ui.debug_gpu_resource_infos.push_back({
            .type = "buffer",
            .name = buffer_info.name,
            .size = buffer_info.size,
        });
        result_size += buffer_info.size;
    };

    image_size(gpu_resources.render_images.pos_images[0]);
    image_size(gpu_resources.render_images.pos_images[1]);
    image_size(gpu_resources.render_images.col_images[0]);
    image_size(gpu_resources.render_images.col_images[1]);
    image_size(gpu_resources.render_images.final_image);

    buffer_size(gpu_resources.settings_buffer);
    buffer_size(gpu_resources.input_buffer);
    buffer_size(gpu_resources.globals_buffer);
#if 0
    buffer_size(gpu_resources.test_data_buffer);
#endif
    buffer_size(gpu_resources.temp_voxel_chunks_buffer);
    buffer_size(gpu_resources.voxel_malloc.global_allocator_buffer);
    buffer_size(gpu_resources.voxel_chunks.buffer);
    buffer_size(gpu_resources.voxel_malloc.pages_buffer);
    buffer_size(gpu_resources.voxel_malloc.available_pages_stack_buffer);
    buffer_size(gpu_resources.voxel_malloc.released_pages_stack_buffer);
#if USE_OLD_ALLOC
    buffer_size(gpu_resources.gpu_heap.buffer);
#endif
    buffer_size(gpu_resources.gvox_model_buffer);
    buffer_size(gpu_resources.simulated_voxel_particles_buffer);
    buffer_size(gpu_resources.rendered_voxel_particles_buffer);

    needs_vram_calc = false;
}
auto VoxelApp::load_gvox_data() -> GvoxModelData {
    auto result = GvoxModelData{};
    auto file = std::ifstream(ui.gvox_model_path, std::ios::binary);
    if (!file.is_open()) {
        ui.console.add_log("[error] Failed to load the model");
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
        // ui.console.add_log("(pulling file into memory: {}s)", cpu_time_used);
    }
    GvoxByteBufferInputAdapterConfig i_config = {
        .data = temp_gvox_model.data(),
        .size = temp_gvox_model_size,
    };
    GvoxByteBufferOutputAdapterConfig o_config = {
        .out_size = &result.size,
        .out_byte_buffer_ptr = &result.ptr,
        .allocate = NULL,
    };
    void *i_config_ptr = nullptr;
    auto voxlap_config = GvoxVoxlapParseAdapterConfig{
        .size_x = 512,
        .size_y = 512,
        .size_z = 64,
        .make_solid = true,
        .is_ace_of_spades = true,
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
    GvoxAdapterContext *s_ctx = gvox_create_adapter_context(gvox_ctx, gvox_get_serialize_adapter(gvox_ctx, "gvox_palette"), NULL);

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
        // ui.console.add_log("{}s, new size: {} bytes", cpu_time_used, result.size);

        GvoxResult res = gvox_get_result(gvox_ctx);
        // int error_count = 0;
        while (res != GVOX_RESULT_SUCCESS) {
            size_t size;
            gvox_get_result_message(gvox_ctx, NULL, &size);
            char *str = new char[size + 1];
            gvox_get_result_message(gvox_ctx, str, NULL);
            str[size] = '\0';
            ui.console.add_log("ERROR loading model: {}", str);
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

void VoxelApp::on_update() {
    auto now = Clock::now();
    gpu_input.time = std::chrono::duration<f32>(now - start).count();
    gpu_input.delta_time = std::chrono::duration<f32>(now - prev_time).count();
    prev_time = now;
    gpu_input.frame_dim = gpu_resources.render_images.size;

    {
        auto reload_result = main_pipeline_manager.reload_all();
        if (reload_result.has_value()) {
            if (reload_result.value().is_err()) {
                ui.console.add_log(reload_result.value().to_string());
            }
        }
    }

    ui.update(gpu_input.delta_time);

    swapchain_image = swapchain.acquire_next_image();
    task_swapchain_image.set_images({.images = {&swapchain_image, 1}});
    if (swapchain_image.is_empty()) {
        return;
    }

    if (ui.should_recreate_voxel_buffers) {
        recreate_voxel_chunks();
    }

    bool model_is_ready = false;
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

    auto const max_size_after_cpu_catch_up = static_cast<size_t>(gpu_output.heap_size) * sizeof(u32) + VOXEL_MALLOC_MAX_PAGE_ALLOCATIONS_PER_FRAME * (FRAMES_IN_FLIGHT + 1) * VOXEL_MALLOC_PAGE_SIZE_BYTES;
    auto const current_size = static_cast<size_t>(gpu_resources.voxel_malloc.current_page_count) * VOXEL_MALLOC_PAGE_SIZE_BYTES;
    gpu_resources.voxel_malloc.next_page_count = 0;
    if (max_size_after_cpu_catch_up > current_size) {
        gpu_resources.voxel_malloc.next_page_count = gpu_resources.voxel_malloc.current_page_count + VOXEL_MALLOC_MAX_PAGE_ALLOCATIONS_PER_FRAME * (FRAMES_IN_FLIGHT + 1);

        auto &current_page_count = gpu_resources.voxel_malloc.current_page_count;
        auto const &next_page_count = gpu_resources.voxel_malloc.next_page_count;

        assert(next_page_count > current_page_count);

        prev_page_count = current_page_count;
        current_page_count = next_page_count * 2;

        auto new_pages_buffer = device.create_buffer({
            .size = VOXEL_MALLOC_PAGE_SIZE_BYTES * current_page_count,
            .name = "voxel_malloc_pages_buffer",
        });
        auto new_available_pages_stack_buffer = device.create_buffer({
            .size = static_cast<u32>(sizeof(VoxelMalloc_PageIndex)) * current_page_count,
            .name = "available_pages_stack_buffer",
        });
        auto new_released_pages_stack_buffer = device.create_buffer({
            .size = static_cast<u32>(sizeof(VoxelMalloc_PageIndex)) * current_page_count,
            .name = "released_pages_stack_buffer",
        });
        task_voxel_malloc_old_pages_buffer.swap_buffers(task_voxel_malloc_pages_buffer);

        gpu_resources.voxel_malloc.pages_buffer = new_pages_buffer;
        gpu_resources.voxel_malloc.available_pages_stack_buffer = new_available_pages_stack_buffer;
        gpu_resources.voxel_malloc.released_pages_stack_buffer = new_released_pages_stack_buffer;

        task_voxel_malloc_pages_buffer.set_buffers({.buffers = std::array{gpu_resources.voxel_malloc.pages_buffer, gpu_resources.voxel_malloc.available_pages_stack_buffer, gpu_resources.voxel_malloc.released_pages_stack_buffer}});
    }

    condition_values[static_cast<usize>(Conditions::STARTUP)] = ui.should_run_startup || model_is_ready;
    condition_values[static_cast<usize>(Conditions::UPLOAD_SETTINGS)] = ui.should_upload_settings;
    condition_values[static_cast<usize>(Conditions::UPLOAD_GVOX_MODEL)] = model_is_ready;
    condition_values[static_cast<usize>(Conditions::VOXEL_MALLOC_REALLOC)] = gpu_resources.voxel_malloc.next_page_count != 0;
    gpu_input.fif_index = gpu_input.frame_index % (FRAMES_IN_FLIGHT + 1);
    main_task_list.execute({.permutation_condition_values = condition_values});

    gpu_input.resize_factor = 1.0f;
    gpu_input.mouse.pos_delta = {0.0f, 0.0f};
    gpu_input.mouse.scroll_delta = {0.0f, 0.0f};

    ui.debug_gpu_heap_usage = gpu_output.heap_size;
    ui.debug_player_pos = gpu_output.player_pos;
    ui.debug_page_count = gpu_resources.voxel_malloc.current_page_count;
    ui.debug_job_counters = std::bit_cast<ChunkHierarchyJobCounters>(gpu_output.job_counters_packed);
    ui.debug_total_jobs_ran = gpu_output.total_jobs_ran;

    task_render_pos_image.swap_images(task_render_prev_pos_image);
    task_render_col_image.swap_images(task_render_prev_col_image);
    ++gpu_input.frame_index;
}
void VoxelApp::on_mouse_move(f32 x, f32 y) {
    f32vec2 const center = {static_cast<f32>(window_size.x / 2), static_cast<f32>(window_size.y / 2)};
    gpu_input.mouse.pos = f32vec2{x, y};
    auto offset = gpu_input.mouse.pos - center;
    gpu_input.mouse.pos = gpu_input.mouse.pos *f32vec2{static_cast<f32>(gpu_resources.render_images.size.x), static_cast<f32>(gpu_resources.render_images.size.y)} / f32vec2{static_cast<f32>(window_size.x), static_cast<f32>(window_size.y)};
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
    if (ui.limbo_action_index != GAME_ACTION_LAST + 1) {
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
    if (ui.limbo_action_index != GAME_ACTION_LAST + 1) {
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
        if (key_id == GLFW_KEY_N && action == GLFW_PRESS) {
            ui.toggle_node_editor();
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
            gpu_resources.render_images.size.x = static_cast<u32>(static_cast<f32>(window_size.x) * render_res_scl);
            gpu_resources.render_images.size.y = static_cast<u32>(static_cast<f32>(window_size.y) * render_res_scl);
            recreate_render_images();
        }
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
    gpu_resources.render_images.destroy(device);
    gpu_resources.render_images.create(device);
    task_render_pos_image.set_images({.images = {&gpu_resources.render_images.pos_images[gpu_input.frame_index % 2], 1}});
    task_render_col_image.set_images({.images = {&gpu_resources.render_images.col_images[gpu_input.frame_index % 2], 1}});
    task_render_prev_pos_image.set_images({.images = {&gpu_resources.render_images.pos_images[(gpu_input.frame_index + 1) % 2], 1}});
    task_render_prev_col_image.set_images({.images = {&gpu_resources.render_images.col_images[(gpu_input.frame_index + 1) % 2], 1}});
    task_render_final_image.set_images({.images = {&gpu_resources.render_images.final_image, 1}});
    task_render_raster_color_image.set_images({.images = std::array{gpu_resources.render_images.raster_color_image}});
    task_render_raster_depth_image.set_images({.images = std::array{gpu_resources.render_images.raster_depth_image}});
    needs_vram_calc = true;
}
void VoxelApp::recreate_voxel_chunks() {
    // main_task_list.remove_runtime_buffer(task_voxel_chunks_buffer, gpu_resources.voxel_chunks.buffer);
    gpu_resources.voxel_chunks.destroy(device);
    gpu_resources.voxel_chunks.create(device, ui.settings.log2_chunks_per_axis);
    task_voxel_chunks_buffer.set_buffers({.buffers = {&gpu_resources.voxel_chunks.buffer, 1}});
    ui.should_recreate_voxel_buffers = false;
    needs_vram_calc = true;
}

void VoxelApp::run_startup(daxa::TaskList &temp_task_list) {
    temp_task_list.add_task({
        .uses = {
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{task_globals_buffer},
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{task_temp_voxel_chunks_buffer},
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{task_voxel_chunks_buffer},
#if USE_OLD_ALLOC
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{task_gpu_heap_buffer},
#endif
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{task_voxel_malloc_pages_buffer},
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
                .size = static_cast<u32>(sizeof(VoxelLeafChunk)) * chunk_n,
                .clear_value = 0,
            });
#if USE_OLD_ALLOC
            cmd_list.clear_buffer({
                .buffer = task_gpu_heap_buffer.get_state().buffers[0],
                .offset = 0,
                .size = ui.settings.gpu_heap_size,
                .clear_value = 0,
            });
#endif
            cmd_list.clear_buffer({
                .buffer = task_voxel_malloc_pages_buffer.get_state().buffers[0],
                .offset = 0,
                .size = VOXEL_MALLOC_PAGE_SIZE_BYTES * gpu_resources.voxel_malloc.current_page_count,
                .clear_value = 0,
            });
            cmd_list.clear_buffer({
                .buffer = task_voxel_malloc_pages_buffer.get_state().buffers[1],
                .offset = 0,
                .size = static_cast<u32>(sizeof(VoxelMalloc_PageIndex)) * gpu_resources.voxel_malloc.current_page_count,
                .clear_value = 0,
            });
            cmd_list.clear_buffer({
                .buffer = task_voxel_malloc_pages_buffer.get_state().buffers[2],
                .offset = 0,
                .size = static_cast<u32>(sizeof(VoxelMalloc_PageIndex)) * gpu_resources.voxel_malloc.current_page_count,
                .clear_value = 0,
            });
        },
        .name = "StartupTask (Globals Clear)",
    });
    temp_task_list.add_task(StartupComputeTask{
        {
            .uses = {
                .settings = task_settings_buffer.handle(),
                .globals = task_globals_buffer.handle(),
                .voxel_chunks = task_voxel_chunks_buffer.handle(),
            },
        },
        &startup_task_state,
    });

    temp_task_list.add_task({
        .uses = {
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{task_voxel_malloc_global_allocator_buffer},
        },
        .task = [this](daxa::TaskInterface task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            auto staging_global_allocator_buffer = device.create_buffer({
                .size = sizeof(VoxelMalloc_GlobalAllocator),
                .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = "staging_global_allocator_buffer",
            });
            cmd_list.destroy_buffer_deferred(staging_global_allocator_buffer);
            auto *buffer_ptr = device.get_host_address_as<VoxelMalloc_GlobalAllocator>(staging_global_allocator_buffer);
            *buffer_ptr = VoxelMalloc_GlobalAllocator {
#if USE_OLD_ALLOC
                .offset = 0,
                .heap = device.get_device_address(gpu_resources.gpu_heap.buffer),
#else
                .heap = device.get_device_address(gpu_resources.voxel_malloc.pages_buffer),
                .available_pages_stack = device.get_device_address(gpu_resources.voxel_malloc.available_pages_stack_buffer),
                .released_pages_stack = device.get_device_address(gpu_resources.voxel_malloc.released_pages_stack_buffer),
                .page_count = 0,
                .available_pages_stack_size = 0,
                .released_pages_stack_size = 0,
#endif
            };
            cmd_list.copy_buffer_to_buffer({
                .src_buffer = staging_global_allocator_buffer,
                .dst_buffer = task_voxel_malloc_global_allocator_buffer.get_state().buffers[0],
                .size = sizeof(VoxelMalloc_GlobalAllocator),
            });
        },
        .name = "Initialize",
    });
}
void VoxelApp::upload_settings(daxa::TaskList &temp_task_list) {
    temp_task_list.add_task({
        .uses = {
            daxa::TaskBufferUse<daxa::TaskBufferAccess::HOST_TRANSFER_WRITE>{task_settings_buffer},
        },
        .task = [this](daxa::TaskInterface task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            auto staging_settings_buffer = device.create_buffer({
                .size = sizeof(GpuSettings),
                .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = "staging_settings_buffer",
            });
            cmd_list.destroy_buffer_deferred(staging_settings_buffer);
            auto *buffer_ptr = device.get_host_address_as<GpuSettings>(staging_settings_buffer);
            *buffer_ptr = {
                .fov = ui.settings.camera_fov * (std::numbers::pi_v<f32> / 180.0f),
                .sensitivity = ui.settings.mouse_sensitivity,
                .log2_chunks_per_axis = ui.settings.log2_chunks_per_axis,
            };
            cmd_list.copy_buffer_to_buffer({
                .src_buffer = staging_settings_buffer,
                .dst_buffer = task_settings_buffer.get_state().buffers[0],
                .size = sizeof(GpuSettings),
            });
            ui.should_upload_settings = false;
        },
        .name = "StartupTask (Globals Clear)",
    });
}
void VoxelApp::upload_model(daxa::TaskList &temp_task_list) {
    temp_task_list.add_task({
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
            if (gvox_model_data.ptr) {
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
}
void VoxelApp::voxel_malloc_realloc(daxa::TaskList &temp_task_list) {
    temp_task_list.add_task({
        .uses = {
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_READ>{task_voxel_malloc_old_pages_buffer},
            daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{task_voxel_malloc_pages_buffer},
        },
        .task = [&](daxa::TaskInterface task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            cmd_list.copy_buffer_to_buffer({
                .src_buffer = task_voxel_malloc_old_pages_buffer.get_state().buffers[0],
                .src_offset = 0,
                .dst_buffer = task_voxel_malloc_pages_buffer.get_state().buffers[0],
                .dst_offset = 0,
                .size = VOXEL_MALLOC_PAGE_SIZE_BYTES * prev_page_count,
            });
            cmd_list.copy_buffer_to_buffer({
                .src_buffer = task_voxel_malloc_old_pages_buffer.get_state().buffers[1],
                .src_offset = 0,
                .dst_buffer = task_voxel_malloc_pages_buffer.get_state().buffers[1],
                .dst_offset = 0,
                .size = VOXEL_MALLOC_PAGE_SIZE_BYTES * sizeof(VoxelMalloc_PageIndex),
            });
            cmd_list.copy_buffer_to_buffer({
                .src_buffer = task_voxel_malloc_old_pages_buffer.get_state().buffers[2],
                .src_offset = 0,
                .dst_buffer = task_voxel_malloc_pages_buffer.get_state().buffers[2],
                .dst_offset = 0,
                .size = VOXEL_MALLOC_PAGE_SIZE_BYTES * sizeof(VoxelMalloc_PageIndex),
            });

            cmd_list.destroy_buffer_deferred(task_voxel_malloc_old_pages_buffer.get_state().buffers[0]);
            cmd_list.destroy_buffer_deferred(task_voxel_malloc_old_pages_buffer.get_state().buffers[1]);
            cmd_list.destroy_buffer_deferred(task_voxel_malloc_old_pages_buffer.get_state().buffers[2]);
            task_voxel_malloc_old_pages_buffer.set_buffers({});

#if !USE_OLD_ALLOC
            auto staging_global_allocator_buffer = device.create_buffer({
                .size = sizeof(VoxelMalloc_GlobalAllocator),
                .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = "staging_global_allocator_buffer",
            });
            cmd_list.destroy_buffer_deferred(staging_global_allocator_buffer);
            auto *buffer_ptr = device.get_host_address_as<VoxelMalloc_GlobalAllocator>(staging_global_allocator_buffer);
            *buffer_ptr = VoxelMalloc_GlobalAllocator{
                .heap = device.get_device_address(gpu_resources.voxel_malloc.pages_buffer),
                .available_pages_stack = device.get_device_address(gpu_resources.voxel_malloc.available_pages_stack_buffer),
                .released_pages_stack = device.get_device_address(gpu_resources.voxel_malloc.released_pages_stack_buffer),
            };
            cmd_list.copy_buffer_to_buffer({
                .src_buffer = staging_global_allocator_buffer,
                .dst_buffer = gpu_resources.voxel_malloc.global_allocator_buffer,
                .size = offsetof(VoxelMalloc_GlobalAllocator, page_count),
            });
#endif

            needs_vram_calc = true;
        },
        .name = "Transfer Task",
    });
}

auto VoxelApp::record_main_task_list() -> daxa::TaskList {
    daxa::TaskList result_task_list = daxa::TaskList({
        .device = device,
        .swapchain = swapchain,
        .permutation_condition_count = static_cast<usize>(Conditions::LAST),
        .name = "main_task_list",
    });

    result_task_list.use_persistent_image(task_blue_noise_vec1_image);
    result_task_list.use_persistent_image(task_blue_noise_vec2_image);
    result_task_list.use_persistent_image(task_blue_noise_unit_vec3_image);
    result_task_list.use_persistent_image(task_blue_noise_cosine_vec3_image);
    task_blue_noise_vec1_image.set_images({.images = std::array{gpu_resources.blue_noise_vec1_image}});
    task_blue_noise_vec2_image.set_images({.images = std::array{gpu_resources.blue_noise_vec2_image}});
    task_blue_noise_unit_vec3_image.set_images({.images = std::array{gpu_resources.blue_noise_unit_vec3_image}});
    task_blue_noise_cosine_vec3_image.set_images({.images = std::array{gpu_resources.blue_noise_cosine_vec3_image}});

    result_task_list.use_persistent_buffer(task_settings_buffer);
    result_task_list.use_persistent_buffer(task_input_buffer);
    result_task_list.use_persistent_buffer(task_output_buffer);
    result_task_list.use_persistent_buffer(task_staging_output_buffer);
    result_task_list.use_persistent_buffer(task_globals_buffer);
    result_task_list.use_persistent_buffer(task_test_data_buffer);
    result_task_list.use_persistent_buffer(task_temp_voxel_chunks_buffer);
    result_task_list.use_persistent_buffer(task_voxel_malloc_global_allocator_buffer);
    result_task_list.use_persistent_buffer(task_voxel_chunks_buffer);
    result_task_list.use_persistent_buffer(task_gvox_model_buffer);
    result_task_list.use_persistent_buffer(task_voxel_malloc_pages_buffer);
    result_task_list.use_persistent_buffer(task_voxel_malloc_old_pages_buffer);

    task_settings_buffer.set_buffers({.buffers = std::array{gpu_resources.settings_buffer}});
    task_input_buffer.set_buffers({.buffers = std::array{gpu_resources.input_buffer}});
    task_output_buffer.set_buffers({.buffers = std::array{gpu_resources.output_buffer}});
    task_staging_output_buffer.set_buffers({.buffers = std::array{gpu_resources.staging_output_buffer}});
    task_globals_buffer.set_buffers({.buffers = std::array{gpu_resources.globals_buffer}});
#if 0
    task_test_data_buffer.set_buffers({.buffers = std::array{gpu_resources.test_data_buffer}});
#endif
    task_temp_voxel_chunks_buffer.set_buffers({.buffers = std::array{gpu_resources.temp_voxel_chunks_buffer}});
    task_voxel_malloc_global_allocator_buffer.set_buffers({.buffers = std::array{gpu_resources.voxel_malloc.global_allocator_buffer}});
    task_voxel_chunks_buffer.set_buffers({.buffers = std::array{gpu_resources.voxel_chunks.buffer}});
    task_gvox_model_buffer.set_buffers({.buffers = std::array{gpu_resources.gvox_model_buffer}});
    task_voxel_malloc_pages_buffer.set_buffers({.buffers = std::array{gpu_resources.voxel_malloc.pages_buffer, gpu_resources.voxel_malloc.available_pages_stack_buffer, gpu_resources.voxel_malloc.released_pages_stack_buffer}});
    task_voxel_malloc_old_pages_buffer.set_buffers({.buffers = std::array{gpu_resources.voxel_malloc.pages_buffer, gpu_resources.voxel_malloc.available_pages_stack_buffer, gpu_resources.voxel_malloc.released_pages_stack_buffer}});

    result_task_list.use_persistent_buffer(task_simulated_voxel_particles_buffer);
    result_task_list.use_persistent_buffer(task_rendered_voxel_particles_buffer);
    task_simulated_voxel_particles_buffer.set_buffers({.buffers = std::array{gpu_resources.simulated_voxel_particles_buffer}});
    task_rendered_voxel_particles_buffer.set_buffers({.buffers = std::array{gpu_resources.rendered_voxel_particles_buffer}});

#if USE_OLD_ALLOC
    result_task_list.use_persistent_buffer(task_gpu_heap_buffer);
    task_gpu_heap_buffer.set_buffers({.buffers = std::array{gpu_resources.gpu_heap.buffer}});
#endif

    result_task_list.use_persistent_image(task_swapchain_image);
    result_task_list.use_persistent_image(task_render_pos_image);
    result_task_list.use_persistent_image(task_render_col_image);
    result_task_list.use_persistent_image(task_render_prev_pos_image);
    result_task_list.use_persistent_image(task_render_prev_col_image);
    result_task_list.use_persistent_image(task_render_final_image);
    result_task_list.use_persistent_image(task_render_raster_color_image);
    result_task_list.use_persistent_image(task_render_raster_depth_image);
    task_swapchain_image.set_images({.images = std::array{swapchain_image}});
    task_render_pos_image.set_images({.images = std::array{gpu_resources.render_images.pos_images[gpu_input.frame_index % 2]}});
    task_render_col_image.set_images({.images = std::array{gpu_resources.render_images.col_images[gpu_input.frame_index % 2]}});
    task_render_prev_pos_image.set_images({.images = std::array{gpu_resources.render_images.pos_images[(gpu_input.frame_index + 1) % 2]}});
    task_render_prev_col_image.set_images({.images = std::array{gpu_resources.render_images.col_images[(gpu_input.frame_index + 1) % 2]}});
    task_render_final_image.set_images({.images = std::array{gpu_resources.render_images.final_image}});
    task_render_raster_color_image.set_images({.images = std::array{gpu_resources.render_images.raster_color_image}});
    task_render_raster_depth_image.set_images({.images = std::array{gpu_resources.render_images.raster_depth_image}});

    result_task_list.conditional({
        .condition_index = static_cast<u32>(Conditions::STARTUP),
        .when_true = [&, this]() { this->run_startup(result_task_list); },
    });
    result_task_list.conditional({
        .condition_index = static_cast<u32>(Conditions::UPLOAD_SETTINGS),
        .when_true = [&, this]() { this->upload_settings(result_task_list); },
    });
    result_task_list.conditional({
        .condition_index = static_cast<u32>(Conditions::UPLOAD_GVOX_MODEL),
        .when_true = [&, this]() { this->upload_model(result_task_list); },
    });
    result_task_list.conditional({
        .condition_index = static_cast<u32>(Conditions::VOXEL_MALLOC_REALLOC),
        .when_true = [&, this]() { this->voxel_malloc_realloc(result_task_list); },
    });

    // GpuInputUploadTransferTask
    result_task_list.add_task({
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

    // PerframeTask
    result_task_list.add_task(PerframeComputeTask{
        {
            .uses = {
                .settings = task_settings_buffer.handle(),
                .gpu_input = task_input_buffer.handle(),
                .gpu_output = task_output_buffer.handle(),
                .globals = task_globals_buffer.handle(),
                .simulated_voxel_particles = task_simulated_voxel_particles_buffer.handle(),
                .voxel_malloc_global_allocator = task_voxel_malloc_global_allocator_buffer.handle(),
                .voxel_chunks = task_voxel_chunks_buffer.handle(),
            },
        },
        &perframe_task_state,
    });

    // VoxelParticleSimComputeTask
    result_task_list.add_task(VoxelParticleSimComputeTask{
        {
            .uses = {
                .settings = task_settings_buffer.handle(),
                .gpu_input = task_input_buffer.handle(),
                .globals = task_globals_buffer.handle(),
                .voxel_malloc_global_allocator = task_voxel_malloc_global_allocator_buffer.handle(),
                .voxel_chunks = task_voxel_chunks_buffer.handle(),
                .simulated_voxel_particles = task_simulated_voxel_particles_buffer.handle(),
                .rendered_voxel_particles = task_rendered_voxel_particles_buffer.handle(),
            },
        },
        &voxel_particle_sim_task_state,
    });

    // VoxelParticleRasterTask
    result_task_list.add_task(VoxelParticleRasterTask{
        {
            .uses = {
                .globals = task_globals_buffer.handle(),
                .simulated_voxel_particles = task_simulated_voxel_particles_buffer.handle(),
                .rendered_voxel_particles = task_rendered_voxel_particles_buffer.handle(),
                .render_image = task_render_raster_color_image.handle(),
                .depth_image = task_render_raster_depth_image.handle(),
            },
        },
        &voxel_particle_raster_task_state,
    });

    // ChunkHierarchy
    result_task_list.add_task(ChunkHierarchyComputeTaskL0{
        {
            .uses = {
                .settings = task_settings_buffer.handle(),
                .gpu_input = task_input_buffer.handle(),
                .gvox_model = task_gvox_model_buffer.handle(),
                .globals = task_globals_buffer.handle(),
                .voxel_chunks = task_voxel_chunks_buffer.handle(),
            },
        },
        &chunk_hierarchy_task_state,
    });
    result_task_list.add_task(ChunkHierarchyComputeTaskL1{
        {
            .uses = {
                .settings = task_settings_buffer.handle(),
                .gpu_input = task_input_buffer.handle(),
                .gvox_model = task_gvox_model_buffer.handle(),
                .globals = task_globals_buffer.handle(),
                .voxel_chunks = task_voxel_chunks_buffer.handle(),
            },
        },
        &chunk_hierarchy_task_state,
    });

    // ChunkEdit
    result_task_list.add_task(ChunkEditComputeTask{
        {
            .uses = {
                .settings = task_settings_buffer.handle(),
                .gpu_input = task_input_buffer.handle(),
                .globals = task_globals_buffer.handle(),
                .gvox_model = task_gvox_model_buffer.handle(),
#if 0
                .test_data_buffer = task_test_data_buffer.handle(),
#endif
                .voxel_chunks = task_voxel_chunks_buffer.handle(),
                .temp_voxel_chunks = task_temp_voxel_chunks_buffer.handle(),
                .voxel_malloc_global_allocator = task_voxel_malloc_global_allocator_buffer.handle(),
            },
        },
        &chunk_edit_task_state,
    });

    // ChunkOpt_x2x4
    result_task_list.add_task(ChunkOpt_x2x4_ComputeTask{
        {
            .uses = {
                .settings = task_settings_buffer.handle(),
                .gpu_input = task_input_buffer.handle(),
                .globals = task_globals_buffer.handle(),
                .temp_voxel_chunks = task_temp_voxel_chunks_buffer.handle(),
                .voxel_chunks = task_voxel_chunks_buffer.handle(),
            },
        },
        &chunk_opt_x2x4_task_state,
    });

    // ChunkOpt_x8up
    result_task_list.add_task(ChunkOpt_x8up_ComputeTask{
        {
            .uses = {
                .settings = task_settings_buffer.handle(),
                .gpu_input = task_input_buffer.handle(),
                .globals = task_globals_buffer.handle(),
                .temp_voxel_chunks = task_temp_voxel_chunks_buffer.handle(),
                .voxel_chunks = task_voxel_chunks_buffer.handle(),
            },
        },
        &chunk_opt_x8up_task_state,
    });

    // ChunkAlloc
    result_task_list.add_task(ChunkAllocComputeTask{
        {
            .uses = {
                .settings = task_settings_buffer.handle(),
                .globals = task_globals_buffer.handle(),
                .temp_voxel_chunks = task_temp_voxel_chunks_buffer.handle(),
                .voxel_chunks = task_voxel_chunks_buffer.handle(),
                .voxel_malloc_global_allocator = task_voxel_malloc_global_allocator_buffer.handle(),
            },
        },
        &chunk_alloc_task_state,
    });

    // TracePrimaryTask
    result_task_list.add_task(TracePrimaryComputeTask{
        {
            .uses = {
                .settings = task_settings_buffer.handle(),
                .gpu_input = task_input_buffer.handle(),
                .globals = task_globals_buffer.handle(),
                .voxel_malloc_global_allocator = task_voxel_malloc_global_allocator_buffer.handle(),
                .voxel_chunks = task_voxel_chunks_buffer.handle(),
                .blue_noise_vec2 = task_blue_noise_vec2_image.handle(),
                .render_pos_image_id = task_render_pos_image.handle(),
            },
        },
        &trace_primary_task_state,
    });

    // ColorSceneTask
    result_task_list.add_task(ColorSceneComputeTask{
        {
            .uses = {
                .settings = task_settings_buffer.handle(),
                .gpu_input = task_input_buffer.handle(),
                .globals = task_globals_buffer.handle(),
                .voxel_malloc_global_allocator = task_voxel_malloc_global_allocator_buffer.handle(),
                .voxel_chunks = task_voxel_chunks_buffer.handle(),
                .blue_noise_cosine_vec3 = task_blue_noise_cosine_vec3_image.handle(),
                .render_pos_image_id = task_render_pos_image.handle(),
                .render_prev_pos_image_id = task_render_prev_pos_image.handle(),
                .render_col_image_id = task_render_col_image.handle(),
                .render_prev_col_image_id = task_render_prev_col_image.handle(),
                .raster_color_image = task_render_raster_color_image.handle(),
                .simulated_voxel_particles = task_simulated_voxel_particles_buffer.handle(),
            },
        },
        &color_scene_task_state,
    });

    // PostprocessingTask
    result_task_list.add_task(PostprocessingComputeTask{
        {
            .uses = {
                .settings = task_settings_buffer.handle(),
                .gpu_input = task_input_buffer.handle(),
                .render_col_image_id = task_render_col_image.handle(),
                .final_image_id = task_render_final_image.handle(),
            },
        },
        &postprocessing_task_state,
    });

    // GpuOutputDownloadTransferTask
    result_task_list.add_task({
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
            u32 offset = frame_index % (FRAMES_IN_FLIGHT + 1);
            gpu_output = (*buffer_ptr)[offset];
            cmd_list.copy_buffer_to_buffer({
                .src_buffer = output_buffer,
                .dst_buffer = staging_output_buffer,
                .size = sizeof(GpuOutput) * (FRAMES_IN_FLIGHT + 1),
            });
        },
        .name = "GpuOutputDownloadTransferTask",
    });

    // Blit (render to swapchain)
    result_task_list.add_task({
        .uses = {
            daxa::TaskImageUse<daxa::TaskImageAccess::TRANSFER_READ, daxa::ImageViewType::REGULAR_2D>{task_render_final_image},
            daxa::TaskImageUse<daxa::TaskImageAccess::TRANSFER_WRITE, daxa::ImageViewType::REGULAR_2D>{task_swapchain_image},
        },
        .task = [this](daxa::TaskInterface task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            cmd_list.blit_image_to_image({
                .src_image = task_render_final_image.get_state().images[0],
                .src_image_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
                .dst_image = task_swapchain_image.get_state().images[0],
                .dst_image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                .src_slice = {.image_aspect = daxa::ImageAspectFlagBits::COLOR},
                .src_offsets = {{{0, 0, 0}, {static_cast<i32>(gpu_resources.render_images.size.x), static_cast<i32>(gpu_resources.render_images.size.y), 1}}},
                .dst_slice = {.image_aspect = daxa::ImageAspectFlagBits::COLOR},
                .dst_offsets = {{{0, 0, 0}, {static_cast<i32>(window_size.x), static_cast<i32>(window_size.y), 1}}},
                .filter = daxa::Filter::LINEAR,
            });
        },
        .name = "Blit (render to swapchain)",
    });

    // ImGui draw
    result_task_list.add_task({
        .uses = {
            daxa::TaskImageUse<daxa::TaskImageAccess::COLOR_ATTACHMENT, daxa::ImageViewType::REGULAR_2D>{task_swapchain_image},
        },
        .task = [this](daxa::TaskInterface task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            imgui_renderer.record_commands(ImGui::GetDrawData(), cmd_list, swapchain_image, window_size.x, window_size.y);
        },
        .name = "ImGui draw",
    });

    result_task_list.submit({});
    result_task_list.present({});
    result_task_list.complete({});

    return result_task_list;
}
