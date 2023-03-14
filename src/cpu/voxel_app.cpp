#include "voxel_app.hpp"

#include <thread>
#include <numbers>
#include <fstream>

#include <gvox/adapters/input/byte_buffer.h>
#include <gvox/adapters/output/byte_buffer.h>
#include <gvox/adapters/parse/voxlap.h>

#define APPNAME "Voxel App"
#define APPNAME_PREFIX(x) ("[" APPNAME "] " x)

using namespace std::chrono_literals;

#include <iostream>

void GpuInputUploadTransferTask::record(daxa::Device &device, daxa::CommandList &cmd_list, daxa::BufferId input_buffer, GpuInput &gpu_input) {
    auto staging_input_buffer = device.create_buffer({
        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
        .size = sizeof(GpuInput),
        .debug_name = APPNAME_PREFIX("staging_input_buffer"),
    });
    cmd_list.destroy_buffer_deferred(staging_input_buffer);
    auto *buffer_ptr = device.get_host_address_as<GpuInput>(staging_input_buffer);
    *buffer_ptr = gpu_input;
    cmd_list.copy_buffer_to_buffer({
        .src_buffer = staging_input_buffer,
        .dst_buffer = input_buffer,
        .size = sizeof(GpuInput),
    });
}

void StartupTask::record(
    daxa::CommandList &cmd_list,
    BDA globals_buffer_ptr,
    BDA voxel_chunks_buffer_ptr) const {
    if (!pipeline) {
        return;
    }
    cmd_list.set_pipeline(*pipeline);
    cmd_list.push_constant(StartupComputePush{
        .gpu_globals = globals_buffer_ptr,
        .voxel_chunks = voxel_chunks_buffer_ptr,
    });
    cmd_list.dispatch(1, 1, 1);
}

void PerframeTask::record(
    daxa::CommandList &cmd_list,
    BDA settings_buffer_ptr,
    BDA input_buffer_ptr,
    BDA output_buffer_ptr,
    BDA globals_buffer_ptr,
    BDA voxel_chunks_buffer_ptr,
    GpuAllocator gpu_allocator) const {
    if (!pipeline) {
        return;
    }
    cmd_list.set_pipeline(*pipeline);
    cmd_list.push_constant(PerframeComputePush{
        .gpu_settings = settings_buffer_ptr,
        .gpu_input = input_buffer_ptr,
        .gpu_output = output_buffer_ptr,
        .gpu_globals = globals_buffer_ptr,
        .voxel_chunks = voxel_chunks_buffer_ptr,
        .gpu_allocator = gpu_allocator,
    });
    cmd_list.dispatch(1, 1, 1);
}

void PerChunkTask::record(
    daxa::CommandList &cmd_list,
    BDA settings_buffer_ptr,
    BDA input_buffer_ptr,
    BDA globals_buffer_ptr,
    BDA voxel_chunks_buffer_ptr,
    u32vec3 chunk_n) const {
    if (!pipeline) {
        return;
    }
    cmd_list.set_pipeline(*pipeline);
    cmd_list.push_constant(PerChunkComputePush{
        .gpu_settings = settings_buffer_ptr,
        .gpu_input = input_buffer_ptr,
        .gpu_globals = globals_buffer_ptr,
        .voxel_chunks = voxel_chunks_buffer_ptr,
    });
    cmd_list.dispatch((chunk_n.x + 7) / 8, (chunk_n.y + 7) / 8, (chunk_n.z + 7) / 8);
}

void ChunkEdit::record(
    daxa::CommandList &cmd_list,
    BDA settings_buffer_ptr,
    BDA input_buffer_ptr,
    BDA globals_buffer_ptr,
    BDA temp_voxel_chunks_ptr,
    daxa::BufferId globals_buffer_id) const {
    if (!pipeline) {
        return;
    }
    cmd_list.set_pipeline(*pipeline);
    cmd_list.push_constant(ChunkEditComputePush{
        .gpu_settings = settings_buffer_ptr,
        .gpu_input = input_buffer_ptr,
        .gpu_globals = globals_buffer_ptr,
        .temp_voxel_chunks = temp_voxel_chunks_ptr,
    });
    cmd_list.dispatch_indirect({
        .indirect_buffer = globals_buffer_id,
        .offset = offsetof(GpuGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, chunk_edit_dispatch),
    });
}

void ChunkOpt_x2x4::record(
    daxa::CommandList &cmd_list,
    BDA settings_buffer_ptr,
    BDA input_buffer_ptr,
    BDA globals_buffer_ptr,
    BDA temp_voxel_chunks_ptr,
    BDA voxel_chunks_buffer_ptr,
    daxa::BufferId globals_buffer_id) const {
    if (!pipeline) {
        return;
    }
    cmd_list.set_pipeline(*pipeline);
    cmd_list.push_constant(ChunkOptComputePush{
        .gpu_settings = settings_buffer_ptr,
        .gpu_input = input_buffer_ptr,
        .gpu_globals = globals_buffer_ptr,
        .temp_voxel_chunks = temp_voxel_chunks_ptr,
        .voxel_chunks = voxel_chunks_buffer_ptr,
    });
    cmd_list.dispatch_indirect({
        .indirect_buffer = globals_buffer_id,
        .offset = offsetof(GpuGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, subchunk_x2x4_dispatch),
    });
}

void ChunkOpt_x8up::record(
    daxa::CommandList &cmd_list,
    BDA settings_buffer_ptr,
    BDA input_buffer_ptr,
    BDA globals_buffer_ptr,
    BDA temp_voxel_chunks_ptr,
    BDA voxel_chunks_buffer_ptr,
    daxa::BufferId globals_buffer_id) const {
    if (!pipeline) {
        return;
    }
    cmd_list.set_pipeline(*pipeline);
    cmd_list.push_constant(ChunkOptComputePush{
        .gpu_settings = settings_buffer_ptr,
        .gpu_input = input_buffer_ptr,
        .gpu_globals = globals_buffer_ptr,
        .temp_voxel_chunks = temp_voxel_chunks_ptr,
        .voxel_chunks = voxel_chunks_buffer_ptr,
    });
    cmd_list.dispatch_indirect({
        .indirect_buffer = globals_buffer_id,
        .offset = offsetof(GpuGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, subchunk_x8up_dispatch),
    });
}

void ChunkAlloc::record(
    daxa::CommandList &cmd_list,
    BDA settings_buffer_ptr,
    BDA globals_buffer_ptr,
    BDA temp_voxel_chunks_ptr,
    BDA voxel_chunks_buffer_ptr,
    GpuAllocator gpu_allocator,
    daxa::BufferId globals_buffer_id) const {
    if (!pipeline) {
        return;
    }
    cmd_list.set_pipeline(*pipeline);
    cmd_list.push_constant(ChunkAllocComputePush{
        .gpu_settings = settings_buffer_ptr,
        .gpu_globals = globals_buffer_ptr,
        .temp_voxel_chunks = temp_voxel_chunks_ptr,
        .voxel_chunks = voxel_chunks_buffer_ptr,
        .gpu_allocator = gpu_allocator,
    });
    cmd_list.dispatch_indirect({
        .indirect_buffer = globals_buffer_id,
        // NOTE: This should always have the same value as the chunk edit dispatch, so we're re-using it here
        .offset = offsetof(GpuGlobals, indirect_dispatch) + offsetof(GpuIndirectDispatch, chunk_edit_dispatch),
    });
}

void TracePrimaryTask::record(
    daxa::CommandList &cmd_list,
    BDA settings_buffer_ptr,
    BDA input_buffer_ptr,
    BDA globals_buffer_ptr,
    BDA gpu_heap_buffer_ptr,
    BDA voxel_chunks_buffer_ptr,
    daxa::ImageId render_image,
    u32vec2 render_size) const {
    if (!pipeline) {
        return;
    }
    cmd_list.set_pipeline(*pipeline);
    cmd_list.push_constant(TracePrimaryComputePush{
        .gpu_settings = settings_buffer_ptr,
        .gpu_input = input_buffer_ptr,
        .gpu_globals = globals_buffer_ptr,
        .gpu_heap = gpu_heap_buffer_ptr,
        .voxel_chunks = voxel_chunks_buffer_ptr,
        .render_pos_image_id = render_image.default_view(),
    });
    cmd_list.dispatch((render_size.x + 7) / 8, (render_size.y + 7) / 8);
}

void ColorSceneTask::record(
    daxa::CommandList &cmd_list,
    BDA settings_buffer_ptr,
    BDA input_buffer_ptr,
    BDA globals_buffer_ptr,
    BDA gpu_heap_buffer_ptr,
    BDA voxel_chunks_buffer_ptr,
    daxa::ImageId render_pos_image,
    daxa::ImageId render_col_image,
    u32vec2 render_size) const {
    if (!pipeline) {
        return;
    }
    cmd_list.set_pipeline(*pipeline);
    cmd_list.push_constant(ColorSceneComputePush{
        .gpu_settings = settings_buffer_ptr,
        .gpu_input = input_buffer_ptr,
        .gpu_globals = globals_buffer_ptr,
        .gpu_heap = gpu_heap_buffer_ptr,
        .voxel_chunks = voxel_chunks_buffer_ptr,
        .render_pos_image_id = render_pos_image.default_view(),
        .render_col_image_id = render_col_image.default_view(),
    });
    cmd_list.dispatch((render_size.x + 7) / 8, (render_size.y + 7) / 8);
}

void PostprocessingTask::record(
    daxa::CommandList &cmd_list,
    BDA settings_buffer_ptr,
    BDA input_buffer_ptr,
    BDA globals_buffer_ptr,
    BDA allocator_state_buffer_info,
    daxa::ImageId render_col_image,
    daxa::ImageId final_image,
    u32vec2 render_size) const {
    if (!pipeline) {
        return;
    }
    cmd_list.set_pipeline(*pipeline);
    cmd_list.push_constant(PostprocessingComputePush{
        .gpu_settings = settings_buffer_ptr,
        .gpu_input = input_buffer_ptr,
        .gpu_globals = globals_buffer_ptr,
        .gpu_allocator_state = allocator_state_buffer_info,
        .render_col_image_id = render_col_image.default_view(),
        .final_image_id = final_image.default_view(),
    });
    cmd_list.dispatch((render_size.x + 7) / 8, (render_size.y + 7) / 8);
}

void GpuOutputDownloadTransferTask::record(daxa::Device &device, daxa::CommandList &cmd_list, daxa::BufferId output_buffer, daxa::BufferId staging_output_buffer, GpuOutput &gpu_output, u32 frame_index) {
    auto *buffer_ptr = device.get_host_address_as<std::array<GpuOutput, (FRAMES_IN_FLIGHT + 1)>>(staging_output_buffer);
    u32 offset = frame_index % (FRAMES_IN_FLIGHT + 1);
    gpu_output = (*buffer_ptr)[offset];
    cmd_list.copy_buffer_to_buffer({
        .src_buffer = output_buffer,
        .dst_buffer = staging_output_buffer,
        .size = sizeof(GpuOutput) * (FRAMES_IN_FLIGHT + 1),
    });
}

void RenderImages::create(daxa::Device &device) {
    pos_image = device.create_image({
        .format = daxa::Format::R32G32B32A32_SFLOAT,
        .size = {size.x, size.y, 1},
        .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::SHADER_READ_ONLY,
        .debug_name = "pos_image",
    });
    col_image = device.create_image({
        .format = daxa::Format::R32G32B32A32_SFLOAT,
        .size = {size.x, size.y, 1},
        .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::SHADER_READ_ONLY,
        .debug_name = "col_image",
    });
    final_image = device.create_image({
        .format = daxa::Format::R32G32B32A32_SFLOAT,
        .size = {size.x, size.y, 1},
        .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::SHADER_READ_ONLY | daxa::ImageUsageFlagBits::TRANSFER_SRC,
        .debug_name = "final_image",
    });
}

void RenderImages::destroy(daxa::Device &device) const {
    device.destroy_image(pos_image);
    device.destroy_image(col_image);
    device.destroy_image(final_image);
}

void VoxelChunks::create(daxa::Device &device, u32 log2_chunks_per_axis) {
    auto chunk_n = (1u << log2_chunks_per_axis);
    chunk_n = chunk_n * chunk_n * chunk_n;
    buffer = device.create_buffer({
        .size = static_cast<u32>(sizeof(VoxelChunk)) * chunk_n,
        .debug_name = "voxel_chunks_buffer",
    });
}

void VoxelChunks::destroy(daxa::Device &device) const {
    if (!buffer.is_empty()) {
        device.destroy_buffer(buffer);
    }
}

void GpuHeap::create(daxa::Device &device, u32 size) {
    buffer = device.create_buffer({
        .size = size,
        .debug_name = "gpu_heap_buffer",
    });
}

void GpuHeap::destroy(daxa::Device &device) const {
    if (!buffer.is_empty()) {
        device.destroy_buffer(buffer);
    }
}

void GpuResources::create(daxa::Device &device) {
    render_images.create(device);
    settings_buffer = device.create_buffer({
        .size = sizeof(GpuSettings),
        .debug_name = "settings_buffer",
    });
    input_buffer = device.create_buffer({
        .size = sizeof(GpuInput),
        .debug_name = "input_buffer",
    });
    output_buffer = device.create_buffer({
        .size = sizeof(GpuOutput) * (FRAMES_IN_FLIGHT + 1),
        .debug_name = "output_buffer",
    });
    staging_output_buffer = device.create_buffer({
        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
        .size = sizeof(GpuOutput) * (FRAMES_IN_FLIGHT + 1),
        .debug_name = "staging_output_buffer",
    });
    globals_buffer = device.create_buffer({
        .size = sizeof(GpuGlobals),
        .debug_name = "globals_buffer",
    });
    temp_voxel_chunks_buffer = device.create_buffer({
        .size = sizeof(TempVoxelChunk) * MAX_CHUNK_UPDATES,
        .debug_name = "temp_voxel_chunks_buffer",
    });
    allocator_state_buffer = device.create_buffer({
        .size = sizeof(GpuAllocatorState),
        .debug_name = "allocator_state_buffer",
    });
}

void GpuResources::destroy(daxa::Device &device) const {
    render_images.destroy(device);
    device.destroy_buffer(settings_buffer);
    device.destroy_buffer(input_buffer);
    device.destroy_buffer(output_buffer);
    device.destroy_buffer(staging_output_buffer);
    device.destroy_buffer(globals_buffer);
    device.destroy_buffer(temp_voxel_chunks_buffer);
    device.destroy_buffer(allocator_state_buffer);
    voxel_chunks.destroy(device);
    gpu_heap.destroy(device);
    if (!gvox_model_buffer.is_empty()) {
        device.destroy_buffer(gvox_model_buffer);
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
            .name = image_info.debug_name,
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
            .name = buffer_info.debug_name,
            .size = buffer_info.size,
        });
        result_size += buffer_info.size;
    };

    image_size(gpu_resources.render_images.pos_image);
    image_size(gpu_resources.render_images.col_image);
    image_size(gpu_resources.render_images.final_image);

    buffer_size(gpu_resources.settings_buffer);
    buffer_size(gpu_resources.input_buffer);
    buffer_size(gpu_resources.globals_buffer);
    buffer_size(gpu_resources.temp_voxel_chunks_buffer);
    buffer_size(gpu_resources.allocator_state_buffer);
    buffer_size(gpu_resources.voxel_chunks.buffer);
    buffer_size(gpu_resources.gpu_heap.buffer);
    buffer_size(gpu_resources.gvox_model_buffer);

    needs_vram_calc = false;
}

// static auto handle_pipeline_creation(AppUi &ui, auto compile_result) {
// }

VoxelApp::VoxelApp()
    : AppWindow(APPNAME),
      daxa_ctx{daxa::create_context({.enable_validation = false})},
      device{daxa_ctx.create_device({
          // .enable_buffer_device_address_capture_replay = false,
          .debug_name = APPNAME_PREFIX("device"),
      })},
      swapchain{device.create_swapchain({
          .native_window = AppWindow::get_native_handle(),
          .native_window_platform = AppWindow::get_native_platform(),
          .present_mode = daxa::PresentMode::IMMEDIATE,
          .image_usage = daxa::ImageUsageFlagBits::TRANSFER_DST,
          .max_allowed_frames_in_flight = FRAMES_IN_FLIGHT,
          .debug_name = APPNAME_PREFIX("swapchain"),
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
          .debug_name = APPNAME_PREFIX("pipeline_manager"),
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
      startup_task{
          .pipeline{[this]() -> std::shared_ptr<daxa::ComputePipeline> {
              auto compile_result = main_pipeline_manager.add_compute_pipeline({
                  .shader_info = {.source = daxa::ShaderFile{"startup.comp.glsl"}},
                  .push_constant_size = sizeof(StartupComputePush),
                  .debug_name = APPNAME_PREFIX("startup_task"),
              });
              if (compile_result.is_err()) {
                  ui.console.add_log("%s", compile_result.to_string().c_str());
                  return {};
              }
              return compile_result.value();
          }()},
      },
      perframe_task{
          .pipeline{[this]() -> std::shared_ptr<daxa::ComputePipeline> {
              auto compile_result = main_pipeline_manager.add_compute_pipeline({
                  .shader_info = {.source = daxa::ShaderFile{"perframe.comp.glsl"}},
                  .push_constant_size = sizeof(PerframeComputePush),
                  .debug_name = APPNAME_PREFIX("perframe_task"),
              });
              if (compile_result.is_err()) {
                  ui.console.add_log("%s", compile_result.to_string().c_str());
                  return {};
              }
              return compile_result.value();
          }()},
      },
      per_chunk_task{
          .pipeline{[this]() -> std::shared_ptr<daxa::ComputePipeline> {
              auto compile_result = main_pipeline_manager.add_compute_pipeline({
                  .shader_info = {.source = daxa::ShaderFile{"per_chunk.comp.glsl"}},
                  .push_constant_size = sizeof(PerChunkComputePush),
                  .debug_name = APPNAME_PREFIX("per_chunk_task"),
              });
              if (compile_result.is_err()) {
                  ui.console.add_log("%s", compile_result.to_string().c_str());
                  return {};
              }
              return compile_result.value();
          }()},
      },
      chunk_edit_task{
          .pipeline{[this]() -> std::shared_ptr<daxa::ComputePipeline> {
              auto compile_result = main_pipeline_manager.add_compute_pipeline({
                  .shader_info = {.source = daxa::ShaderFile{"chunk_edit.comp.glsl"}},
                  .push_constant_size = sizeof(ChunkEditComputePush),
                  .debug_name = APPNAME_PREFIX("chunk_edit_task"),
              });
              if (compile_result.is_err()) {
                  ui.console.add_log("%s", compile_result.to_string().c_str());
                  return {};
              }
              return compile_result.value();
          }()},
      },
      chunk_opt_x2x4_task{
          .pipeline{[this]() -> std::shared_ptr<daxa::ComputePipeline> {
              auto compile_result = main_pipeline_manager.add_compute_pipeline({
                  .shader_info = {
                      .source = daxa::ShaderFile{"chunk_opt.comp.glsl"},
                      .compile_options = {.defines = {{"SUBCHUNK_X2X4", "1"}}},
                  },
                  .push_constant_size = sizeof(ChunkOptComputePush),
                  .debug_name = APPNAME_PREFIX("chunk_opt_x2x4"),
              });
              if (compile_result.is_err()) {
                  ui.console.add_log("%s", compile_result.to_string().c_str());
                  return {};
              }
              return compile_result.value();
          }()},
      },
      chunk_opt_x8up_task{
          .pipeline{[this]() -> std::shared_ptr<daxa::ComputePipeline> {
              auto compile_result = main_pipeline_manager.add_compute_pipeline({
                  .shader_info = {
                      .source = daxa::ShaderFile{"chunk_opt.comp.glsl"},
                      .compile_options = {.defines = {{"SUBCHUNK_X8UP", "1"}}},
                  },
                  .push_constant_size = sizeof(ChunkOptComputePush),
                  .debug_name = APPNAME_PREFIX("chunk_opt_x8up"),
              });
              if (compile_result.is_err()) {
                  ui.console.add_log("%s", compile_result.to_string().c_str());
                  return {};
              }
              return compile_result.value();
          }()},
      },
      chunk_alloc_task{
          .pipeline{[this]() -> std::shared_ptr<daxa::ComputePipeline> {
              auto compile_result = main_pipeline_manager.add_compute_pipeline({
                  .shader_info = {.source = daxa::ShaderFile{"chunk_alloc.comp.glsl"}},
                  .push_constant_size = sizeof(ChunkAllocComputePush),
                  .debug_name = APPNAME_PREFIX("chunk_alloc_task"),
              });
              if (compile_result.is_err()) {
                  ui.console.add_log("%s", compile_result.to_string().c_str());
                  return {};
              }
              return compile_result.value();
          }()},
      },
      trace_primary_task{
          .pipeline{[this]() -> std::shared_ptr<daxa::ComputePipeline> {
              auto compile_result = main_pipeline_manager.add_compute_pipeline({
                  .shader_info = {.source = daxa::ShaderFile{"trace_primary.comp.glsl"}},
                  .push_constant_size = sizeof(TracePrimaryComputePush),
                  .debug_name = APPNAME_PREFIX("trace_primary_task"),
              });
              if (compile_result.is_err()) {
                  ui.console.add_log("%s", compile_result.to_string().c_str());
                  return {};
              }
              return compile_result.value();
          }()},
      },
      color_scene_task{
          .pipeline{[this]() -> std::shared_ptr<daxa::ComputePipeline> {
              auto compile_result = main_pipeline_manager.add_compute_pipeline({
                  .shader_info = {.source = daxa::ShaderFile{"color_scene.comp.glsl"}},
                  .push_constant_size = sizeof(ColorSceneComputePush),
                  .debug_name = APPNAME_PREFIX("color_scene_task"),
              });
              if (compile_result.is_err()) {
                  ui.console.add_log("%s", compile_result.to_string().c_str());
                  return {};
              }
              return compile_result.value();
          }()},
      },
      postprocessing_task{
          .pipeline{[this]() -> std::shared_ptr<daxa::ComputePipeline> {
              auto compile_result = main_pipeline_manager.add_compute_pipeline({
                  .shader_info = {.source = daxa::ShaderFile{"postprocessing.comp.glsl"}},
                  .push_constant_size = sizeof(PostprocessingComputePush),
                  .debug_name = APPNAME_PREFIX("postprocessing_task"),
              });
              if (compile_result.is_err()) {
                  ui.console.add_log("%s", compile_result.to_string().c_str());
                  return {};
              }
              return compile_result.value();
          }()},
      },
      main_task_list{[this]() {
          gpu_resources.create(device);
          gpu_resources.voxel_chunks.create(device, ui.settings.log2_chunks_per_axis);
          gpu_resources.gpu_heap.create(device, ui.settings.gpu_heap_size);
          return record_main_task_list();
      }()} {
    gvox_ctx = gvox_create_context();
}

VoxelApp::~VoxelApp() {
    gvox_destroy_context(gvox_ctx);
    device.wait_idle();
    device.collect_garbage();
    gpu_resources.destroy(device);
}

auto VoxelApp::update() -> bool {
    glfwPollEvents();
    if (glfwWindowShouldClose(AppWindow::glfw_window_ptr) != 0) {
        return true;
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

    return false;
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
        // ui.console.add_log("(pulling file into memory: %fs)", cpu_time_used);
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
            GVOX_CHANNEL_BIT_COLOR | GVOX_CHANNEL_BIT_MATERIAL_ID);
        // time_t end = clock();
        // double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        // ui.console.add_log("%fs, new size: %llu bytes", cpu_time_used, result.size);
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
        if (reload_result.v.value_or(true)) {
            if (reload_result.is_err()) {
                ui.console.add_log("%s", reload_result.to_string().c_str());
            }
        }
    }

    ui.update(gpu_input.delta_time);

    main_task_list.remove_runtime_image(task_swapchain_image, swapchain_image);
    swapchain_image = swapchain.acquire_next_image();
    main_task_list.add_runtime_image(task_swapchain_image, swapchain_image);
    if (swapchain_image.is_empty()) {
        return;
    }

    if (ui.should_run_startup) {
        run_startup();
    }

    if (ui.should_upload_settings) {
        upload_settings();
    }

    if (ui.should_recreate_voxel_buffers) {
        recreate_voxel_chunks();
    }

    if (ui.should_upload_gvox_model) {
        if (!model_is_loading) {
            gvox_model_data_future = std::async(std::launch::async, &VoxelApp::load_gvox_data, this);
            model_is_loading = true;
        }
        if (model_is_loading && gvox_model_data_future.wait_for(0.01s) == std::future_status::ready) {
            upload_model();
            model_is_loading = false;
        }
    }

    if (needs_vram_calc) {
        calc_vram_usage();
    }

    main_task_list.execute();
    gpu_input.resize_factor = 1.0f;

    gpu_input.mouse.pos_delta = {0.0f, 0.0f};
    gpu_input.mouse.scroll_delta = {0.0f, 0.0f};

    ui.debug_gpu_heap_usage = gpu_output.heap_size;
    ui.debug_player_pos = gpu_output.player_pos;
}

void VoxelApp::on_mouse_move(f32 x, f32 y) {
    f32vec2 const center = {static_cast<f32>(window_size.x / 2), static_cast<f32>(window_size.y / 2)};
    gpu_input.mouse.pos = f32vec2{x, y};
    auto offset = gpu_input.mouse.pos - center;
    gpu_input.mouse.pos = gpu_input.mouse.pos * f32vec2{static_cast<f32>(gpu_resources.render_images.size.x), static_cast<f32>(gpu_resources.render_images.size.y)} / f32vec2{static_cast<f32>(window_size.x), static_cast<f32>(window_size.y)};
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
    }

    if (key_id == GLFW_KEY_R && action == GLFW_PRESS) {
        if (glfwGetKey(glfw_window_ptr, GLFW_KEY_LEFT_CONTROL) != GLFW_RELEASE) {
            ui.should_run_startup = true;
            start = Clock::now();
        }
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

void VoxelApp::recreate_render_images() {
    main_task_list.remove_runtime_image(task_render_pos_image, gpu_resources.render_images.pos_image);
    main_task_list.remove_runtime_image(task_render_col_image, gpu_resources.render_images.col_image);
    main_task_list.remove_runtime_image(task_render_final_image, gpu_resources.render_images.final_image);
    device.wait_idle();
    gpu_resources.render_images.destroy(device);
    gpu_resources.render_images.create(device);
    main_task_list.add_runtime_image(task_render_pos_image, gpu_resources.render_images.pos_image);
    main_task_list.add_runtime_image(task_render_col_image, gpu_resources.render_images.col_image);
    main_task_list.add_runtime_image(task_render_final_image, gpu_resources.render_images.final_image);
    needs_vram_calc = true;
}

void VoxelApp::recreate_voxel_chunks() {
    main_task_list.remove_runtime_buffer(task_voxel_chunks_buffer, gpu_resources.voxel_chunks.buffer);
    device.wait_idle();
    gpu_resources.voxel_chunks.destroy(device);
    gpu_resources.voxel_chunks.create(device, ui.settings.log2_chunks_per_axis);
    main_task_list.add_runtime_buffer(task_voxel_chunks_buffer, gpu_resources.voxel_chunks.buffer);
    ui.should_recreate_voxel_buffers = false;
    needs_vram_calc = true;
}

void VoxelApp::run_startup() {
    auto temp_task_list = daxa::TaskList({
        .device = device,
        .use_split_barriers = false,
        .debug_name = APPNAME_PREFIX("temp_task_list"),
    });

    auto temp_task_globals_buffer = temp_task_list.create_task_buffer({.debug_name = APPNAME_PREFIX("temp_task_globals_buffer")});
    temp_task_list.add_runtime_buffer(temp_task_globals_buffer, gpu_resources.globals_buffer);

    auto temp_task_voxel_chunks_buffer = temp_task_list.create_task_buffer({.debug_name = APPNAME_PREFIX("temp_task_voxel_chunks_buffer")});
    temp_task_list.add_runtime_buffer(temp_task_voxel_chunks_buffer, gpu_resources.voxel_chunks.buffer);

    // StartupTask
    temp_task_list.add_task({
        .used_buffers = {
            {temp_task_globals_buffer, daxa::TaskBufferAccess::HOST_TRANSFER_WRITE},
        },
        .task = [&temp_task_globals_buffer](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            cmd_list.clear_buffer({
                .buffer = task_runtime.get_buffers(temp_task_globals_buffer)[0],
                .offset = 0,
                .size = sizeof(GpuGlobals),
                .clear_value = 0,
            });
        },
        .debug_name = APPNAME_PREFIX("StartupTask (Globals Clear)"),
    });
    temp_task_list.add_task({
        .used_buffers = {
            {temp_task_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {temp_task_voxel_chunks_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
        },
        .task = [this, &temp_task_globals_buffer, &temp_task_voxel_chunks_buffer](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            startup_task.record(
                cmd_list,
                device.get_device_address(task_runtime.get_buffers(temp_task_globals_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(temp_task_voxel_chunks_buffer)[0]));
        },
        .debug_name = APPNAME_PREFIX("StartupTask"),
    });
    auto temp_submit_info = daxa::CommandSubmitInfo{};
    temp_task_list.submit(&temp_submit_info);
    temp_task_list.complete();
    temp_task_list.execute();

    ui.should_run_startup = false;
}

void VoxelApp::upload_settings() {
    auto temp_task_list = daxa::TaskList({
        .device = device,
        .use_split_barriers = false,
        .debug_name = APPNAME_PREFIX("temp_task_list"),
    });

    auto temp_task_settings_buffer = temp_task_list.create_task_buffer({.debug_name = APPNAME_PREFIX("temp_task_settings_buffer")});
    temp_task_list.add_runtime_buffer(temp_task_settings_buffer, gpu_resources.settings_buffer);

    temp_task_list.add_task({
        .used_buffers = {
            {temp_task_settings_buffer, daxa::TaskBufferAccess::HOST_TRANSFER_WRITE},
        },
        .task = [this, &temp_task_settings_buffer](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            auto staging_settings_buffer = device.create_buffer({
                .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .size = sizeof(GpuSettings),
                .debug_name = APPNAME_PREFIX("staging_settings_buffer"),
            });
            cmd_list.destroy_buffer_deferred(staging_settings_buffer);
            auto *buffer_ptr = device.get_host_address_as<GpuSettings>(staging_settings_buffer);
            *buffer_ptr = {
                .fov = ui.settings.camera_fov * (std::numbers::pi_v<f32> / 180.0f),
                .sensitivity = ui.settings.mouse_sensitivity,
                .log2_chunks_per_axis = ui.settings.log2_chunks_per_axis,
                .gpu_heap_size = ui.settings.gpu_heap_size,
            };
            cmd_list.copy_buffer_to_buffer({
                .src_buffer = staging_settings_buffer,
                .dst_buffer = task_runtime.get_buffers(temp_task_settings_buffer)[0],
                .size = sizeof(GpuSettings),
            });
        },
        .debug_name = APPNAME_PREFIX("StartupTask (Globals Clear)"),
    });
    temp_task_list.add_task({
        .used_buffers = {
            {temp_task_settings_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
        },
        .task = [](daxa::TaskRuntime) {},
        .debug_name = APPNAME_PREFIX("upload_settings_dummy_task"),
    });
    auto temp_submit_info = daxa::CommandSubmitInfo{};
    temp_task_list.submit(&temp_submit_info);
    temp_task_list.complete();
    temp_task_list.execute();

    ui.should_upload_settings = false;
}

void VoxelApp::upload_model() {
    auto temp_task_list = daxa::TaskList({
        .device = device,
        .use_split_barriers = false,
        .debug_name = APPNAME_PREFIX("temp_task_list"),
    });
    auto temp_task_gvox_model_buffer = temp_task_list.create_task_buffer({.debug_name = APPNAME_PREFIX("temp_task_gvox_model_buffer")});
    temp_task_list.add_task({
        .used_buffers = {
            {temp_task_gvox_model_buffer, daxa::TaskBufferAccess::HOST_TRANSFER_WRITE},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            gvox_model_data_future.wait();
            auto gvox_model_data = gvox_model_data_future.get();
            if (!gpu_resources.gvox_model_buffer.is_empty()) {
                main_task_list.remove_runtime_buffer(task_gvox_model_buffer, gpu_resources.gvox_model_buffer);
                cmd_list.destroy_buffer_deferred(gpu_resources.gvox_model_buffer);
            }
            gpu_resources.gvox_model_buffer = device.create_buffer({
                .size = static_cast<u32>(gvox_model_data.size),
                .debug_name = "gvox_model_buffer",
            });
            main_task_list.add_runtime_buffer(task_gvox_model_buffer, gpu_resources.gvox_model_buffer);
            cmd_list.pipeline_barrier({
                .waiting_pipeline_access = daxa::AccessConsts::TRANSFER_WRITE,
            });
            auto staging_gvox_model_buffer = device.create_buffer({
                .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .size = static_cast<u32>(gvox_model_data.size),
                .debug_name = APPNAME_PREFIX("staging_gvox_model_buffer"),
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
        },
        .debug_name = APPNAME_PREFIX("upload_model"),
    });
    temp_task_list.add_task({
        .used_buffers = {
            {temp_task_gvox_model_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
        },
        .task = [](daxa::TaskRuntime) {},
        .debug_name = APPNAME_PREFIX("upload_model_dummy_task"),
    });
    auto temp_submit_info = daxa::CommandSubmitInfo{};
    temp_task_list.submit(&temp_submit_info);
    temp_task_list.complete();
    temp_task_list.execute();

    ui.should_upload_gvox_model = false;
    has_model = true;
    needs_vram_calc = true;
}

auto VoxelApp::record_main_task_list() -> daxa::TaskList {
    daxa::TaskList result_task_list = daxa::TaskList({
        .device = device,
        .use_split_barriers = false,
        .swapchain = swapchain,
        .debug_name = APPNAME_PREFIX("main_task_list"),
    });

    task_settings_buffer = result_task_list.create_task_buffer({
        .initial_access = daxa::AccessConsts::COMPUTE_SHADER_READ,
        .debug_name = APPNAME_PREFIX("task_settings_buffer"),
    });
    result_task_list.add_runtime_buffer(task_settings_buffer, gpu_resources.settings_buffer);

    task_input_buffer = result_task_list.create_task_buffer({
        .debug_name = APPNAME_PREFIX("task_input_buffer"),
    });
    result_task_list.add_runtime_buffer(task_input_buffer, gpu_resources.input_buffer);

    task_output_buffer = result_task_list.create_task_buffer({
        .debug_name = APPNAME_PREFIX("task_output_buffer"),
    });
    result_task_list.add_runtime_buffer(task_output_buffer, gpu_resources.output_buffer);

    task_staging_output_buffer = result_task_list.create_task_buffer({
        .debug_name = APPNAME_PREFIX("task_staging_output_buffer"),
    });
    result_task_list.add_runtime_buffer(task_staging_output_buffer, gpu_resources.staging_output_buffer);

    task_globals_buffer = result_task_list.create_task_buffer({
        .initial_access = daxa::AccessConsts::COMPUTE_SHADER_READ_WRITE,
        .debug_name = APPNAME_PREFIX("task_globals_buffer"),
    });
    result_task_list.add_runtime_buffer(task_globals_buffer, gpu_resources.globals_buffer);

    task_temp_voxel_chunks_buffer = result_task_list.create_task_buffer({
        .debug_name = APPNAME_PREFIX("task_temp_voxel_chunks_buffer"),
    });
    result_task_list.add_runtime_buffer(task_temp_voxel_chunks_buffer, gpu_resources.temp_voxel_chunks_buffer);

    task_allocator_state_buffer = result_task_list.create_task_buffer({
        .debug_name = APPNAME_PREFIX("task_allocator_state_buffer"),
    });
    result_task_list.add_runtime_buffer(task_allocator_state_buffer, gpu_resources.allocator_state_buffer);

    task_voxel_chunks_buffer = result_task_list.create_task_buffer({
        .debug_name = APPNAME_PREFIX("task_voxel_chunks_buffer"),
    });
    result_task_list.add_runtime_buffer(task_voxel_chunks_buffer, gpu_resources.voxel_chunks.buffer);

    task_gpu_heap_buffer = result_task_list.create_task_buffer({
        .debug_name = APPNAME_PREFIX("task_gpu_heap_buffer"),
    });
    result_task_list.add_runtime_buffer(task_gpu_heap_buffer, gpu_resources.gpu_heap.buffer);

    task_gvox_model_buffer = result_task_list.create_task_buffer({
        .initial_access = daxa::AccessConsts::COMPUTE_SHADER_READ,
        .debug_name = APPNAME_PREFIX("task_gvox_model_buffer"),
    });

    task_swapchain_image = result_task_list.create_task_image({.swapchain_image = true, .debug_name = APPNAME_PREFIX("task_swapchain_image")});
    result_task_list.add_runtime_image(task_swapchain_image, swapchain_image);
    task_render_pos_image = result_task_list.create_task_image({.debug_name = APPNAME_PREFIX("task_render_pos_image")});
    result_task_list.add_runtime_image(task_render_pos_image, gpu_resources.render_images.pos_image);
    task_render_col_image = result_task_list.create_task_image({.debug_name = APPNAME_PREFIX("task_render_col_image")});
    result_task_list.add_runtime_image(task_render_col_image, gpu_resources.render_images.col_image);
    task_render_final_image = result_task_list.create_task_image({.debug_name = APPNAME_PREFIX("task_render_final_image")});
    result_task_list.add_runtime_image(task_render_final_image, gpu_resources.render_images.final_image);

    // GpuInputUploadTransferTask
    result_task_list.add_task({
        .used_buffers = {
            {task_input_buffer, daxa::TaskBufferAccess::TRANSFER_WRITE},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            gpu_input_upload_transfer_task.record(device, cmd_list, task_runtime.get_buffers(task_input_buffer)[0], gpu_input);
        },
        .debug_name = APPNAME_PREFIX("GpuInputUploadTransferTask"),
    });

    // PerframeTask
    result_task_list.add_task({
        .used_buffers = {
            {task_settings_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_output_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_voxel_chunks_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            u32 offset = gpu_input.frame_index % (FRAMES_IN_FLIGHT + 1);
            perframe_task.record(
                cmd_list,
                device.get_device_address(task_runtime.get_buffers(task_settings_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_input_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_output_buffer)[0]) + offset * sizeof(GpuOutput),
                device.get_device_address(task_runtime.get_buffers(task_globals_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_voxel_chunks_buffer)[0]),
                GpuAllocator{
                    .state = device.get_device_address(task_runtime.get_buffers(task_allocator_state_buffer)[0]),
                    .heap = device.get_device_address(task_runtime.get_buffers(task_gpu_heap_buffer)[0]),
                });
        },
        .debug_name = APPNAME_PREFIX("PerframeTask"),
    });

    // PerChunkTask
    result_task_list.add_task({
        .used_buffers = {
            {task_settings_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_voxel_chunks_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            auto chunk_nx = 1u << ui.settings.log2_chunks_per_axis;
            per_chunk_task.record(
                cmd_list,
                device.get_device_address(task_runtime.get_buffers(task_settings_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_input_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_globals_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_voxel_chunks_buffer)[0]),
                u32vec3{chunk_nx, chunk_nx, chunk_nx});
        },
        .debug_name = APPNAME_PREFIX("PerChunkTask"),
    });

    // ChunkEdit
    result_task_list.add_task({
        .used_buffers = {
            {task_settings_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_temp_voxel_chunks_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_WRITE_ONLY},
            {task_voxel_chunks_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            chunk_edit_task.record(
                cmd_list,
                device.get_device_address(task_runtime.get_buffers(task_settings_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_input_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_globals_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_temp_voxel_chunks_buffer)[0]),
                task_runtime.get_buffers(task_globals_buffer)[0]);
        },
        .debug_name = APPNAME_PREFIX("ChunkEdit"),
    });

    // ChunkOpt_x2x4
    result_task_list.add_task({
        .used_buffers = {
            {task_settings_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_temp_voxel_chunks_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_voxel_chunks_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            chunk_opt_x2x4_task.record(
                cmd_list,
                device.get_device_address(task_runtime.get_buffers(task_settings_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_input_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_globals_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_temp_voxel_chunks_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_voxel_chunks_buffer)[0]),
                task_runtime.get_buffers(task_globals_buffer)[0]);
        },
        .debug_name = APPNAME_PREFIX("ChunkOpt_x2x4"),
    });

    // ChunkOpt_x8up
    result_task_list.add_task({
        .used_buffers = {
            {task_settings_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_temp_voxel_chunks_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_voxel_chunks_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            chunk_opt_x8up_task.record(
                cmd_list,
                device.get_device_address(task_runtime.get_buffers(task_settings_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_input_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_globals_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_temp_voxel_chunks_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_voxel_chunks_buffer)[0]),
                task_runtime.get_buffers(task_globals_buffer)[0]);
        },
        .debug_name = APPNAME_PREFIX("ChunkOpt_x8up"),
    });

    // ChunkAlloc
    result_task_list.add_task({
        .used_buffers = {
            {task_settings_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_temp_voxel_chunks_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_voxel_chunks_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_allocator_state_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            {task_gpu_heap_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            chunk_alloc_task.record(
                cmd_list,
                device.get_device_address(task_runtime.get_buffers(task_settings_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_globals_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_temp_voxel_chunks_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_voxel_chunks_buffer)[0]),
                GpuAllocator{
                    .state = device.get_device_address(task_runtime.get_buffers(task_allocator_state_buffer)[0]),
                    .heap = device.get_device_address(task_runtime.get_buffers(task_gpu_heap_buffer)[0]),
                },
                task_runtime.get_buffers(task_globals_buffer)[0]);
        },
        .debug_name = APPNAME_PREFIX("ChunkOpt_x8up"),
    });

    // TracePrimaryTask
    result_task_list.add_task({
        .used_buffers = {
            {task_settings_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_gpu_heap_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_voxel_chunks_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
        },
        .used_images = {
            {task_render_pos_image, daxa::TaskImageAccess::COMPUTE_SHADER_WRITE_ONLY, daxa::ImageMipArraySlice{}},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            trace_primary_task.record(
                cmd_list,
                device.get_device_address(task_runtime.get_buffers(task_settings_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_input_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_globals_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_gpu_heap_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_voxel_chunks_buffer)[0]),
                task_runtime.get_images(task_render_pos_image)[0],
                gpu_resources.render_images.size);
        },
        .debug_name = APPNAME_PREFIX("TracePrimaryTask"),
    });

    // ColorSceneTask
    result_task_list.add_task({
        .used_buffers = {
            {task_settings_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_gpu_heap_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_voxel_chunks_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
        },
        .used_images = {
            {task_render_pos_image, daxa::TaskImageAccess::COMPUTE_SHADER_READ_ONLY, daxa::ImageMipArraySlice{}},
            {task_render_col_image, daxa::TaskImageAccess::COMPUTE_SHADER_WRITE_ONLY, daxa::ImageMipArraySlice{}},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            color_scene_task.record(
                cmd_list,
                device.get_device_address(task_runtime.get_buffers(task_settings_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_input_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_globals_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_gpu_heap_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_voxel_chunks_buffer)[0]),
                task_runtime.get_images(task_render_pos_image)[0],
                task_runtime.get_images(task_render_col_image)[0],
                gpu_resources.render_images.size);
        },
        .debug_name = APPNAME_PREFIX("ColorSceneTask"),
    });

    // PostprocessingTask
    result_task_list.add_task({
        .used_buffers = {
            {task_settings_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_gpu_heap_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            {task_voxel_chunks_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
        },
        .used_images = {
            {task_render_col_image, daxa::TaskImageAccess::COMPUTE_SHADER_READ_ONLY, daxa::ImageMipArraySlice{}},
            {task_render_final_image, daxa::TaskImageAccess::COMPUTE_SHADER_WRITE_ONLY, daxa::ImageMipArraySlice{}},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            postprocessing_task.record(
                cmd_list,
                device.get_device_address(task_runtime.get_buffers(task_settings_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_input_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_globals_buffer)[0]),
                device.get_device_address(task_runtime.get_buffers(task_allocator_state_buffer)[0]),
                task_runtime.get_images(task_render_col_image)[0],
                task_runtime.get_images(task_render_final_image)[0],
                gpu_resources.render_images.size);
        },
        .debug_name = APPNAME_PREFIX("PostprocessingTask"),
    });

    // GpuOutputDownloadTransferTask
    result_task_list.add_task({
        .used_buffers = {
            {task_output_buffer, daxa::TaskBufferAccess::TRANSFER_READ},
            {task_staging_output_buffer, daxa::TaskBufferAccess::HOST_TRANSFER_WRITE},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            ++gpu_input.frame_index;
            gpu_output_download_transfer_task.record(device, cmd_list, task_runtime.get_buffers(task_output_buffer)[0], gpu_resources.staging_output_buffer, gpu_output, gpu_input.frame_index);
        },
        .debug_name = APPNAME_PREFIX("GpuOutputDownloadTransferTask"),
    });

    // Blit (render to swapchain)
    result_task_list.add_task({
        .used_images = {
            {task_render_final_image, daxa::TaskImageAccess::TRANSFER_READ, daxa::ImageMipArraySlice{}},
            {task_swapchain_image, daxa::TaskImageAccess::TRANSFER_WRITE, daxa::ImageMipArraySlice{}},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            cmd_list.blit_image_to_image({
                .src_image = task_runtime.get_images(task_render_final_image)[0],
                .src_image_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
                .dst_image = task_runtime.get_images(task_swapchain_image)[0],
                .dst_image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                .src_slice = {.image_aspect = daxa::ImageAspectFlagBits::COLOR},
                .src_offsets = {{{0, 0, 0}, {static_cast<i32>(gpu_resources.render_images.size.x), static_cast<i32>(gpu_resources.render_images.size.y), 1}}},
                .dst_slice = {.image_aspect = daxa::ImageAspectFlagBits::COLOR},
                .dst_offsets = {{{0, 0, 0}, {static_cast<i32>(window_size.x), static_cast<i32>(window_size.y), 1}}},
            });
        },
        .debug_name = APPNAME_PREFIX("Blit (render to swapchain)"),
    });

    // ImGui draw
    result_task_list.add_task({
        .used_images = {
            {task_swapchain_image, daxa::TaskImageAccess::COLOR_ATTACHMENT, daxa::ImageMipArraySlice{}},
        },
        .task = [this](daxa::TaskRuntime task_runtime) {
            auto cmd_list = task_runtime.get_command_list();
            imgui_renderer.record_commands(ImGui::GetDrawData(), cmd_list, swapchain_image, window_size.x, window_size.y);
        },
        .debug_name = APPNAME_PREFIX("ImGui draw"),
    });

    result_task_list.submit(&submit_info);
    result_task_list.present({});
    result_task_list.complete();

    return result_task_list;
}
