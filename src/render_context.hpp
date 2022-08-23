#pragma once

#include <daxa/daxa.hpp>
using namespace daxa::types;

#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

using Clock = std::chrono::high_resolution_clock;

static constexpr u32 RENDER_SCL = 1;

#define APPNAME "Voxel Game"
#define APPNAME_PREFIX(x) ("[" APPNAME "] " x)

struct RenderContext {
    daxa::Context daxa_ctx = daxa::create_context({
        .enable_validation = false,
    });
    daxa::Device device = daxa_ctx.create_device({
        .use_scalar_layout = false,
        .debug_name = APPNAME_PREFIX("device"),
    });

    glm::uvec2 dim;

    daxa::PipelineCompiler pipeline_compiler = device.create_pipeline_compiler({
        .root_paths = {
            ".out/debug/vcpkg_installed/x64-windows/include",
            "shaders/pipelines",
            "shaders",
        },
        .shader_model_major = 6,
        .shader_model_minor = 0,
        .debug_name = APPNAME_PREFIX("pipeline_compiler"),
    });

    daxa::Swapchain swapchain;

    daxa::ImageId swapchain_image;
    daxa::ImageId render_col_images[2];
    daxa::ImageId render_pos_images[2];

    daxa::BinarySemaphore binary_semaphore = device.create_binary_semaphore({
        .debug_name = APPNAME_PREFIX("binary_semaphore"),
    });

    static inline constexpr u64 FRAMES_IN_FLIGHT = 1;
    daxa::TimelineSemaphore gpu_framecount_timeline_sema = device.create_timeline_semaphore(daxa::TimelineSemaphoreInfo{
        .initial_value = 0,
        .debug_name = APPNAME_PREFIX("gpu_framecount_timeline_sema"),
    });
    u64 cpu_framecount = FRAMES_IN_FLIGHT - 1;
    u32 frame_i = 0;

    daxa::ImageId create_color_image(glm::uvec2 dim) {
        return device.create_image({
            .format = daxa::Format::R32G32B32A32_SFLOAT,
            .aspect = daxa::ImageAspectFlagBits::COLOR,
            .size = {dim.x, dim.y, 1},
            .usage = daxa::ImageUsageFlagBits::COLOR_ATTACHMENT | daxa::ImageUsageFlagBits::SHADER_READ_ONLY | daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::TRANSFER_DST,
            .debug_name = "Render Image",
        });
    }

    daxa::ImageId create_depth_image(glm::uvec2 dim) {
        return device.create_image({
            .format = daxa::Format::D32_SFLOAT,
            .aspect = daxa::ImageAspectFlagBits::DEPTH,
            .size = {dim.x, dim.y, 1},
            .usage = daxa::ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT | daxa::ImageUsageFlagBits::SHADER_READ_ONLY,
            .debug_name = "Depth Image",
        });
    }

    void create_render_images() {
        render_col_images[0] = create_color_image(dim / RENDER_SCL),
        render_col_images[1] = create_color_image(dim / RENDER_SCL),
        render_pos_images[0] = create_color_image(dim / RENDER_SCL);
        render_pos_images[1] = create_color_image(dim / RENDER_SCL);
        // render_nrm_images[0] = create_color_image(dim / RENDER_SCL);
        // render_nrm_images[1] = create_color_image(dim / RENDER_SCL);
        // render_depth_image = create_depth_image(dim / RENDER_SCL);
    }
    void destroy_render_images() {
        device.destroy_image(render_col_images[0]);
        device.destroy_image(render_col_images[1]);
        device.destroy_image(render_pos_images[0]);
        device.destroy_image(render_pos_images[1]);
        // device.destroy_image(render_nrm_images[0]);
        // device.destroy_image(render_nrm_images[1]);
        // device.destroy_image(render_depth_image);
    }

    RenderContext(daxa::NativeWindowHandle native_window_handle, glm::uvec2 dim)
        : dim{dim},
          swapchain{device.create_swapchain({
              .native_window = native_window_handle,
              .width = dim.x,
              .height = dim.y,
              .surface_format_selector = [](daxa::Format format) {
                  switch (format) {
                  case daxa::Format::R8G8B8A8_UINT: return 100;
                  default: return daxa::default_format_score(format);
                  }
              },
              .present_mode = daxa::PresentMode::DO_NOT_WAIT_FOR_VBLANK,
              .image_usage = daxa::ImageUsageFlagBits::TRANSFER_DST,
              .debug_name = APPNAME_PREFIX("swapchain"),
          })} {
        swapchain_image = swapchain.acquire_next_image();
        create_render_images();
    }

    ~RenderContext() {
        wait_idle();
        destroy_render_images();
    }

    void wait_idle() {
        device.wait_idle();
    }

    void resize(glm::uvec2 dim) {
        wait_idle();
        this->dim = dim;
        swapchain.resize(dim.x, dim.y);
        destroy_render_images();
        create_render_images();
    }
};
