#include "base_app.hpp"

struct App : BaseApp<App> {
    // clang-format off
    daxa::ComputePipeline compute_pipeline = pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"compute.glsl"}},
        .push_constant_size = sizeof(ComputePush),
        .debug_name = APPNAME_PREFIX("compute_pipeline"),
    }).value();
    // clang-format on

    GpuInput gpu_input = {
        .view_origin = {0, 0},
        .mouse_pos = {0, 0},
        .zoom = 2.0f,
        .max_steps = 512,
    };

    daxa::BufferId gpu_input_buffer = device.create_buffer({
        .size = sizeof(GpuInput),
        .debug_name = APPNAME_PREFIX("gpu_input_buffer"),
    });
    daxa::TaskBufferId task_gpu_input_buffer;

    daxa::BufferId staging_gpu_input_buffer = device.create_buffer({
        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
        .size = sizeof(GpuInput),
        .debug_name = APPNAME_PREFIX("staging_gpu_input_buffer"),
    });
    daxa::TaskBufferId task_staging_gpu_input_buffer;

    daxa::ImageId render_image = device.create_image(daxa::ImageInfo{
        .format = daxa::Format::R8G8B8A8_UNORM,
        .size = {size_x, size_y, 1},
        .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::TRANSFER_SRC,
        .debug_name = APPNAME_PREFIX("render_image"),
    });
    daxa::TaskImageId task_render_image;

    App() : BaseApp<App>() {}

    ~App() {
        device.wait_idle();
        device.collect_garbage();
        device.destroy_buffer(gpu_input_buffer);
        device.destroy_buffer(staging_gpu_input_buffer);
        device.destroy_image(render_image);
    }

    void ui_update() {
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::Begin("Test");
        ImGui::DragFloat2("View Origin", reinterpret_cast<f32 *>(&gpu_input.view_origin), 0.001f, -2.0f, 2.0f, "%.7f");
        ImGui::DragFloat("Zoom", &gpu_input.zoom, 0.01f, 0.0f, 4.0f, "%.7f", ImGuiSliderFlags_Logarithmic);
        ImGui::DragInt("Max Steps", &gpu_input.max_steps, 1.0f, 1, 1024, "%d", ImGuiSliderFlags_Logarithmic);
        ImGui::End();
        ImGui::Render();
    }

    void on_update() {
        auto now = Clock::now();
        elapsed_s = std::chrono::duration<f32>(now - prev_time).count();
        prev_time = now;

        gpu_input.time = elapsed_s;
        gpu_input.frame_dim = {size_x, size_y};

        ui_update();
        reload_pipeline(compute_pipeline);
        submit_task_list();
    }

    void on_mouse_move(f32 x, f32 y) {
        gpu_input.mouse_pos = {x, y};
    }
    void on_mouse_scroll(f32 x, f32 y) {
        f32 mul = 0;
        if (y < 0)
            mul = pow(1.05f, abs(y));
        else if (y > 0)
            mul = 1.0f / pow(1.05f, abs(y));
        gpu_input.zoom *= mul;
    }
    void on_mouse_button(i32, i32) {
    }
    void on_key(i32, i32) {
    }
    void on_resize(u32 sx, u32 sy) {
        minimized = (sx == 0 || sy == 0);
        if (!minimized) {
            swapchain.resize();
            size_x = swapchain.info().width;
            size_y = swapchain.info().height;
            device.destroy_image(render_image);
            render_image = device.create_image({
                .format = daxa::Format::R8G8B8A8_UNORM,
                .size = {size_x, size_y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::TRANSFER_SRC,
            });
            on_update();
        }
    }

    void record_tasks(daxa::TaskList &new_task_list) {
        task_render_image = new_task_list.create_task_image({
            .fetch_callback = [this]() { return render_image; },
            .debug_name = APPNAME_PREFIX("task_render_image"),
        });

        task_gpu_input_buffer = new_task_list.create_task_buffer({
            .fetch_callback = [this]() { return gpu_input_buffer; },
            .debug_name = APPNAME_PREFIX("task_gpu_input_buffer"),
        });
        task_staging_gpu_input_buffer = new_task_list.create_task_buffer({
            .fetch_callback = [this]() { return staging_gpu_input_buffer; },
            .debug_name = APPNAME_PREFIX("task_staging_gpu_input_buffer"),
        });

        new_task_list.add_task({
            .used_buffers = {
                {task_staging_gpu_input_buffer, daxa::TaskBufferAccess::HOST_TRANSFER_WRITE},
            },
            .task = [this](daxa::TaskInterface /* interf */) {
                GpuInput *buffer_ptr = device.map_memory_as<GpuInput>(staging_gpu_input_buffer);
                *buffer_ptr = this->gpu_input;
                device.unmap_memory(staging_gpu_input_buffer);
            },
            .debug_name = APPNAME_PREFIX("Input MemMap"),
        });
        new_task_list.add_task({
            .used_buffers = {
                {task_gpu_input_buffer, daxa::TaskBufferAccess::TRANSFER_WRITE},
                {task_staging_gpu_input_buffer, daxa::TaskBufferAccess::TRANSFER_READ},
            },
            .task = [this](daxa::TaskInterface interf) {
                auto cmd_list = interf.get_command_list();
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
                {task_gpu_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            },
            .used_images = {
                {task_render_image, daxa::TaskImageAccess::COMPUTE_SHADER_WRITE_ONLY},
            },
            .task = [this](daxa::TaskInterface interf) {
                auto cmd_list = interf.get_command_list();
                cmd_list.set_pipeline(compute_pipeline);
                cmd_list.push_constant(ComputePush{
                    .image_id = render_image.default_view(),
                    .gpu_input = device.buffer_reference(gpu_input_buffer),
                });
                cmd_list.dispatch((size_x + 7) / 8, (size_y + 7) / 8);
            },
            .debug_name = APPNAME_PREFIX("Compute Task"),
        });

        new_task_list.add_task({
            .used_images = {
                {task_render_image, daxa::TaskImageAccess::TRANSFER_READ},
                {task_swapchain_image, daxa::TaskImageAccess::TRANSFER_WRITE},
            },
            .task = [this](daxa::TaskInterface interf) {
                auto cmd_list = interf.get_command_list();
                cmd_list.blit_image_to_image({
                    .src_image = render_image,
                    .src_image_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    .dst_image = swapchain_image,
                    .dst_image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                    .src_slice = {.image_aspect = daxa::ImageAspectFlagBits::COLOR},
                    .src_offsets = {{{0, 0, 0}, {static_cast<i32>(size_x), static_cast<i32>(size_y), 1}}},
                    .dst_slice = {.image_aspect = daxa::ImageAspectFlagBits::COLOR},
                    .dst_offsets = {{{0, 0, 0}, {static_cast<i32>(size_x), static_cast<i32>(size_y), 1}}},
                });
            },
            .debug_name = APPNAME_PREFIX("Blit Task (render to swapchain)"),
        });
    }
};

int main() {
    App app = {};
    while (true) {
        if (app.update())
            break;
    }
}
