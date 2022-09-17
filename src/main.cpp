#include "base_app.hpp"
#include <map>

#define TEMP_BARRIER(cmd_list)                                     \
    cmd_list.pipeline_barrier({                                    \
        .awaited_pipeline_access = daxa::AccessConsts::READ_WRITE, \
        .waiting_pipeline_access = daxa::AccessConsts::READ_WRITE, \
    })

struct App : BaseApp<App> {
    // clang-format off
    daxa::ComputePipeline startup_comp_pipeline = pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"startup.comp.glsl"}},
        .push_constant_size = sizeof(StartupCompPush),
        .debug_name = APPNAME_PREFIX("startup_comp_pipeline"),
    }).value();
    daxa::ComputePipeline perframe_comp_pipeline = pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"perframe.comp.glsl"}},
        .push_constant_size = sizeof(PerframeCompPush),
        .debug_name = APPNAME_PREFIX("perframe_comp_pipeline"),
    }).value();
    daxa::ComputePipeline chunkgen_comp_pipeline = pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"chunkgen.comp.glsl"}},
        .push_constant_size = sizeof(ChunkgenCompPush),
        .debug_name = APPNAME_PREFIX("chunkgen_comp_pipeline"),
    }).value();
    daxa::ComputePipeline subchunk_x2x4_comp_pipeline = pipeline_compiler.create_compute_pipeline({
        .shader_info = {
            .source = daxa::ShaderFile{"chunk_opt.comp.glsl"}, 
            .compile_options = {
                .defines = {{.name = "SUBCHUNK_X2X4", .value = "1"}},
            },
        },
        .push_constant_size = sizeof(ChunkOptCompPush),
        .debug_name = APPNAME_PREFIX("subchunk_x2x4_comp_pipeline"),
    }).value();
    daxa::ComputePipeline subchunk_x8up_comp_pipeline = pipeline_compiler.create_compute_pipeline({
        .shader_info = {
            .source = daxa::ShaderFile{"chunk_opt.comp.glsl"}, 
            .compile_options = {
                .defines = {{.name = "SUBCHUNK_X8UP", .value = "1"}},
            },
        },
        .push_constant_size = sizeof(ChunkOptCompPush),
        .debug_name = APPNAME_PREFIX("subchunk_x8up_comp_pipeline"),
    }).value();
    daxa::ComputePipeline chunk_edit_comp_pipeline = pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"chunk_edit.comp.glsl"}},
        .push_constant_size = sizeof(ChunkEditCompPush),
        .debug_name = APPNAME_PREFIX("chunk_edit_comp_pipeline"),
    }).value();
    daxa::ComputePipeline draw_comp_pipeline = pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"draw.comp.glsl"}},
        .push_constant_size = sizeof(DrawCompPush),
        .debug_name = APPNAME_PREFIX("draw_comp_pipeline"),
    }).value();
    // clang-format on

    GpuInput gpu_input = default_gpu_input();
    auto default_gpu_input() -> GpuInput {
        return {
            .fov = 90.0f,
        };
    }

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

    BufferId gpu_globals_buffer = device.create_buffer({
        .size = sizeof(GpuGlobals),
        .debug_name = "gpu_globals_buffer",
    });
    daxa::TaskBufferId task_gpu_globals_buffer;

    std::map<i32, usize> mouse_bindings;
    std::map<i32, usize> key_bindings;

    bool battery_saving_mode = false;
    bool should_run_startup = true;
    bool paused = true;

    App() : BaseApp<App>() {
        key_bindings[GLFW_KEY_W] = GAME_KEY_W;
        key_bindings[GLFW_KEY_A] = GAME_KEY_A;
        key_bindings[GLFW_KEY_S] = GAME_KEY_S;
        key_bindings[GLFW_KEY_D] = GAME_KEY_D;
        key_bindings[GLFW_KEY_R] = GAME_KEY_R;
        key_bindings[GLFW_KEY_F] = GAME_KEY_F;
        key_bindings[GLFW_KEY_Q] = GAME_KEY_Q;
        key_bindings[GLFW_KEY_E] = GAME_KEY_E;
        key_bindings[GLFW_KEY_SPACE] = GAME_KEY_SPACE;
        key_bindings[GLFW_KEY_LEFT_CONTROL] = GAME_KEY_LEFT_CONTROL;
        key_bindings[GLFW_KEY_LEFT_SHIFT] = GAME_KEY_LEFT_SHIFT;
        key_bindings[GLFW_KEY_LEFT_ALT] = GAME_KEY_LEFT_ALT;
        key_bindings[GLFW_KEY_F5] = GAME_KEY_F5;

        mouse_bindings[GLFW_MOUSE_BUTTON_1] = GAME_MOUSE_BUTTON_1;
        mouse_bindings[GLFW_MOUSE_BUTTON_2] = GAME_MOUSE_BUTTON_2;
        mouse_bindings[GLFW_MOUSE_BUTTON_3] = GAME_MOUSE_BUTTON_3;
        mouse_bindings[GLFW_MOUSE_BUTTON_4] = GAME_MOUSE_BUTTON_4;
        mouse_bindings[GLFW_MOUSE_BUTTON_5] = GAME_MOUSE_BUTTON_5;
    }

    ~App() {
        device.wait_idle();
        device.collect_garbage();
        device.destroy_buffer(gpu_globals_buffer);
        device.destroy_buffer(gpu_input_buffer);
        device.destroy_buffer(staging_gpu_input_buffer);
        device.destroy_image(render_image);
    }

    void ui_update() {
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::ShowMetricsWindow();
        ImGui::Begin("Test");
        ImGui::Checkbox("Battery Saving Mode", &battery_saving_mode);
        ImGui::End();
        ImGui::Render();
    }

    void on_update() {
        auto now = Clock::now();
        gpu_input.time = std::chrono::duration<f32>(now - start).count();
        gpu_input.delta_time = std::chrono::duration<f32>(now - prev_time).count();
        prev_time = now;

        gpu_input.frame_dim = {size_x, size_y};

        if (battery_saving_mode) {
            std::this_thread::sleep_for(10ms);
        }

        reload_pipeline(draw_comp_pipeline);
        reload_pipeline(perframe_comp_pipeline);
        reload_pipeline(chunkgen_comp_pipeline);
        reload_pipeline(subchunk_x2x4_comp_pipeline);
        reload_pipeline(subchunk_x8up_comp_pipeline);
        if (reload_pipeline(startup_comp_pipeline))
            should_run_startup = true;

        ui_update();
        submit_task_list();

        gpu_input.mouse.pos_delta = {0.0f, 0.0f};
        gpu_input.mouse.scroll_delta = {0.0f, 0.0f};
    }

    void on_mouse_move(f32 x, f32 y) {
        if (!paused) {
            f32vec2 center = {static_cast<f32>(size_x / 2), static_cast<f32>(size_y / 2)};
            gpu_input.mouse.pos = f32vec2{x, y};
            auto offset = gpu_input.mouse.pos - center;
            gpu_input.mouse.pos_delta = gpu_input.mouse.pos_delta + offset;
            set_mouse_pos(center.x, center.y);
        }
    }
    void on_mouse_scroll(f32 dx, f32 dy) {
        if (!paused) {
            gpu_input.mouse.scroll_delta = gpu_input.mouse.scroll_delta + f32vec2{dx, dy};
        }
    }
    void on_mouse_button(i32 button_id, i32 action) {
        if (!paused && mouse_bindings.contains(button_id)) {
            auto index = mouse_bindings[button_id];
            gpu_input.mouse.buttons[index] = action;
        }
    }
    void on_key(i32 key_id, i32 action) {
        if (key_id == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            toggle_pause();
        if (key_id == GLFW_KEY_R && action == GLFW_PRESS)
            should_run_startup = true;

        if (!paused && key_bindings.contains(key_id)) {
            auto index = key_bindings[key_id];
            gpu_input.keyboard.keys[index] = action;
        }
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
    void toggle_pause() {
        set_mouse_capture(paused);
        if (paused)
            gpu_input = default_gpu_input();
        gpu_input.mouse.pos_delta = {0.0f, 0.0f};
        gpu_input.mouse.scroll_delta = {0.0f, 0.0f};
        paused = !paused;
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

        task_gpu_globals_buffer = new_task_list.create_task_buffer({
            .fetch_callback = [this]() { return gpu_globals_buffer; },
            .debug_name = "task_gpu_globals_buffer",
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
                TEMP_BARRIER(cmd_list);
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
                {task_gpu_globals_buffer, daxa::TaskBufferAccess::HOST_TRANSFER_WRITE},
            },
            .task = [this](daxa::TaskInterface interf) {
                if (should_run_startup) {
                    auto cmd_list = interf.get_command_list();
                    TEMP_BARRIER(cmd_list);
                    cmd_list.clear_buffer({
                        .buffer = gpu_globals_buffer,
                        .offset = 0,
                        .size = sizeof(GpuGlobals),
                        .clear_value = 0,
                    });
                }
            },
            .debug_name = "Startup (Globals Clear)",
        });
        new_task_list.add_task({
            .used_buffers = {
                {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            },
            .task = [this](daxa::TaskInterface interf) {
                if (should_run_startup) {
                    should_run_startup = false;
                    auto cmd_list = interf.get_command_list();
                    TEMP_BARRIER(cmd_list);
                    cmd_list.set_pipeline(startup_comp_pipeline);
                    auto push = StartupCompPush{
                        .gpu_globals = this->device.buffer_reference(gpu_globals_buffer),
                    };
                    cmd_list.push_constant(push);
                    cmd_list.dispatch(1, 1, 1);
                }
            },
            .debug_name = "Startup (Compute)",
        });

        new_task_list.add_task({
            .used_buffers = {
                {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
                {task_gpu_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            },
            .task = [this](daxa::TaskInterface interf) {
                auto cmd_list = interf.get_command_list();
                TEMP_BARRIER(cmd_list);
                cmd_list.set_pipeline(perframe_comp_pipeline);
                auto push = PerframeCompPush{
                    .gpu_globals = this->device.buffer_reference(gpu_globals_buffer),
                    .gpu_input = this->device.buffer_reference(gpu_input_buffer),
                };
                cmd_list.push_constant(push);
                cmd_list.dispatch(1, 1, 1);
            },
            .debug_name = "Perframe (Compute)",
        });

        new_task_list.add_task({
            .used_buffers = {
                {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            },
            .task = [this](daxa::TaskInterface interf) {
                auto cmd_list = interf.get_command_list();
                TEMP_BARRIER(cmd_list);
                cmd_list.set_pipeline(chunkgen_comp_pipeline);
                cmd_list.push_constant(ChunkgenCompPush{
                    .gpu_globals = device.buffer_reference(gpu_globals_buffer),
                });
                cmd_list.dispatch((CHUNK_SIZE + 7) / 8, (CHUNK_SIZE + 7) / 8, (CHUNK_SIZE + 7) / 8);
            },
            .debug_name = APPNAME_PREFIX("Chunkgen (Compute)"),
        });
        new_task_list.add_task({
            .used_buffers = {
                {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            },
            .task = [this](daxa::TaskInterface interf) {
                auto cmd_list = interf.get_command_list();
                TEMP_BARRIER(cmd_list);
                cmd_list.set_pipeline(chunk_edit_comp_pipeline);
                cmd_list.push_constant(ChunkEditCompPush{
                    .gpu_globals = device.buffer_reference(gpu_globals_buffer),
                });
                cmd_list.dispatch((CHUNK_SIZE + 7) / 8, (CHUNK_SIZE + 7) / 8, (CHUNK_SIZE + 7) / 8);
            },
            .debug_name = APPNAME_PREFIX("Chunk Edit (Compute)"),
        });
        new_task_list.add_task({
            .used_buffers = {
                {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            },
            .task = [this](daxa::TaskInterface interf) {
                auto cmd_list = interf.get_command_list();
                TEMP_BARRIER(cmd_list);
                cmd_list.set_pipeline(subchunk_x2x4_comp_pipeline);
                cmd_list.push_constant(ChunkOptCompPush{
                    .gpu_globals = device.buffer_reference(gpu_globals_buffer),
                });
                cmd_list.dispatch(1, 64, 1);
            },
            .debug_name = APPNAME_PREFIX("Subchunk x2x4 (Compute)"),
        });
        new_task_list.add_task({
            .used_buffers = {
                {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            },
            .task = [this](daxa::TaskInterface interf) {
                auto cmd_list = interf.get_command_list();
                TEMP_BARRIER(cmd_list);
                cmd_list.set_pipeline(subchunk_x8up_comp_pipeline);
                cmd_list.push_constant(ChunkOptCompPush{
                    .gpu_globals = device.buffer_reference(gpu_globals_buffer),
                });
                cmd_list.dispatch(1, 1, 1);
            },
            .debug_name = APPNAME_PREFIX("Subchunk x8up (Compute)"),
        });

        new_task_list.add_task({
            .used_buffers = {
                {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
                {task_gpu_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            },
            .used_images = {
                {task_render_image, daxa::TaskImageAccess::COMPUTE_SHADER_WRITE_ONLY},
            },
            .task = [this](daxa::TaskInterface interf) {
                auto cmd_list = interf.get_command_list();
                TEMP_BARRIER(cmd_list);
                cmd_list.set_pipeline(draw_comp_pipeline);
                cmd_list.push_constant(DrawCompPush{
                    .gpu_globals = device.buffer_reference(gpu_globals_buffer),
                    .gpu_input = device.buffer_reference(gpu_input_buffer),
                    .image_id = render_image.default_view(),
                });
                cmd_list.dispatch((size_x + 7) / 8, (size_y + 7) / 8);
            },
            .debug_name = APPNAME_PREFIX("Draw (Compute)"),
        });

        new_task_list.add_task({
            .used_buffers = {
                {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
            },
            .used_images = {
                {task_render_image, daxa::TaskImageAccess::TRANSFER_READ},
                {task_swapchain_image, daxa::TaskImageAccess::TRANSFER_WRITE},
            },
            .task = [this](daxa::TaskInterface interf) {
                auto cmd_list = interf.get_command_list();
                TEMP_BARRIER(cmd_list);
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
            .debug_name = APPNAME_PREFIX("Blit (render to swapchain)"),
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
