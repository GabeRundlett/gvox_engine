#define APPNAME "Voxel Game"
#define APPNAME_PREFIX(x) ("[" APPNAME "] " x)

#include "daxa_common/common.hpp"

#include <thread>
#include <iostream>
#include <map>

#include "../shaders/shared.inl"

#define PI 3.14159265

float deg2rad(float d) {
    return d * PI / 180.0;
}

struct GpuGlobals {
    u8 data[1 << 30];
};

struct App : daxa_common::App<App> {
    // clang-format off
    daxa::ComputePipeline startup_pipeline = pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"startup.hlsl"}},
        .push_constant_size = sizeof(StartupPush),
        .debug_name = APPNAME_PREFIX("startup_pipeline"),
    }).value();
    daxa::ComputePipeline perframe_pipeline = pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"perframe.hlsl"}},
        .push_constant_size = sizeof(PerframePush),
        .debug_name = APPNAME_PREFIX("perframe_pipeline"),
    }).value();
    daxa::ComputePipeline chunkgen_pipeline = pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"chunkgen.hlsl"}},
        .push_constant_size = sizeof(ChunkgenPush),
        .debug_name = APPNAME_PREFIX("chunkgen_pipeline"),
    }).value();
    daxa::ComputePipeline subchunk_x2x4_pipeline = pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"subchunk_x2x4.hlsl"}},
        .push_constant_size = sizeof(SubchunkPush),
        .debug_name = APPNAME_PREFIX("subchunk_x2x4_pipeline"),
    }).value();
    daxa::ComputePipeline subchunk_x8up_pipeline = pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"subchunk_x8up.hlsl"}},
        .push_constant_size = sizeof(SubchunkPush),
        .debug_name = APPNAME_PREFIX("subchunk_x8up_pipeline"),
    }).value();
    daxa::ComputePipeline depth_prepass0_pipeline = pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"depth_prepass0.hlsl"}},
        .push_constant_size = sizeof(DepthPrepassPush),
        .debug_name = APPNAME_PREFIX("depth_prepass0_pipeline"),
    }).value();
    daxa::ComputePipeline depth_prepass1_pipeline = pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"depth_prepass1.hlsl"}},
        .push_constant_size = sizeof(DepthPrepassPush),
        .debug_name = APPNAME_PREFIX("depth_prepass1_pipeline"),
    }).value();
    daxa::ComputePipeline draw_pipeline = pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"draw.hlsl"}},
        .push_constant_size = sizeof(DrawPush),
        .debug_name = APPNAME_PREFIX("draw_pipeline"),
    }).value();
    // clang-format on

    GpuInput gpu_input = {
        .block_color = {1.0f, 0.1f, 0.1f},
    };
    daxa::BufferId gpu_input_buffer = device.create_buffer({
        .size = sizeof(GpuInput),
        .debug_name = "gpu_input_buffer",
    });
    daxa::BufferId staging_gpu_input_buffer = device.create_buffer({
        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
        .size = sizeof(GpuInput),
        .debug_name = "staging_gpu_input_buffer",
    });
    daxa::TaskBufferId task_gpu_input_buffer;
    daxa::TaskBufferId task_staging_gpu_input_buffer;
    GpuOutput gpu_output = {};
    daxa::BufferId gpu_output_buffer = device.create_buffer({
        .size = sizeof(GpuOutput),
        .debug_name = "gpu_output_buffer",
    });
    daxa::BufferId staging_gpu_output_buffer = device.create_buffer({
        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
        .size = sizeof(GpuOutput),
        .debug_name = "staging_gpu_output_buffer",
    });
    daxa::TaskBufferId task_gpu_output_buffer;
    daxa::TaskBufferId task_staging_gpu_output_buffer;

    daxa::BufferId gpu_globals_buffer = device.create_buffer({
        .size = sizeof(GpuGlobals),
        .debug_name = "gpu_globals_buffer",
    });
    daxa::BufferId staging_gpu_globals_buffer = device.create_buffer({
        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
        .size = sizeof(GpuGlobals),
        .debug_name = "staging_gpu_globals_buffer",
    });
    daxa::TaskBufferId task_gpu_globals_buffer;
    daxa::TaskBufferId task_staging_gpu_globals_buffer;

    std::map<i32, usize> mouse_bindings;
    std::map<i32, usize> key_bindings;

    Clock::time_point prev_place_time, prev_break_time;
    float place_speed = 0.2f, break_speed = 0.2f;
    bool limit_place_speed = true;
    bool limit_break_speed = true;
    bool should_break = false;
    bool should_place = false;
    bool started = false;
    bool enable_depth_prepass = false;
    bool should_run_startup = true;
    bool laptop_power_saving = false;

    App() : daxa_common::App<App>(APPNAME) {
        std::cout << device.properties().device_name << std::endl;

        key_bindings[GLFW_KEY_W] = GAME_KEY_W;
        key_bindings[GLFW_KEY_A] = GAME_KEY_A;
        key_bindings[GLFW_KEY_S] = GAME_KEY_S;
        key_bindings[GLFW_KEY_D] = GAME_KEY_D;
        key_bindings[GLFW_KEY_R] = GAME_KEY_R;
        key_bindings[GLFW_KEY_F] = GAME_KEY_F;
        key_bindings[GLFW_KEY_SPACE] = GAME_KEY_SPACE;
        key_bindings[GLFW_KEY_LEFT_CONTROL] = GAME_KEY_LEFT_CONTROL;
        key_bindings[GLFW_KEY_LEFT_SHIFT] = GAME_KEY_LEFT_SHIFT;
        key_bindings[GLFW_KEY_F5] = GAME_KEY_F5;

        mouse_bindings[GLFW_MOUSE_BUTTON_1] = GAME_MOUSE_BUTTON_1;
        mouse_bindings[GLFW_MOUSE_BUTTON_2] = GAME_MOUSE_BUTTON_2;
        mouse_bindings[GLFW_MOUSE_BUTTON_3] = GAME_MOUSE_BUTTON_3;
        mouse_bindings[GLFW_MOUSE_BUTTON_4] = GAME_MOUSE_BUTTON_4;
        mouse_bindings[GLFW_MOUSE_BUTTON_5] = GAME_MOUSE_BUTTON_5;

        fov = 90.0f;
    }

    ~App() {
        device.wait_idle();
        device.destroy_buffer(gpu_input_buffer);
        device.destroy_buffer(staging_gpu_input_buffer);
        device.destroy_buffer(gpu_output_buffer);
        device.destroy_buffer(staging_gpu_output_buffer);
        device.destroy_buffer(gpu_globals_buffer);
        device.destroy_buffer(staging_gpu_globals_buffer);
    }

    void run_startup() {
        should_run_startup = true;
    }

    void ui_settings() {
        ImGui::Begin("Settings");
        ImGui::Text("Game");
        ImGui::ColorEdit3("Block Color", reinterpret_cast<float *>(&gpu_input.block_color));
        ImGui::Text("Graphics");
        f32 new_scl = render_scl;
        ImGui::SliderFloat("Render Scl", &new_scl, 1.0f / static_cast<f32>(std::min(size_x, size_y)), 1.0f);
        ImGui::Checkbox("Enable FSR", &fsr_enabled);
        if (new_scl != render_scl) {
            render_scl = new_scl;
            destroy_render_images();
            create_render_images();
        }
        ImGui::SliderFloat("FOV", &gpu_input.fov, 1.0f, 179.9f);
        bool enable_shadows = (gpu_input.flags >> 0) & 0x1;
        ImGui::Checkbox("Shadows", &enable_shadows);
        gpu_input.flags = (gpu_input.flags & ~(1 << 0)) | (static_cast<u32>(enable_shadows) << 0);
        ImGui::Checkbox("Depth Prepass", &enable_depth_prepass);
        ImGui::Checkbox("Laptop Power Saving", &laptop_power_saving);
    }

    void ui_update() {
        ImGui::NewFrame();

        if (!started) {
            ui_settings();
            if (ImGui::Button("START")) {
                started = true;
            }
            ImGui::End();
        } else {
            if (paused) {
                ui_settings();
                if (ImGui::Button("STOP")) {
                    started = false;
                }
                ImGui::End();
            }
        }

        auto &io = ImGui::GetIO();

        ImGui::Begin("Debug Stats");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        ImGui::Text("Pos: %.4f, %.4f, %.4f", gpu_output.player_pos.x, gpu_output.player_pos.y, gpu_output.player_pos.z);
        ImGui::Text("Rot: %.4f, %.4f, %.4f", gpu_output.player_rot.x, gpu_output.player_rot.y, gpu_output.player_rot.z);
        ImGui::End();

        ImGui::Render();
    }

    void on_update() {
        gpu_input.fov = fov;
        gpu_input.time = time;
        gpu_input.delta_time = delta_time;
        gpu_input.render_size = u32vec2{render_size_x, render_size_y};
        gpu_input.jitter = (jitter - prev_jitter) * f32vec2{2.0f / static_cast<f32>(render_size_x), 2.0f / static_cast<f32>(render_size_y)};
        auto start_pipe0 = try_recreate_pipeline(startup_pipeline);
        auto chunk_pipe0 = try_recreate_pipeline(chunkgen_pipeline);
        auto chunk_pipe1 = try_recreate_pipeline(subchunk_x2x4_pipeline);
        auto chunk_pipe2 = try_recreate_pipeline(subchunk_x8up_pipeline);
        if (start_pipe0 || chunk_pipe0 || chunk_pipe1 || chunk_pipe2)
            run_startup();
        try_recreate_pipeline(perframe_pipeline);
        try_recreate_pipeline(depth_prepass0_pipeline);
        try_recreate_pipeline(depth_prepass1_pipeline);
        try_recreate_pipeline(draw_pipeline);
        ui_update();
        execute_loop_task_list();
        gpu_input.mouse.pos_delta = {0.0f, 0.0f};
        gpu_input.mouse.scroll_delta = {0.0f, 0.0f};
    }

    void on_mouse_move(f32 x, f32 y) {
        if (!paused) {
            f32vec2 center = {static_cast<f32>(size_x / 2), static_cast<f32>(size_y / 2)};
            auto offset = gpu_input.mouse.pos - center;
            set_mouse_pos(center.x, center.y);
            gpu_input.mouse.pos_delta = gpu_input.mouse.pos_delta + offset;
            gpu_input.mouse.pos = f32vec2{x, y};
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

            switch (button_id) {
            case GLFW_MOUSE_BUTTON_LEFT:
                should_break = action != GLFW_RELEASE;
                if (!should_break)
                    prev_break_time = start;
                break;
            case GLFW_MOUSE_BUTTON_RIGHT:
                should_place = action != GLFW_RELEASE;
                if (!should_place)
                    prev_place_time = start;
                break;
            default: break;
            }
        }
    }

    void on_key(i32 key_id, i32 action) {
        if (key_id == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            toggle_pause();
        if (key_id == GLFW_KEY_R && action == GLFW_PRESS)
            run_startup();

        if (!paused && key_bindings.contains(key_id)) {
            auto index = key_bindings[key_id];
            gpu_input.keyboard.keys[index] = action;
        }
    }

    void on_resize(u32 sx, u32 sy) {
        size_x = sx;
        size_y = sy;
        minimized = (sx == 0 || sy == 0);

        if (!minimized) {
            do_resize();
        }
    }

    void do_resize() {
        if (size_x < 1 || size_y < 1)
            return;

        swapchain.resize(size_x, size_y);
        destroy_render_images();
        create_render_images();
        on_update();
    }

    void record_loop_task_list(daxa::TaskList &new_task_list) {
        task_gpu_input_buffer = new_task_list.create_task_buffer({
            .fetch_callback = [this]() { return gpu_input_buffer; },
            .debug_name = "task_gpu_input_buffer",
        });
        task_staging_gpu_input_buffer = new_task_list.create_task_buffer({
            .fetch_callback = [this]() { return staging_gpu_input_buffer; },
            .debug_name = "task_staging_gpu_input_buffer",
        });
        task_gpu_output_buffer = new_task_list.create_task_buffer({
            .fetch_callback = [this]() { return gpu_output_buffer; },
            .debug_name = "task_gpu_output_buffer",
        });
        task_staging_gpu_output_buffer = new_task_list.create_task_buffer({
            .fetch_callback = [this]() { return staging_gpu_output_buffer; },
            .debug_name = "task_staging_gpu_output_buffer",
        });
        task_gpu_globals_buffer = new_task_list.create_task_buffer({
            .fetch_callback = [this]() { return gpu_globals_buffer; },
            .debug_name = "task_gpu_globals_buffer",
        });
        task_staging_gpu_globals_buffer = new_task_list.create_task_buffer({
            .fetch_callback = [this]() { return staging_gpu_globals_buffer; },
            .debug_name = "task_staging_gpu_globals_buffer",
        });

        new_task_list.add_task({
            .resources = {
                .buffers = {
                    {task_staging_gpu_input_buffer, daxa::TaskBufferAccess::HOST_TRANSFER_WRITE},
                },
            },
            .task = [this](daxa::TaskInterface /* interf */) {
                GpuInput *buffer_ptr = device.map_memory_as<GpuInput>(staging_gpu_input_buffer);
                *buffer_ptr = this->gpu_input;
                device.unmap_memory(staging_gpu_input_buffer);
            },
            .debug_name = "Gpu Input MemMap",
        });

        new_task_list.add_task({
            .resources = {
                .buffers = {
                    {task_gpu_input_buffer, daxa::TaskBufferAccess::TRANSFER_WRITE},
                    {task_staging_gpu_input_buffer, daxa::TaskBufferAccess::TRANSFER_READ},
                },
            },
            .task = [this](daxa::TaskInterface interf) {
                auto cmd_list = interf.get_command_list();
                cmd_list.copy_buffer_to_buffer({
                    .src_buffer = staging_gpu_input_buffer,
                    .dst_buffer = gpu_input_buffer,
                    .size = sizeof(GpuInput),
                });
            },
            .debug_name = "Gpu Input Transfer",
        });

        new_task_list.add_task({
            .resources = {
                .buffers = {
                    {task_staging_gpu_globals_buffer, daxa::TaskBufferAccess::HOST_TRANSFER_WRITE},
                },
            },
            .task = [this](daxa::TaskInterface interf) {
                if (should_run_startup) {
                    GpuGlobals *buffer_ptr = device.map_memory_as<GpuGlobals>(staging_gpu_globals_buffer);
                    memset(buffer_ptr, 0, sizeof(GpuGlobals));
                    device.unmap_memory(staging_gpu_globals_buffer);
                }
            },
            .debug_name = "Startup Task (clear globals buffer MemMap)",
        });
        new_task_list.add_task({
            .resources = {
                .buffers = {
                    {task_gpu_globals_buffer, daxa::TaskBufferAccess::TRANSFER_WRITE},
                    {task_staging_gpu_globals_buffer, daxa::TaskBufferAccess::TRANSFER_READ},
                },
            },
            .task = [this](daxa::TaskInterface interf) {
                if (should_run_startup) {
                    auto cmd_list = interf.get_command_list();
                    cmd_list.copy_buffer_to_buffer({
                        .src_buffer = staging_gpu_globals_buffer,
                        .dst_buffer = gpu_globals_buffer,
                        .size = sizeof(GpuGlobals),
                    });
                }
            },
            .debug_name = "Startup Task (clear globals buffer Transfer)",
        });

        new_task_list.add_task({
            .resources = {
                .buffers = {
                    {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
                },
            },
            .task = [this](daxa::TaskInterface interf) {
                if (should_run_startup) {
                    auto cmd_list = interf.get_command_list();
                    cmd_list.set_pipeline(startup_pipeline);
                    auto push = StartupPush{
                        .globals_buffer_id = gpu_globals_buffer,
                    };
                    cmd_list.push_constant(push);
                    cmd_list.dispatch(1, 1, 1);
                    should_run_startup = false;
                }
            },
            .debug_name = "Startup Task",
        });

        new_task_list.add_task({
            .resources = {
                .buffers = {
                    {task_gpu_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
                    {task_gpu_output_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
                    {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
                },
            },
            .task = [this](daxa::TaskInterface interf) {
                auto cmd_list = interf.get_command_list();
                cmd_list.set_pipeline(perframe_pipeline);
                auto push = PerframePush{
                    .globals_buffer_id = gpu_globals_buffer,
                    .input_buffer_id = gpu_input_buffer,
                    .output_buffer_id = gpu_output_buffer,
                };
                cmd_list.push_constant(push);
                cmd_list.dispatch(1, 1, 1);
            },
            .debug_name = "Perframe Task",
        });

        new_task_list.add_task({
            .resources = {
                .buffers = {
                    {task_gpu_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
                    {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
                },
            },
            .task = [this](daxa::TaskInterface interf) {
                if (started) {
                    auto cmd_list = interf.get_command_list();
                    cmd_list.set_pipeline(chunkgen_pipeline);
                    auto push = ChunkgenPush{
                        .globals_buffer_id = gpu_globals_buffer,
                        .input_buffer_id = gpu_input_buffer,
                    };
                    cmd_list.push_constant(push);
                    cmd_list.dispatch((CHUNK_SIZE + 7) / 8, (CHUNK_SIZE + 7) / 8, (CHUNK_SIZE + 7) / 8);
                }
            },
            .debug_name = "Chunkgen Task",
        });

        new_task_list.add_task({
            .resources = {
                .buffers = {
                    {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
                },
            },
            .task = [this](daxa::TaskInterface interf) {
                if (started) {
                    auto cmd_list = interf.get_command_list();
                    cmd_list.set_pipeline(subchunk_x2x4_pipeline);
                    auto push = SubchunkPush{
                        .globals_buffer_id = gpu_globals_buffer,
                        // .mode = 0,
                    };
                    cmd_list.push_constant(push);
                    cmd_list.dispatch(1, 64, 1);
                }
            },
            .debug_name = "Subchunk (x2x4) Task",
        });

        new_task_list.add_task({
            .resources = {
                .buffers = {
                    {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
                },
            },
            .task = [this](daxa::TaskInterface interf) {
                if (started) {
                    auto cmd_list = interf.get_command_list();
                    cmd_list.set_pipeline(subchunk_x8up_pipeline);
                    auto push = SubchunkPush{
                        .globals_buffer_id = gpu_globals_buffer,
                        // .mode = 0,
                    };
                    cmd_list.push_constant(push);
                    cmd_list.dispatch(1, 1, 1);
                }
            },
            .debug_name = "Subchunk (x8+) Task",
        });

        new_task_list.add_task({
            .resources = {
                .buffers = {
                    {task_gpu_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
                    {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
                },
                .images = {
                    {task_depth_image, daxa::TaskImageAccess::COMPUTE_SHADER_WRITE_ONLY},
                },
            },
            .task = [this](daxa::TaskInterface interf) {
                if (started && enable_depth_prepass) {
                    auto cmd_list = interf.get_command_list();
                    cmd_list.set_pipeline(depth_prepass0_pipeline);
                    auto push = DrawPush{
                        .globals_buffer_id = gpu_globals_buffer,
                        .input_buffer_id = gpu_input_buffer,
                        .render_depth_image_id = depth_image.default_view(),
                    };
                    cmd_list.push_constant(push);
                    cmd_list.dispatch((render_size_x + 15) / 16, (render_size_y + 15) / 16, 1);
                }
            },
            .debug_name = "Depth prepass 0 Task",
        });

        new_task_list.add_task({
            .resources = {
                .buffers = {
                    {task_gpu_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
                    {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
                },
                .images = {
                    {task_depth_image, daxa::TaskImageAccess::COMPUTE_SHADER_WRITE_ONLY},
                },
            },
            .task = [this](daxa::TaskInterface interf) {
                if (started && enable_depth_prepass) {
                    auto cmd_list = interf.get_command_list();
                    cmd_list.set_pipeline(depth_prepass1_pipeline);
                    auto push = DrawPush{
                        .globals_buffer_id = gpu_globals_buffer,
                        .input_buffer_id = gpu_input_buffer,
                        .render_depth_image_id = depth_image.default_view(),
                    };
                    cmd_list.push_constant(push);
                    cmd_list.dispatch((render_size_x + 15) / 16, (render_size_y + 15) / 16, 1);
                }
            },
            .debug_name = "Depth prepass 1 Task",
        });

        new_task_list.add_task({
            .resources = {
                .buffers = {
                    {task_gpu_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
                    {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
                },
                .images = {
                    {task_color_image, daxa::TaskImageAccess::COMPUTE_SHADER_WRITE_ONLY},
                    {task_motion_vectors_image, daxa::TaskImageAccess::COMPUTE_SHADER_WRITE_ONLY},
                    {task_depth_image, daxa::TaskImageAccess::COMPUTE_SHADER_READ_WRITE},
                },
            },
            .task = [this](daxa::TaskInterface interf) {
                if (started) {
                    auto cmd_list = interf.get_command_list();
                    cmd_list.set_pipeline(draw_pipeline);
                    auto push = DrawPush{
                        .globals_buffer_id = gpu_globals_buffer,
                        .input_buffer_id = gpu_input_buffer,
                        .render_color_image_id = color_image.default_view(),
                        .render_motion_image_id = motion_vectors_image.default_view(),
                        .render_depth_image_id = depth_image.default_view(),
                    };
                    cmd_list.push_constant(push);
                    cmd_list.dispatch((render_size_x + 7) / 8, (render_size_y + 7) / 8, 1);
                }
            },
            .debug_name = "Draw Task",
        });

        new_task_list.add_task({
            .resources = {
                .buffers = {
                    {task_gpu_output_buffer, daxa::TaskBufferAccess::TRANSFER_WRITE},
                    {task_staging_gpu_output_buffer, daxa::TaskBufferAccess::TRANSFER_READ},
                },
            },
            .task = [this](daxa::TaskInterface interf) {
                if (started) {
                    auto cmd_list = interf.get_command_list();
                    cmd_list.copy_buffer_to_buffer({
                        .src_buffer = gpu_output_buffer,
                        .dst_buffer = staging_gpu_output_buffer,
                        .size = sizeof(GpuOutput),
                    });
                }
            },
            .debug_name = "Gpu Output Transfer",
        });

        new_task_list.add_task({
            .resources = {
                .buffers = {
                    {task_staging_gpu_output_buffer, daxa::TaskBufferAccess::HOST_TRANSFER_WRITE},
                },
            },
            .task = [this](daxa::TaskInterface /* interf */) {
                if (started) {
                    GpuOutput *buffer_ptr = device.map_memory_as<GpuOutput>(staging_gpu_output_buffer);
                    this->gpu_output = *buffer_ptr;
                    device.unmap_memory(staging_gpu_output_buffer);
                }
            },
            .debug_name = "Gpu Output MemMap",
        });
    }
};

auto main() -> int {
    App app = {};
    while (true) {
        if (app.update())
            break;
    }
}
