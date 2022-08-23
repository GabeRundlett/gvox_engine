#define GLM_DEPTH_ZERO_TO_ONE

#include "render_context.hpp"
#include "window.hpp"

#include <iostream>
#include <chrono>
#include <thread>
using namespace std::literals;

#include "defines.inl"

#include <daxa/utils/imgui.hpp>
#include "imgui/imgui_impl_glfw.h"

// clang-format off
#define GAME_KEY_W                  0
#define GAME_KEY_A                  1
#define GAME_KEY_S                  2
#define GAME_KEY_D                  3
#define GAME_KEY_R                  4
#define GAME_KEY_F                  5
#define GAME_KEY_SPACE              6
#define GAME_KEY_LEFT_CONTROL       7
#define GAME_KEY_LEFT_SHIFT         8
#define GAME_KEY_F5                 9
#define GAME_KEY_LAST               GAME_KEY_F5
// clang-format on

namespace gpu {
    struct MouseInput {
        glm::vec2 pos = {};
        glm::vec2 pos_delta = {};
        glm::vec2 scroll_delta = {};
        u32 buttons[GLFW_MOUSE_BUTTON_LAST + 1] = {};
    };
    struct KeyboardInput {
        u32 keys[GAME_KEY_LAST + 1] = {};
    };

    struct Input {
        glm::ivec2 frame_dim = {};
        float time = {}, delta_time = {};
        float fov = {};
        glm::vec3 block_color = {1.0f, 0.05f, 0.08f};
        u32 flags = 0;
        u32 _pad0[3] = {50, 0, 0};
        MouseInput mouse = {};
        KeyboardInput keyboard = {};
    };
    struct Globals {
        u8 data[u64{1} << 30];
    };
    namespace push {
        struct Startup {
            daxa::BufferId globals_id;
        };
        struct Chunkgen {
            daxa::BufferId globals_id;
            daxa::BufferId input_id;
        };
        struct Subchunk {
            daxa::BufferId globals_id;
            u32 mode;
        };
        struct Perframe {
            daxa::BufferId globals_id;
            daxa::BufferId input_id;
        };
        struct Draw {
            daxa::BufferId globals_id;
            daxa::BufferId input_id;
            daxa::ImageViewId col_image_id_in, col_image_id_out;
            daxa::ImageViewId pos_image_id_in, pos_image_id_out;
        };
        struct DepthPrepass {
            daxa::BufferId globals_id;
            daxa::BufferId input_id;
            daxa::ImageViewId pos_image_id_in, pos_image_id_out;
            u32 scl;
        };
    } // namespace push
} // namespace gpu

struct Game {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point prev_frame_time, start_time;

    Clock::time_point prev_place_time, prev_break_time;
    float place_speed = 0.2f, break_speed = 0.2f;
    bool limit_place_speed = true;
    bool limit_break_speed = true;
    bool should_break = false;
    bool should_place = false;
    bool started = false;
    bool enable_depth_prepass = true;
    bool should_run_startup = true;

    Window window = {};
    RenderContext render_context{window.get_native_handle(), window.frame_dim};
    daxa::ImGuiRenderer imgui_renderer = create_imgui_renderer();
    auto create_imgui_renderer() -> daxa::ImGuiRenderer {
        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForVulkan(window.window_ptr, true);
        return daxa::ImGuiRenderer({
            .device = render_context.device,
            .pipeline_compiler = render_context.pipeline_compiler,
            .format = render_context.swapchain.get_format(),
        });
    }

    // clang-format off
    daxa::ComputePipeline startup_compute_pipeline = render_context.pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"startup.hlsl"}},
        .push_constant_size = sizeof(gpu::push::Startup),
        .debug_name = APPNAME_PREFIX("startup_compute_pipeline"),
    }).value();
    daxa::ComputePipeline perframe_compute_pipeline = render_context.pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"chunkgen.hlsl"}},
        .push_constant_size = sizeof(gpu::push::Perframe),
        .debug_name = APPNAME_PREFIX("startup_compute_pipeline"),
    }).value();
    daxa::ComputePipeline subchunk_x2x4_compute_pipeline = render_context.pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"subchunk_x2x4.hlsl"}},
        .push_constant_size = sizeof(gpu::push::Subchunk),
        .debug_name = APPNAME_PREFIX("subchunk_x2x4_compute_pipeline"),
    }).value();
    daxa::ComputePipeline subchunk_x8up_compute_pipeline = render_context.pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"subchunk_x8up.hlsl"}},
        .push_constant_size = sizeof(gpu::push::Subchunk),
        .debug_name = APPNAME_PREFIX("subchunk_x8up_compute_pipeline"),
    }).value();
    daxa::ComputePipeline chunkgen_compute_pipeline = render_context.pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"perframe.hlsl"}},
        .push_constant_size = sizeof(gpu::push::Chunkgen),
        .debug_name = APPNAME_PREFIX("chunkgen_compute_pipeline"),
    }).value();
    daxa::ComputePipeline depth_prepass0_compute_pipeline = render_context.pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"depth_prepass0.hlsl"}},
        .push_constant_size = sizeof(gpu::push::DepthPrepass),
        .debug_name = APPNAME_PREFIX("depth_prepass0_compute_pipeline"),
    }).value();
    daxa::ComputePipeline depth_prepass1_compute_pipeline = render_context.pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"depth_prepass1.hlsl"}},
        .push_constant_size = sizeof(gpu::push::DepthPrepass),
        .debug_name = APPNAME_PREFIX("depth_prepass1_compute_pipeline"),
    }).value();
    daxa::ComputePipeline draw_compute_pipeline = render_context.pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"draw.hlsl"}},
        .push_constant_size = sizeof(gpu::push::Draw),
        .debug_name = APPNAME_PREFIX("draw_compute_pipeline"),
    }).value();
    // clang-format on

    daxa::BufferId gpu_input_buffer = render_context.device.create_buffer({
        .size = sizeof(gpu::Input),
        .debug_name = "gpu_input_buffer",
    });
    daxa::BufferId staging_gpu_input_buffer = render_context.device.create_buffer({
        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
        .size = sizeof(gpu::Input),
        .debug_name = "staging_gpu_input_buffer",
    });
    daxa::BufferId gpu_globals_buffer = render_context.device.create_buffer({
        .size = sizeof(gpu::Globals),
        .debug_name = "gpu_globals_buffer",
    });
    daxa::TaskBufferId task_gpu_input_buffer;
    daxa::TaskBufferId task_staging_gpu_input_buffer;
    daxa::TaskBufferId task_gpu_globals_buffer;

    daxa::TaskImageId task_swapchain_image;
    daxa::TaskImageId task_color_image, task_pos_image;

    gpu::Input gpu_input{.fov = 90.0f};

    bool paused = true;
    bool laptop_power_saving = true;

    daxa::TaskList loop_task_list = record_loop_task_list();

    bool try_recreate_pipeline(daxa::ComputePipeline &compute_pipeline) {
        if (render_context.pipeline_compiler.check_if_sources_changed(compute_pipeline)) {
            auto new_pipeline = render_context.pipeline_compiler.recreate_compute_pipeline(compute_pipeline);
            if (new_pipeline.is_ok()) {
                compute_pipeline = new_pipeline.value();
                std::cout << "Shader Compilation SUCCESS!" << std::endl;
            } else {
                std::cout << new_pipeline.message() << std::endl;
            }
            return true;
        }
        return false;
    }

    void run_startup() {
        should_run_startup = true;
    }

    Game() {
        window.set_user_pointer<Game>(this);

        update();
        run_startup();
    }

    ~Game() {
        render_context.wait_idle();
        ImGui_ImplGlfw_Shutdown();
        render_context.device.destroy_buffer(gpu_input_buffer);
        render_context.device.destroy_buffer(staging_gpu_input_buffer);
        render_context.device.destroy_buffer(gpu_globals_buffer);
    }

    void update() {
        auto now = Clock::now();
        float dt = std::chrono::duration<float>(now - prev_frame_time).count();

        gpu_input.time = std::chrono::duration<float>(now - start_time).count();
        gpu_input.delta_time = dt;

        prev_frame_time = now;
        build_ui();
        window.update();
        if (try_recreate_pipeline(startup_compute_pipeline))
            run_startup();
        auto chunk_pipe0 = try_recreate_pipeline(chunkgen_compute_pipeline);
        auto chunk_pipe1 = try_recreate_pipeline(subchunk_x2x4_compute_pipeline);
        auto chunk_pipe2 = try_recreate_pipeline(subchunk_x8up_compute_pipeline);
        if (chunk_pipe0 || chunk_pipe1 || chunk_pipe2)
            run_startup();
        try_recreate_pipeline(perframe_compute_pipeline);
        try_recreate_pipeline(depth_prepass0_compute_pipeline);
        try_recreate_pipeline(depth_prepass1_compute_pipeline);
        try_recreate_pipeline(draw_compute_pipeline);

        run_frame();

        gpu_input.mouse.pos_delta = {0.0f, 0.0f};
        gpu_input.mouse.scroll_delta = {0.0f, 0.0f};

        if (laptop_power_saving)
            std::this_thread::sleep_for(10ms);
    }
    void run_frame() {
        if (window.frame_dim.x < 1 || window.frame_dim.y < 1) {
            std::this_thread::sleep_for(1ms);
            return;
        }

        gpu_input.frame_dim = {render_context.dim.x, render_context.dim.y};

        render_context.swapchain_image = render_context.swapchain.acquire_next_image();

        loop_task_list.execute();
        auto command_lists = loop_task_list.command_lists();
        auto cmd_list = render_context.device.create_command_list({});
        cmd_list.pipeline_barrier_image_transition({
            .awaited_pipeline_access = loop_task_list.last_access(task_swapchain_image),
            .before_layout = loop_task_list.last_layout(task_swapchain_image),
            .after_layout = daxa::ImageLayout::PRESENT_SRC,
            .image_id = render_context.swapchain_image,
        });
        cmd_list.complete();
        ++render_context.cpu_framecount;
        command_lists.push_back(cmd_list);
        render_context.device.submit_commands({
            .command_lists = command_lists,
            .signal_binary_semaphores = {render_context.binary_semaphore},
            .signal_timeline_semaphores = {{render_context.gpu_framecount_timeline_sema, render_context.cpu_framecount}},
        });
        render_context.device.present_frame({
            .wait_binary_semaphores = {render_context.binary_semaphore},
            .swapchain = render_context.swapchain,
        });
        render_context.gpu_framecount_timeline_sema.wait_for_value(render_context.cpu_framecount - 1);
    }

    void ImGui_settings_begin() {
        ImGui::Begin("Settings");

        ImGui::Text("Game");
        ImGui::ColorEdit3("Block Color", reinterpret_cast<float *>(&gpu_input.block_color));

        ImGui::Text("Graphics");
        ImGui::SliderFloat("FOV", &gpu_input.fov, 1.0f, 179.9f);
        bool enable_shadows = (gpu_input.flags >> 0) & 0x1;
        ImGui::Checkbox("Shadows", &enable_shadows);
        gpu_input.flags = (gpu_input.flags & ~(1 << 0)) | (static_cast<u32>(enable_shadows) << 0);
        i32 max_steps = gpu_input._pad0[0];
        ImGui::SliderInt("Maximum Steps", &max_steps, 1, 120);
        gpu_input._pad0[0] = max_steps;
        ImGui::Checkbox("Depth Prepass", &enable_depth_prepass);
        ImGui::Checkbox("Laptop Power Saving", &laptop_power_saving);
    }
    void build_ui() {
        auto &io = ImGui::GetIO();

        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (!started) {
            ImGui_settings_begin();
            if (ImGui::Button("START")) {
                started = true;
            }
            ImGui::End();
        } else {
            if (paused) {
                ImGui_settings_begin();
                if (ImGui::Button("STOP")) {
                    started = false;
                }
                ImGui::End();
            }
        }
        ImGui::Begin("Debug Stats");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        ImGui::End();

        ImGui::Render();
    }
    Window &get_window() { return window; }
    void on_mouse_move(const glm::dvec2 m) {
        if (!paused) {
            double center_x = static_cast<double>(window.frame_dim.x / 2);
            double center_y = static_cast<double>(window.frame_dim.y / 2);
            auto offset = glm::dvec2{m.x - center_x, center_y - m.y};
            window.set_mouse_pos(glm::vec2(center_x, center_y));
            gpu_input.mouse.pos = glm::vec2(m);
            gpu_input.mouse.pos_delta += glm::vec2(offset);
        }
    }
    void on_mouse_scroll(const glm::dvec2 offset) {
        if (!paused) {
            gpu_input.mouse.scroll_delta += glm::vec2(offset);
        }
    }
    void on_mouse_button(int button, int action) {
        if (!paused) {
            switch (button) {
            case GLFW_MOUSE_BUTTON_LEFT:
                should_break = action != GLFW_RELEASE;
                if (!should_break)
                    prev_break_time = start_time;
                break;
            case GLFW_MOUSE_BUTTON_RIGHT:
                should_place = action != GLFW_RELEASE;
                if (!should_place)
                    prev_place_time = start_time;
                break;
            default: break;
            }

            gpu_input.mouse.buttons[button] = action;
        }
    }
    void on_key(int key, int action) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            toggle_pause();
        if (key == GLFW_KEY_R && action == GLFW_PRESS)
            run_startup();

        if (!paused) {
            // clang-format off
            switch (key) {
            case GLFW_KEY_W:                  gpu_input.keyboard.keys[0] = action; break;
            case GLFW_KEY_A:                  gpu_input.keyboard.keys[1] = action; break;
            case GLFW_KEY_S:                  gpu_input.keyboard.keys[2] = action; break;
            case GLFW_KEY_D:                  gpu_input.keyboard.keys[3] = action; break;
            case GLFW_KEY_R:                  gpu_input.keyboard.keys[4] = action; break;
            case GLFW_KEY_F:                  gpu_input.keyboard.keys[5] = action; break;
            case GLFW_KEY_SPACE:              gpu_input.keyboard.keys[6] = action; break;
            case GLFW_KEY_LEFT_CONTROL:       gpu_input.keyboard.keys[7] = action; break;
            case GLFW_KEY_LEFT_SHIFT:         gpu_input.keyboard.keys[8] = action; break;
            case GLFW_KEY_F5:                 gpu_input.keyboard.keys[9] = action; break;
            }
            // clang-format on
            // gpu_input.keyboard.keys[key] = action;
        }
    }
    void on_resize() {
        if (window.frame_dim.x < 1 || window.frame_dim.y < 1) {
            return;
        }
        render_context.resize(window.frame_dim);
        update();
    }

    void toggle_pause() {
        window.set_mouse_capture(paused);
        paused = !paused;
    }

    auto record_loop_task_list() -> daxa::TaskList {
        daxa::TaskList new_task_list = daxa::TaskList({
            .device = render_context.device,
            .debug_name = "task_list",
        });
        task_swapchain_image = new_task_list.create_task_image({
            .fetch_callback = [this]() { return render_context.swapchain_image; },
            .debug_name = "task_swapchain_image",
        });
        task_color_image = new_task_list.create_task_image({
            .fetch_callback = [this]() { return render_context.render_col_images[render_context.frame_i]; },
            .debug_name = "task_color_image",
        });
        task_pos_image = new_task_list.create_task_image({
            .fetch_callback = [this]() { return render_context.render_pos_images[render_context.frame_i]; },
            .debug_name = "task_pos_image",
        });

        task_gpu_input_buffer = new_task_list.create_task_buffer({
            .fetch_callback = [this]() { return gpu_input_buffer; },
            .debug_name = "task_gpu_input_buffer",
        });
        task_staging_gpu_input_buffer = new_task_list.create_task_buffer({
            .fetch_callback = [this]() { return staging_gpu_input_buffer; },
            .debug_name = "task_staging_gpu_input_buffer",
        });
        task_gpu_globals_buffer = new_task_list.create_task_buffer({
            .fetch_callback = [this]() { return gpu_globals_buffer; },
            .debug_name = "task_gpu_globals_buffer",
        });

        new_task_list.add_task({
            .resources = {
                .buffers = {
                    {task_staging_gpu_input_buffer, daxa::TaskBufferAccess::HOST_TRANSFER_WRITE},
                },
            },
            .task = [this](daxa::TaskInterface /* interf */) {
                gpu::Input *buffer_ptr = render_context.device.map_memory_as<gpu::Input>(staging_gpu_input_buffer);
                *buffer_ptr = this->gpu_input;
                render_context.device.unmap_memory(staging_gpu_input_buffer);
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
                    .size = sizeof(gpu::Input),
                });
            },
            .debug_name = "Gpu Input Transfer",
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
                    cmd_list.set_pipeline(startup_compute_pipeline);
                    auto push = gpu::push::Startup{
                        .globals_id = gpu_globals_buffer,
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
                    {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
                },
            },
            .task = [this](daxa::TaskInterface interf) {
                if (started) {
                    auto cmd_list = interf.get_command_list();
                    cmd_list.set_pipeline(chunkgen_compute_pipeline);
                    auto push = gpu::push::Chunkgen{
                        .globals_id = gpu_globals_buffer,
                        .input_id = gpu_input_buffer,
                    };
                    cmd_list.push_constant(push);
                    cmd_list.dispatch(CHUNKGEN_DISPATCH_SIZE, CHUNKGEN_DISPATCH_SIZE, CHUNKGEN_DISPATCH_SIZE);
                }
            },
            .debug_name = "Chunkgen Task",
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
                    cmd_list.set_pipeline(subchunk_x2x4_compute_pipeline);
                    auto push = gpu::push::Subchunk{
                        .globals_id = gpu_globals_buffer,
                        .mode = 0,
                    };
                    cmd_list.push_constant(push);
                    cmd_list.dispatch(1, 64, 1);
                }
            },
            .debug_name = "Subchunk x2x4 Task",
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
                    cmd_list.set_pipeline(subchunk_x8up_compute_pipeline);
                    auto push = gpu::push::Subchunk{
                        .globals_id = gpu_globals_buffer,
                        .mode = 0,
                    };
                    cmd_list.push_constant(push);
                    cmd_list.dispatch(1, 1, 1);
                }
            },
            .debug_name = "Subchunk x8up Task",
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
                    cmd_list.set_pipeline(perframe_compute_pipeline);
                    auto push = gpu::push::Perframe{
                        .globals_id = gpu_globals_buffer,
                        .input_id = gpu_input_buffer,
                    };
                    cmd_list.push_constant(push);
                    cmd_list.dispatch(1, 1, 1);
                }
            },
            .debug_name = "Perframe Task",
        });

        new_task_list.add_task({
            .resources = {
                .buffers = {
                    {task_gpu_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
                    {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
                },
                .images = {
                    {task_pos_image, daxa::TaskImageAccess::COMPUTE_SHADER_READ_WRITE},
                },
            },
            .task = [this](daxa::TaskInterface interf) {
                if (started && enable_depth_prepass) {
                    auto cmd_list = interf.get_command_list();
                    cmd_list.set_pipeline(depth_prepass0_compute_pipeline);
                    u32 i = render_context.frame_i;
                    auto push = gpu::push::DepthPrepass{
                        .globals_id = gpu_globals_buffer,
                        .input_id = gpu_input_buffer,
                        .pos_image_id_in = render_context.render_pos_images[1 - i].default_view(),
                        .pos_image_id_out = render_context.render_pos_images[i].default_view(),
                        .scl = 1,
                    };
                    cmd_list.push_constant(push);
                    cmd_list.dispatch((render_context.dim.x + 15) / 16, (render_context.dim.y + 15) / 16, 1);
                }
            },
            .debug_name = "Depth-prepass 1 Task",
        });
        new_task_list.add_task({
            .resources = {
                .buffers = {
                    {task_gpu_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
                    {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
                },
                .images = {
                    {task_pos_image, daxa::TaskImageAccess::COMPUTE_SHADER_READ_WRITE},
                },
            },
            .task = [this](daxa::TaskInterface interf) {
                if (started && enable_depth_prepass) {
                    auto cmd_list = interf.get_command_list();
                    cmd_list.set_pipeline(depth_prepass1_compute_pipeline);
                    u32 i = render_context.frame_i;
                    auto push = gpu::push::DepthPrepass{
                        .globals_id = gpu_globals_buffer,
                        .input_id = gpu_input_buffer,
                        .pos_image_id_in = render_context.render_pos_images[1 - i].default_view(),
                        .pos_image_id_out = render_context.render_pos_images[i].default_view(),
                        .scl = 1,
                    };
                    cmd_list.push_constant(push);
                    cmd_list.dispatch((render_context.dim.x + 15) / 16, (render_context.dim.y + 15) / 16, 1);
                }
            },
            .debug_name = "Depth-prepass 2 Task",
        });
        new_task_list.add_task({
            .resources = {
                .buffers = {
                    {task_gpu_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
                    {task_gpu_globals_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
                },
                .images = {
                    {task_color_image, daxa::TaskImageAccess::COMPUTE_SHADER_READ_WRITE},
                    {task_pos_image, daxa::TaskImageAccess::COMPUTE_SHADER_READ_WRITE},
                },
            },
            .task = [this](daxa::TaskInterface interf) {
                if (started) {
                    auto cmd_list = interf.get_command_list();
                    cmd_list.set_pipeline(draw_compute_pipeline);
                    u32 i = render_context.frame_i;
                    auto push = gpu::push::Draw{
                        .globals_id = gpu_globals_buffer,
                        .input_id = gpu_input_buffer,
                        .col_image_id_in = render_context.render_col_images[1 - i].default_view(),
                        .col_image_id_out = render_context.render_col_images[i].default_view(),
                        .pos_image_id_in = render_context.render_pos_images[1 - i].default_view(),
                        .pos_image_id_out = render_context.render_pos_images[i].default_view(),
                    };
                    cmd_list.push_constant(push);
                    cmd_list.dispatch((render_context.dim.x + 7) / 8, (render_context.dim.y + 7) / 8, 1);
                }
            },
            .debug_name = "Draw Task",
        });

        new_task_list.add_task({
            .resources = {
                .images = {
                    {task_color_image, daxa::TaskImageAccess::TRANSFER_READ},
                    {task_swapchain_image, daxa::TaskImageAccess::TRANSFER_WRITE},
                },
            },
            .task = [this](daxa::TaskInterface interf) {
                auto cmd_list = interf.get_command_list();
                cmd_list.blit_image_to_image({
                    .src_image = render_context.render_col_images[render_context.frame_i],
                    .src_image_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    .dst_image = render_context.swapchain_image,
                    .dst_image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                    .src_slice = {.image_aspect = daxa::ImageAspectFlagBits::COLOR},
                    .src_offsets = {{{0, 0, 0}, {static_cast<i32>(render_context.dim.x), static_cast<i32>(render_context.dim.y), 1}}},
                    .dst_slice = {.image_aspect = daxa::ImageAspectFlagBits::COLOR},
                    .dst_offsets = {{{0, 0, 0}, {static_cast<i32>(render_context.dim.x), static_cast<i32>(render_context.dim.y), 1}}},
                });
                render_context.frame_i = 1 - render_context.frame_i;
            },
            .debug_name = "Blit Task",
        });
        new_task_list.add_task({
            .resources = {
                .images = {
                    {task_swapchain_image, daxa::TaskImageAccess::COLOR_ATTACHMENT},
                },
            },
            .task = [this](daxa::TaskInterface interf) {
                auto cmd_list = interf.get_command_list();
                imgui_renderer.record_commands(ImGui::GetDrawData(), cmd_list, render_context.swapchain_image, render_context.dim.x, render_context.dim.y);
            },
            .debug_name = "ImGui Task",
        });

        new_task_list.compile();
        new_task_list.output_graphviz();

        return new_task_list;
    }
};

int main() {
    Game game;
    while (true) {
        game.update();
        if (game.window.should_close())
            break;
    }
}
