#define GLM_DEPTH_ZERO_TO_ONE

#include "render_context.hpp"
#include "window.hpp"

#include <chrono>
#include <thread>
using namespace std::literals;

#include "defines.inl"

namespace gpu {
    struct MouseInput {
        glm::vec2 pos;
        glm::vec2 pos_delta;
        glm::vec2 scroll_delta;
        u32 buttons[GLFW_MOUSE_BUTTON_LAST + 1];
    };
    struct KeyboardInput {
        u32 keys[GLFW_KEY_LAST + 1];
    };

    struct Input {
        glm::ivec2 frame_dim;
        float time, delta_time;
        float fov;
        glm::vec3 block_color = {1.0f, 0.05f, 0.08f};
        u32 flags = 0;
        u32 _pad0[3] = {50, 0, 0};
        MouseInput mouse;
        KeyboardInput keyboard;
    };
    struct Globals {
        u8 data[u64{1} << 30];
    };
    struct Readback {
        glm::vec3 player_pos;
        u32 _pad0;
    };
    namespace push {
        struct Startup {
            u32 globals_id;
        };
        struct Chunkgen {
            u32 globals_id;
            u32 input_id;
        };
        struct Subchunk {
            u32 globals_id;
            u32 mode;
        };
        struct Perframe {
            u32 globals_id;
            u32 input_id;
        };
        struct Draw {
            u32 globals_id;
            u32 input_id;
            u32 col_image_id_in, col_image_id_out;
            u32 pos_image_id_in, pos_image_id_out;
            u32 nrm_image_id_in, nrm_image_id_out;
        };
        struct DepthPrepass {
            u32 globals_id;
            u32 input_id;
            u32 pos_image_id_in, pos_image_id_out;
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

    Window window;
    VkSurfaceKHR vulkan_surface = window.get_vksurface(daxa::instance->getVkInstance());
    RenderContext render_context{vulkan_surface, window.frame_dim};
    std::optional<daxa::ImGuiRenderer> imgui_renderer = std::nullopt;

    daxa::PipelineHandle startup_compute_pipeline;
    daxa::PipelineHandle perframe_compute_pipeline;

    daxa::PipelineHandle chunkgen_compute_pipeline;
    daxa::PipelineHandle subchunk_x2x4_compute_pipeline;
    daxa::PipelineHandle subchunk_x8up_compute_pipeline;
    daxa::PipelineHandle depth_prepass0_compute_pipeline;
    daxa::PipelineHandle depth_prepass1_compute_pipeline;
    daxa::PipelineHandle draw_compute_pipeline;

    daxa::BufferHandle gpu_input_buffer, gpu_globals_buffer;
    daxa::BufferHandle gpu_readback_buffer;
    daxa::DescriptorIndex gpu_input_id, gpu_globals_id;

    gpu::Input gpu_input{.fov = 90.0f};
    gpu::Readback readback_data;

    bool paused = true;

    void create_pipeline(daxa::PipelineHandle &pipe, const std::filesystem::path &path, const char *entry = "main") {
        auto result = render_context.pipeline_compiler->createComputePipeline({
            .shaderCI = {
                .pathToSource = std::filesystem::path("shaders/pipelines") / path,
                .shaderLang = daxa::ShaderLang::HLSL,
                .entryPoint = entry,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            },
            .overwriteSets = {daxa::BIND_ALL_SET_DESCRIPTION},
        });
        pipe = result.value();
    }
    bool try_recreate_pipeline(daxa::PipelineHandle &pipe) {
        if (render_context.pipeline_compiler->checkIfSourcesChanged(pipe)) {
            auto result = render_context.pipeline_compiler->recreatePipeline(pipe);
            std::cout << result << std::endl;
            if (result)
                pipe = result.value();
            return true;
        }
        return false;
    }

    void run_startup() {
        auto cmd_list = render_context.queue->getCommandList({});
        gpu_input.time = 0;
        cmd_list.singleCopyHostToBuffer({
            .src = reinterpret_cast<u8 *>(&gpu_input),
            .dst = gpu_input_buffer,
            .region = {.size = sizeof(gpu_input)},
        });
        cmd_list.queueMemoryBarrier(daxa::FULL_MEMORY_BARRIER);

        cmd_list.bindPipeline(startup_compute_pipeline);
        cmd_list.bindAll();
        cmd_list.pushConstant(
            VK_SHADER_STAGE_COMPUTE_BIT,
            gpu::push::Startup{
                .globals_id = gpu_globals_id,
            });
        cmd_list.dispatch(1, 1, 1);
        cmd_list.queueMemoryBarrier(daxa::FULL_MEMORY_BARRIER);

        render_context.submit(cmd_list);

        start_time = Clock::now();
        prev_place_time = start_time;
        prev_break_time = start_time;
    }

    Game() {
        window.set_user_pointer<Game>(this);
        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForVulkan(window.window_ptr, true);
        imgui_renderer.emplace(render_context.device, render_context.queue, render_context.pipeline_compiler);

        gpu_input_buffer = render_context.device->createBuffer({
            .size = sizeof(gpu::Input),
            .memoryType = daxa::MemoryType::GPU_ONLY,
        });
        gpu_globals_buffer = render_context.device->createBuffer({
            .size = sizeof(gpu::Globals),
            .memoryType = daxa::MemoryType::GPU_ONLY,
        });
        gpu_readback_buffer = render_context.device->createBuffer({
            .size = sizeof(gpu::Readback),
            .memoryType = daxa::MemoryType::GPU_ONLY,
        });

        create_pipeline(startup_compute_pipeline, "startup.hlsl");
        create_pipeline(chunkgen_compute_pipeline, "chunkgen.hlsl");
        create_pipeline(subchunk_x2x4_compute_pipeline, "subchunk_x2x4.hlsl");
        create_pipeline(subchunk_x8up_compute_pipeline, "subchunk_x8up.hlsl");
        create_pipeline(perframe_compute_pipeline, "perframe.hlsl");
        create_pipeline(depth_prepass0_compute_pipeline, "depth_prepass0.hlsl");
        create_pipeline(depth_prepass1_compute_pipeline, "depth_prepass1.hlsl");
        create_pipeline(draw_compute_pipeline, "draw.hlsl");

        gpu_globals_id = gpu_globals_buffer.getDescriptorIndex();
        gpu_input_id = gpu_input_buffer.getDescriptorIndex();

        update();
        run_startup();
    }

    ~Game() {
        render_context.wait_idle();
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
        // std::this_thread::sleep_for(10ms);
        std::cout << std::flush;
    }
    void run_frame() {
        if (window.frame_dim.x < 1 || window.frame_dim.y < 1) {
            std::this_thread::sleep_for(1ms);
            return;
        }

        auto cmd_list = render_context.begin_frame(window.frame_dim);

        auto render_image = render_context.render_col_images[render_context.frame_i];
        auto extent = render_image->getImageHandle()->getVkExtent3D();

        gpu_input.frame_dim = {extent.width, extent.height};
        if (started) {
            cmd_list.singleCopyHostToBuffer({
                .src = reinterpret_cast<u8 *>(&gpu_input),
                .dst = gpu_input_buffer,
                .region = {.size = sizeof(gpu_input)},
            });
            cmd_list.queueMemoryBarrier(daxa::FULL_MEMORY_BARRIER);

            cmd_list.bindPipeline(chunkgen_compute_pipeline);
            cmd_list.bindAll();
            cmd_list.pushConstant(
                VK_SHADER_STAGE_COMPUTE_BIT,
                gpu::push::Chunkgen{
                    .globals_id = gpu_globals_id,
                    .input_id = gpu_input_id,
                });
            cmd_list.dispatch(CHUNKGEN_DISPATCH_SIZE, CHUNKGEN_DISPATCH_SIZE, CHUNKGEN_DISPATCH_SIZE);
            cmd_list.queueMemoryBarrier(daxa::FULL_MEMORY_BARRIER);
            cmd_list.bindPipeline(subchunk_x2x4_compute_pipeline);
            cmd_list.bindAll();
            cmd_list.pushConstant(
                VK_SHADER_STAGE_COMPUTE_BIT,
                gpu::push::Subchunk{
                    .globals_id = gpu_globals_id,
                    .mode = 0,
                });
            cmd_list.dispatch(1, 64, 1);
            cmd_list.queueMemoryBarrier(daxa::FULL_MEMORY_BARRIER);
            cmd_list.bindPipeline(subchunk_x8up_compute_pipeline);
            cmd_list.bindAll();
            cmd_list.pushConstant(
                VK_SHADER_STAGE_COMPUTE_BIT,
                gpu::push::Subchunk{
                    .globals_id = gpu_globals_id,
                    .mode = 0,
                });
            cmd_list.dispatch(1, 1, 1);
            cmd_list.queueMemoryBarrier(daxa::FULL_MEMORY_BARRIER);

            cmd_list.bindPipeline(perframe_compute_pipeline);
            cmd_list.bindAll();
            cmd_list.pushConstant(
                VK_SHADER_STAGE_COMPUTE_BIT,
                gpu::push::Perframe{
                    .globals_id = gpu_globals_id,
                    .input_id = gpu_input_id,
                });
            cmd_list.dispatch(1, 1, 1);
            cmd_list.queueMemoryBarrier(daxa::FULL_MEMORY_BARRIER);

            // auto readback_buffer_future = cmd_list.singleCopyBufferToHost({
            //     .src = gpu_readback_buffer,
            //     .region = {.size = sizeof(gpu::Readback)},
            // });

            u32 i = render_context.frame_i;
            if (enable_depth_prepass) {
                cmd_list.bindPipeline(depth_prepass0_compute_pipeline);
                cmd_list.bindAll();
                cmd_list.pushConstant(
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    gpu::push::DepthPrepass{
                        .globals_id = gpu_globals_id,
                        .input_id = gpu_input_id,
                        .pos_image_id_in = render_context.render_pos_images[1 - i]->getDescriptorIndex(),
                        .pos_image_id_out = render_context.render_pos_images[i]->getDescriptorIndex(),
                        .scl = 1,
                    });
                cmd_list.dispatch((extent.width + 15) / 16, (extent.height + 15) / 16, 1);
                cmd_list.queueMemoryBarrier(daxa::FULL_MEMORY_BARRIER);
                cmd_list.bindPipeline(depth_prepass1_compute_pipeline);
                cmd_list.bindAll();
                cmd_list.pushConstant(
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    gpu::push::DepthPrepass{
                        .globals_id = gpu_globals_id,
                        .input_id = gpu_input_id,
                        .pos_image_id_in = render_context.render_pos_images[1 - i]->getDescriptorIndex(),
                        .pos_image_id_out = render_context.render_pos_images[i]->getDescriptorIndex(),
                        .scl = 1,
                    });
                cmd_list.dispatch((extent.width + 15) / 16, (extent.height + 15) / 16, 1);
                cmd_list.queueMemoryBarrier(daxa::FULL_MEMORY_BARRIER);
            }

            cmd_list.bindPipeline(draw_compute_pipeline);
            cmd_list.bindAll();
            cmd_list.pushConstant(
                VK_SHADER_STAGE_COMPUTE_BIT,
                gpu::push::Draw{
                    .globals_id = gpu_globals_id,
                    .input_id = gpu_input_id,
                    .col_image_id_in = render_context.render_col_images[1 - i]->getDescriptorIndex(),
                    .col_image_id_out = render_context.render_col_images[i]->getDescriptorIndex(),
                    .pos_image_id_in = render_context.render_pos_images[1 - i]->getDescriptorIndex(),
                    .pos_image_id_out = render_context.render_pos_images[i]->getDescriptorIndex(),
                    .nrm_image_id_in = render_context.render_nrm_images[1 - i]->getDescriptorIndex(),
                    .nrm_image_id_out = render_context.render_nrm_images[i]->getDescriptorIndex(),
                });
            cmd_list.dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
        }

        render_context.blit_to_swapchain(cmd_list);
        imgui_renderer->recordCommands(ImGui::GetDrawData(), cmd_list, render_context.swapchain_image.getImageViewHandle());
        render_context.end_frame(cmd_list);

        // render_context.device->waitIdle();
        // while (!readback_buffer_future.ready()) {
        // }
        // {
        //     auto memmapped_ptr = readback_buffer_future.getPtr();
        //     if (memmapped_ptr)
        //         readback_data = *reinterpret_cast<gpu::Readback *>(memmapped_ptr->hostPtr);
        // }
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
            gpu_input.keyboard.keys[key] = action;
        }
    }
    void on_resize() { update(); }

    void toggle_pause() {
        window.set_mouse_capture(paused);
        paused = !paused;
    }
};

int main() {
    daxa::initialize();
    {
        Game game;
        while (true) {
            game.update();
            if (game.window.should_close())
                break;
        }
    }
    daxa::cleanup();
}
