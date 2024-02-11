#pragma once

#include <application/window.hpp>
#include <application/ui.hpp>
#include <application/audio.hpp>

#include <app.inl>
#include <renderer/renderer.hpp>
#include <voxels/voxel_world.inl>
#include <voxels/voxel_particles.inl>
#include <voxels/model.hpp>

#include <chrono>
#include <future>

struct GpuResources {
    daxa::ImageId value_noise_image;
    daxa::ImageId blue_noise_vec2_image;
    daxa::ImageId debug_texture;

    daxa::BufferId input_buffer;
    daxa::BufferId output_buffer;
    daxa::BufferId staging_output_buffer;
    daxa::BufferId globals_buffer;

    daxa::SamplerId sampler_nnc;
    daxa::SamplerId sampler_lnc;
    daxa::SamplerId sampler_llc;
    daxa::SamplerId sampler_llr;

    void create(daxa::Device &device);
    void destroy(daxa::Device &device) const;
};

struct VoxelApp : AppWindow<VoxelApp> {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start = Clock::now();
    Clock::time_point prev_time;

    daxa::Instance daxa_instance;
    daxa::Device device;

    daxa::Swapchain swapchain;
    daxa::ImageId swapchain_image{};
    daxa::TaskImage task_swapchain_image{daxa::TaskImageInfo{.swapchain_image = true}};

    std::shared_ptr<AsyncPipelineManager> main_pipeline_manager;
    std::unordered_map<std::string, std::shared_ptr<AsyncManagedComputePipeline>> compute_pipelines;
    std::unordered_map<std::string, std::shared_ptr<AsyncManagedRasterPipeline>> raster_pipelines;
    TemporalBuffers temporal_buffers;

    AppUi ui;
    AppAudio audio;
    daxa::ImGuiRenderer imgui_renderer;

    // gpu_app
    Renderer renderer;

    VoxelWorld voxel_world;
    VoxelParticles particles;
    VoxelModelLoader voxel_model_loader;

    GpuResources gpu_resources;

    daxa::TaskImage task_value_noise_image{{.name = "task_value_noise_image"}};
    daxa::TaskImage task_blue_noise_vec2_image{{.name = "task_blue_noise_vec2_image"}};
    daxa::TaskImage task_debug_texture{{.name = "task_debug_texture"}};

    daxa::TaskBuffer task_input_buffer{{.name = "task_input_buffer"}};
    daxa::TaskBuffer task_output_buffer{{.name = "task_output_buffer"}};
    daxa::TaskBuffer task_staging_output_buffer{{.name = "task_staging_output_buffer"}};
    daxa::TaskBuffer task_globals_buffer{{.name = "task_globals_buffer"}};

    GpuInput gpu_input{};
    GpuOutput gpu_output{};
    std::vector<std::string> ui_strings;

    bool needs_vram_calc = true;

    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point prev_phys_update_time = Clock::now();
    // end gpu_app

    std::array<daxa_f32vec2, 128> halton_offsets{};
    daxa_f32 render_res_scl{1.0f};

    enum class Conditions {
        COUNT,
    };
    std::array<bool, static_cast<size_t>(Conditions::COUNT)> condition_values{};
    daxa::TaskGraph main_task_graph;

    VoxelApp();
    VoxelApp(VoxelApp const &) = delete;
    VoxelApp(VoxelApp &&) = delete;
    auto operator=(VoxelApp const &) -> VoxelApp & = delete;
    auto operator=(VoxelApp &&) -> VoxelApp & = delete;
    ~VoxelApp();

    void run();

    void on_update();
    void on_mouse_move(daxa_f32 x, daxa_f32 y);
    void on_mouse_scroll(daxa_f32 dx, daxa_f32 dy);
    void on_mouse_button(daxa_i32 button_id, daxa_i32 action);
    void on_key(daxa_i32 key_id, daxa_i32 action);
    void on_resize(daxa_u32 sx, daxa_u32 sy);
    void on_drop(std::span<char const *> filepaths);

    void compute_image_sizes();

    void update_seeded_value_noise();
    void run_startup(daxa::TaskGraph &temp_task_graph);

    auto record_main_task_graph() -> daxa::TaskGraph;

    void gpu_app_draw_ui();
    void gpu_app_calc_vram_usage(daxa::TaskGraph &task_graph);
    void gpu_app_begin_frame(daxa::TaskGraph &task_graph);
    void gpu_app_dynamic_buffers_realloc();
    void gpu_app_record_startup(RecordContext &record_ctx);
    void gpu_app_record_frame(RecordContext &record_ctx);
};
