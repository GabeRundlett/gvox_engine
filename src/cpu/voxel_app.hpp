#pragma once

#include "app_window.hpp"
#include "app_ui.hpp"
#include "app_audio.hpp"

#include <shared/app.inl>

#include <chrono>
#include <future>

struct GvoxModelData {
    size_t size = 0;
    uint8_t *ptr = nullptr;
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

    AsyncPipelineManager main_pipeline_manager;

    AppUi ui;
    AppAudio audio;
    daxa::ImGuiRenderer imgui_renderer;
    GpuApp gpu_app;

    std::array<f32vec2, 128> halton_offsets{};
    GpuInput &gpu_input{gpu_app.gpu_input};
    GpuOutput &gpu_output{gpu_app.gpu_output};
    f32 render_res_scl{1.0f};
    GvoxContext *gvox_ctx;

    bool has_model = false;
    std::future<GvoxModelData> gvox_model_data_future;
    GvoxModelData gvox_model_data;
    bool model_is_loading = false;
    bool model_is_ready = false;

    enum class Conditions {
        COUNT,
    };
    std::array<bool, static_cast<usize>(Conditions::COUNT)> condition_values{};
    daxa::TaskGraph main_task_graph;

    VoxelApp();
    VoxelApp(VoxelApp const &) = delete;
    VoxelApp(VoxelApp &&) = delete;
    auto operator=(VoxelApp const &) -> VoxelApp & = delete;
    auto operator=(VoxelApp &&) -> VoxelApp & = delete;
    ~VoxelApp();

    void run();

    auto load_gvox_data() -> GvoxModelData;

    void on_update();
    void on_mouse_move(f32 x, f32 y);
    void on_mouse_scroll(f32 dx, f32 dy);
    void on_mouse_button(i32 button_id, i32 action);
    void on_key(i32 key_id, i32 action);
    void on_resize(u32 sx, u32 sy);
    void on_drop(std::span<char const *> filepaths);

    void compute_image_sizes();

    void update_seeded_value_noise();
    void run_startup(daxa::TaskGraph &temp_task_graph);
    void upload_model(daxa::TaskGraph &temp_task_graph);

    auto record_main_task_graph() -> daxa::TaskGraph;
};
