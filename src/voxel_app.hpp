#pragma once

#include <application/window.hpp>
#include <application/ui.hpp>
#include <application/audio.hpp>
#include <application/player.hpp>

#include <renderer/renderer.hpp>
#include <voxels/voxel_world.inl>
#include <voxels/model.hpp>
#include <daxa/utils/imgui.hpp>

#include <utilities/gpu_context.hpp>

#include <chrono>
#include <future>

struct VoxelApp : AppWindow<VoxelApp> {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start = Clock::now();
    Clock::time_point prev_time;
    Clock::time_point prev_phys_update_time = Clock::now();

    GpuContext gpu_context;

    AppUi ui;
    AppAudio audio;
    daxa::ImGuiRenderer imgui_renderer;
    Renderer renderer;

    VoxelWorld voxel_world;
    VoxelParticles particles;
    VoxelModelLoader voxel_model_loader;

    PlayerInput player_input{};
    GpuInput gpu_input{};
    GpuOutput gpu_output{};
    std::vector<std::string> ui_strings;

    bool needs_vram_calc = true;

    daxa_f32 render_res_scl{1.0f};

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

    void run_startup();
    void record_tasks();

    void calc_vram_usage();
};
