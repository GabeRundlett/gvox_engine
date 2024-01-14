#pragma once

#include <shared/core.inl>

#include <shared/input.inl>
#include <shared/globals.inl>

#include <shared/voxels/voxels.inl>
#include <shared/voxels/impl/voxel_world.inl>
#include <shared/voxels/voxel_particle_sim.inl>
#include <shared/voxels/gvox_model.inl>

#include <shared/renderer/downscale.inl>
#include <shared/renderer/trace_primary.inl>
#include <shared/renderer/calculate_reprojection_map.inl>
#include <shared/renderer/ssao.inl>
#include <shared/renderer/trace_secondary.inl>
#include <shared/renderer/shadow_denoiser.inl>
#include <shared/renderer/taa.inl>
#include <shared/renderer/postprocessing.inl>
#include <shared/renderer/voxel_particle_raster.inl>
#include <shared/renderer/sky.inl>

#if STARTUP_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(StartupCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(GpuGlobals), globals)
VOXELS_USE_BUFFERS(daxa_RWBufferPtr, COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_HEAD_END
struct StartupComputePush {
    StartupCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(StartupComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_RWBufferPtr)
#endif
#endif

#if PERFRAME_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(PerframeCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(GpuOutput), gpu_output)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(GpuGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(SimulatedVoxelParticle), simulated_voxel_particles)
VOXELS_USE_BUFFERS(daxa_RWBufferPtr, COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_HEAD_END
struct PerframeComputePush {
    PerframeCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(PerframeComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuOutput) gpu_output = push.uses.gpu_output;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_RWBufferPtr(SimulatedVoxelParticle) simulated_voxel_particles = push.uses.simulated_voxel_particles;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_RWBufferPtr)
#endif
#endif

#if defined(__cplusplus)

static_assert(IsVoxelWorld<VoxelWorld>);

#include <minizip/unzip.h>

DECL_TASK_STATE("app.comp.glsl", Startup, STARTUP, 1, 1, 1);
DECL_TASK_STATE("app.comp.glsl", Perframe, PERFRAME, 1, 1, 1);

struct GpuResources {
    daxa::ImageId value_noise_image;
    daxa::ImageId blue_noise_vec2_image;
    daxa::ImageId debug_texture;

    daxa::BufferId input_buffer;
    daxa::BufferId output_buffer;
    daxa::BufferId staging_output_buffer;
    daxa::BufferId globals_buffer;
    daxa::BufferId gvox_model_buffer;

    daxa::BufferId simulated_voxel_particles_buffer;
    daxa::BufferId rendered_voxel_particles_buffer;
    daxa::BufferId placed_voxel_particles_buffer;

    daxa::SamplerId sampler_nnc;
    daxa::SamplerId sampler_lnc;
    daxa::SamplerId sampler_llc;
    daxa::SamplerId sampler_llr;

    void create(daxa::Device &device) {
        value_noise_image = device.create_image({
            .dimensions = 2,
            .format = daxa::Format::R8_UNORM,
            .size = {256, 256, 1},
            .array_layer_count = 256,
            .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
            .name = "value_noise_image",
        });
        blue_noise_vec2_image = device.create_image({
            .dimensions = 3,
            .format = daxa::Format::R8G8B8A8_UNORM,
            .size = {128, 128, 64},
            .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
            .name = "blue_noise_vec2_image",
        });
        input_buffer = device.create_buffer({
            .size = sizeof(GpuInput),
            .name = "input_buffer",
        });
        output_buffer = device.create_buffer({
            .size = sizeof(GpuOutput) * (FRAMES_IN_FLIGHT + 1),
            .name = "output_buffer",
        });
        staging_output_buffer = device.create_buffer({
            .size = sizeof(GpuOutput) * (FRAMES_IN_FLIGHT + 1),
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "staging_output_buffer",
        });
        globals_buffer = device.create_buffer({
            .size = sizeof(GpuGlobals),
            .name = "globals_buffer",
        });
        gvox_model_buffer = device.create_buffer({
            .size = static_cast<daxa_u32>(offsetof(GpuGvoxModel, data)),
            .name = "gvox_model_buffer",
        });
        simulated_voxel_particles_buffer = device.create_buffer({
            .size = sizeof(SimulatedVoxelParticle) * std::max<daxa_u32>(MAX_SIMULATED_VOXEL_PARTICLES, 1),
            .name = "simulated_voxel_particles_buffer",
        });
        rendered_voxel_particles_buffer = device.create_buffer({
            .size = sizeof(daxa_u32) * std::max<daxa_u32>(MAX_RENDERED_VOXEL_PARTICLES, 1),
            .name = "rendered_voxel_particles_buffer",
        });
        placed_voxel_particles_buffer = device.create_buffer({
            .size = sizeof(daxa_u32) * std::max<daxa_u32>(MAX_SIMULATED_VOXEL_PARTICLES, 1),
            .name = "placed_voxel_particles_buffer",
        });
        sampler_nnc = device.create_sampler({
            .magnification_filter = daxa::Filter::NEAREST,
            .minification_filter = daxa::Filter::NEAREST,
            .max_lod = 0.0f,
        });
        sampler_lnc = device.create_sampler({
            .magnification_filter = daxa::Filter::LINEAR,
            .minification_filter = daxa::Filter::NEAREST,
            .max_lod = 0.0f,
        });
        sampler_llc = device.create_sampler({
            .magnification_filter = daxa::Filter::LINEAR,
            .minification_filter = daxa::Filter::LINEAR,
            .max_lod = 0.0f,
        });
        sampler_llr = device.create_sampler({
            .magnification_filter = daxa::Filter::LINEAR,
            .minification_filter = daxa::Filter::LINEAR,
            .address_mode_u = daxa::SamplerAddressMode::REPEAT,
            .address_mode_v = daxa::SamplerAddressMode::REPEAT,
            .address_mode_w = daxa::SamplerAddressMode::REPEAT,
            .max_lod = 0.0f,
        });
    }
    void destroy(daxa::Device &device) const {
        device.destroy_image(value_noise_image);
        device.destroy_image(blue_noise_vec2_image);
        if (!debug_texture.is_empty()) {
            device.destroy_image(debug_texture);
        }
        device.destroy_buffer(input_buffer);
        device.destroy_buffer(output_buffer);
        device.destroy_buffer(staging_output_buffer);
        device.destroy_buffer(globals_buffer);
        if (!gvox_model_buffer.is_empty()) {
            device.destroy_buffer(gvox_model_buffer);
        }
        device.destroy_buffer(simulated_voxel_particles_buffer);
        device.destroy_buffer(rendered_voxel_particles_buffer);
        device.destroy_buffer(placed_voxel_particles_buffer);
        device.destroy_sampler(sampler_nnc);
        device.destroy_sampler(sampler_lnc);
        device.destroy_sampler(sampler_llc);
        device.destroy_sampler(sampler_llr);
    }
};

struct GpuApp : AppUi::DebugDisplayProvider {
    GbufferRenderer gbuffer_renderer;
    ReprojectionRenderer reprojection_renderer;
    SsaoRenderer ssao_renderer;
    ShadowRenderer shadow_renderer;
    Compositor compositor;
#if ENABLE_TAA
    TaaRenderer taa_renderer;
#endif
    SkyRenderer sky_renderer;
    ShadowDenoiser shadow_denoiser;

    VoxelWorld voxel_world;

    GpuResources gpu_resources;
    daxa::BufferId prev_gvox_model_buffer{};

    StartupComputeTaskState startup_task_state;
    PerframeComputeTaskState perframe_task_state;

    PostprocessingRasterTaskState postprocessing_task_state;
    DebugImageRasterTaskState debug_image_task_state;
    TestComputeTaskState test_task_state;
    VoxelParticleSimComputeTaskState voxel_particle_sim_task_state;
    VoxelParticleRasterTaskState voxel_particle_raster_task_state;

    daxa::TaskImage task_value_noise_image{{.name = "task_value_noise_image"}};
    daxa::TaskImage task_blue_noise_vec2_image{{.name = "task_blue_noise_vec2_image"}};
    daxa::TaskImage task_debug_texture{{.name = "task_debug_texture"}};

    daxa::TaskBuffer task_input_buffer{{.name = "task_input_buffer"}};
    daxa::TaskBuffer task_output_buffer{{.name = "task_output_buffer"}};
    daxa::TaskBuffer task_staging_output_buffer{{.name = "task_staging_output_buffer"}};
    daxa::TaskBuffer task_globals_buffer{{.name = "task_globals_buffer"}};
    daxa::TaskBuffer task_gvox_model_buffer{{.name = "task_gvox_model_buffer"}};
    daxa::TaskBuffer task_simulated_voxel_particles_buffer{{.name = "task_simulated_voxel_particles_buffer"}};
    daxa::TaskBuffer task_rendered_voxel_particles_buffer{{.name = "task_rendered_voxel_particles_buffer"}};
    daxa::TaskBuffer task_placed_voxel_particles_buffer{{.name = "task_placed_voxel_particles_buffer"}};

    daxa::BufferId test_buffer;
    daxa::TaskBuffer task_test_buffer{{.name = "task_test_buffer"}};

    GpuInput gpu_input{};
    GpuOutput gpu_output{};
    std::vector<std::string> ui_strings;

    bool needs_vram_calc = true;

    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point prev_phys_update_time = Clock::now();

    GpuApp(daxa::Device &device, AsyncPipelineManager &pipeline_manager, daxa::Format swapchain_format)
        : gbuffer_renderer{pipeline_manager},
          reprojection_renderer{pipeline_manager},
          ssao_renderer{pipeline_manager},
          shadow_renderer{pipeline_manager},
          compositor{pipeline_manager},
#if ENABLE_TAA
          taa_renderer{pipeline_manager},
#endif
          sky_renderer{pipeline_manager},
          shadow_denoiser{pipeline_manager},

          voxel_world{pipeline_manager},
          gpu_resources{},

          startup_task_state{pipeline_manager},
          perframe_task_state{pipeline_manager},
          postprocessing_task_state{pipeline_manager, swapchain_format},
          debug_image_task_state{pipeline_manager, swapchain_format},
          test_task_state{pipeline_manager},
          voxel_particle_sim_task_state{pipeline_manager},
          voxel_particle_raster_task_state{pipeline_manager} {

        gpu_resources.create(device);

        task_input_buffer.set_buffers({.buffers = std::array{gpu_resources.input_buffer}});
        task_output_buffer.set_buffers({.buffers = std::array{gpu_resources.output_buffer}});
        task_staging_output_buffer.set_buffers({.buffers = std::array{gpu_resources.staging_output_buffer}});
        task_globals_buffer.set_buffers({.buffers = std::array{gpu_resources.globals_buffer}});
        task_gvox_model_buffer.set_buffers({.buffers = std::array{gpu_resources.gvox_model_buffer}});
        task_simulated_voxel_particles_buffer.set_buffers({.buffers = std::array{gpu_resources.simulated_voxel_particles_buffer}});
        task_rendered_voxel_particles_buffer.set_buffers({.buffers = std::array{gpu_resources.rendered_voxel_particles_buffer}});
        task_placed_voxel_particles_buffer.set_buffers({.buffers = std::array{gpu_resources.placed_voxel_particles_buffer}});

        task_value_noise_image.set_images({.images = std::array{gpu_resources.value_noise_image}});
        task_blue_noise_vec2_image.set_images({.images = std::array{gpu_resources.blue_noise_vec2_image}});

        voxel_world.create(device);

        test_buffer = device.create_buffer({
            .size = static_cast<daxa_u32>(sizeof(uint32_t) * 8 * 8 * 8 * 64 * 64 * 64),
            .name = "test_buffer",
        });
        task_test_buffer.set_buffers({.buffers = std::array{test_buffer}});

        AppUi::DebugDisplay::s_instance->providers.push_back(this);
        AppUi::DebugDisplay::s_instance->providers.push_back(&voxel_world);

        {
            daxa::TaskGraph temp_task_graph = daxa::TaskGraph({
                .device = device,
                .name = "temp_task_graph",
            });
            temp_task_graph.use_persistent_image(task_blue_noise_vec2_image);
            temp_task_graph.add_task({
                .uses = {
                    daxa::TaskImageUse<daxa::TaskImageAccess::TRANSFER_WRITE>{task_blue_noise_vec2_image},
                },
                .task = [this](daxa::TaskInterface ti) {
                    auto staging_buffer = ti.get_device().create_buffer({
                        .size = static_cast<daxa_u32>(128 * 128 * 4 * 64 * 1),
                        .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                        .name = "staging_buffer",
                    });
                    auto *buffer_ptr = ti.get_device().get_host_address_as<uint8_t>(staging_buffer).value();
                    auto *stbn_zip = unzOpen("assets/STBN.zip");
                    for (auto i = 0; i < 64; ++i) {
                        [[maybe_unused]] int err = 0;
                        daxa_i32 size_x = 0;
                        daxa_i32 size_y = 0;
                        auto load_image = [&](char const *path, uint8_t *buffer_out_ptr) {
                            err = unzLocateFile(stbn_zip, path, 1);
                            assert(err == UNZ_OK);
                            auto file_info = unz_file_info{};
                            err = unzGetCurrentFileInfo(stbn_zip, &file_info, nullptr, 0, nullptr, 0, nullptr, 0);
                            assert(err == UNZ_OK);
                            auto file_data = std::vector<uint8_t>{};
                            file_data.resize(file_info.uncompressed_size);
                            err = unzOpenCurrentFile(stbn_zip);
                            assert(err == UNZ_OK);
                            err = unzReadCurrentFile(stbn_zip, file_data.data(), static_cast<uint32_t>(file_data.size()));
                            assert(err == file_data.size());

                            auto fi_mem = FreeImage_OpenMemory(file_data.data(), static_cast<DWORD>(file_data.size()));
                            auto fi_file_desc = FreeImage_GetFileTypeFromMemory(fi_mem, 0);
                            FIBITMAP *fi_bitmap = FreeImage_LoadFromMemory(fi_file_desc, fi_mem);
                            FreeImage_CloseMemory(fi_mem);
                            size_x = static_cast<int32_t>(FreeImage_GetWidth(fi_bitmap));
                            size_y = static_cast<int32_t>(FreeImage_GetHeight(fi_bitmap));
                            auto *temp_data = FreeImage_GetBits(fi_bitmap);
                            assert(temp_data != nullptr && "Failed to load image");
                            auto pixel_size = FreeImage_GetBPP(fi_bitmap);
                            if (pixel_size != 32) {
                                auto *temp = FreeImage_ConvertTo32Bits(fi_bitmap);
                                FreeImage_Unload(fi_bitmap);
                                fi_bitmap = temp;
                            }

                            if (temp_data != nullptr) {
                                assert(size_x == 128 && size_y == 128);
                                std::copy(temp_data + 0, temp_data + 128 * 128 * 4, buffer_out_ptr);
                            }
                            FreeImage_Unload(fi_bitmap);
                        };
                        auto vec2_name = std::string{"STBN/stbn_vec2_2Dx1D_128x128x64_"} + std::to_string(i) + ".png";
                        load_image(vec2_name.c_str(), buffer_ptr + (128 * 128 * 4) * i + (128 * 128 * 4 * 64) * 0);
                    }

                    auto &recorder = ti.get_recorder();
                    recorder.pipeline_barrier({
                        .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
                    });
                    recorder.destroy_buffer_deferred(staging_buffer);
                    recorder.copy_buffer_to_image({
                        .buffer = staging_buffer,
                        .buffer_offset = (size_t{128} * 128 * 4 * 64) * 0,
                        .image = task_blue_noise_vec2_image.get_state().images[0],
                        .image_extent = {128, 128, 64},
                    });
                    needs_vram_calc = true;
                },
                .name = "upload_blue_noise",
            });
            temp_task_graph.submit({});
            temp_task_graph.complete({});
            temp_task_graph.execute({});
        }

        {
            daxa::TaskGraph temp_task_graph = daxa::TaskGraph({
                .device = device,
                .name = "temp_task_graph",
            });

            auto texture_path = "assets/debug.png";
            auto fi_file_desc = FreeImage_GetFileType(texture_path, 0);
            FIBITMAP *fi_bitmap = FreeImage_Load(fi_file_desc, texture_path);
            auto size_x = static_cast<uint32_t>(FreeImage_GetWidth(fi_bitmap));
            auto size_y = static_cast<uint32_t>(FreeImage_GetHeight(fi_bitmap));
            auto *temp_data = FreeImage_GetBits(fi_bitmap);
            assert(temp_data != nullptr && "Failed to load image");
            auto pixel_size = FreeImage_GetBPP(fi_bitmap);
            if (pixel_size != 32) {
                auto *temp = FreeImage_ConvertTo32Bits(fi_bitmap);
                FreeImage_Unload(fi_bitmap);
                fi_bitmap = temp;
            }
            auto size = static_cast<daxa_u32>(size_x) * static_cast<daxa_u32>(size_y) * 4 * 1;

            gpu_resources.debug_texture = device.create_image({
                .dimensions = 2,
                .format = daxa::Format::R8G8B8A8_UNORM,
                .size = {static_cast<daxa_u32>(size_x), static_cast<daxa_u32>(size_y), 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                .name = "debug_texture",
            });

            task_debug_texture.set_images({.images = std::array{gpu_resources.debug_texture}});
            temp_task_graph.use_persistent_image(task_debug_texture);
            temp_task_graph.add_task({
                .uses = {
                    daxa::TaskImageUse<daxa::TaskImageAccess::TRANSFER_WRITE>{task_debug_texture},
                },
                .task = [&, this](daxa::TaskInterface ti) {
                    auto staging_buffer = ti.get_device().create_buffer({
                        .size = size,
                        .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                        .name = "staging_buffer",
                    });
                    auto *buffer_ptr = ti.get_device().get_host_address_as<uint8_t>(staging_buffer).value();
                    std::copy(temp_data + 0, temp_data + size, buffer_ptr);
                    FreeImage_Unload(fi_bitmap);

                    auto &recorder = ti.get_recorder();
                    recorder.pipeline_barrier({
                        .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
                    });
                    recorder.destroy_buffer_deferred(staging_buffer);
                    recorder.copy_buffer_to_image({
                        .buffer = staging_buffer,
                        .image = task_debug_texture.get_state().images[0],
                        .image_extent = {static_cast<daxa_u32>(size_x), static_cast<daxa_u32>(size_y), 1},
                    });
                    needs_vram_calc = true;
                },
                .name = "upload_debug_texture",
            });
            temp_task_graph.submit({});
            temp_task_graph.complete({});
            temp_task_graph.execute({});
        }
    }
    virtual ~GpuApp() override = default;

    virtual void add_ui() override {
        for (auto const &str : ui_strings) {
            ImGui::Text("%s", str.c_str());
        }
        ImGui::Text("Player pos: %.2f, %.2f, %.2f", static_cast<double>(gpu_output.player_pos.x), static_cast<double>(gpu_output.player_pos.y), static_cast<double>(gpu_output.player_pos.z));
        ImGui::Text("Player y/p/r: %.2f, %.2f, %.2f", static_cast<double>(gpu_output.player_rot.x), static_cast<double>(gpu_output.player_rot.y), static_cast<double>(gpu_output.player_rot.z));
        ImGui::Text("Player unit offs: %.2f, %.2f, %.2f", static_cast<double>(gpu_output.player_unit_offset.x), static_cast<double>(gpu_output.player_unit_offset.y), static_cast<double>(gpu_output.player_unit_offset.z));
    }

    void destroy(daxa::Device &device) {
        gpu_resources.destroy(device);
        voxel_world.destroy(device);

        device.destroy_buffer(test_buffer);
    }

    void calc_vram_usage(daxa::Device &device, daxa::TaskGraph &task_graph) {
        std::vector<AppUi::DebugDisplay::GpuResourceInfo> &debug_gpu_resource_infos = AppUi::DebugDisplay::s_instance->gpu_resource_infos;

        debug_gpu_resource_infos.clear();
        ui_strings.clear();

        size_t result_size = 0;

        // auto format_to_pixel_size = [](daxa::Format format) -> daxa_u32 {
        //     switch (format) {
        //     case daxa::Format::R16G16B16_SFLOAT: return 3 * 2;
        //     case daxa::Format::R16G16B16A16_SFLOAT: return 4 * 2;
        //     case daxa::Format::R32G32B32_SFLOAT: return 3 * 4;
        //     default:
        //     case daxa::Format::R32G32B32A32_SFLOAT: return 4 * 4;
        //     }
        // };

        // auto image_size = [&device, &format_to_pixel_size, &result_size, &debug_gpu_resource_infos](daxa::ImageId image) {
        //     if (image.is_empty()) {
        //         return;
        //     }
        //     auto image_info = device.info_image(image).value();
        //     auto size = format_to_pixel_size(image_info.format) * image_info.size.x * image_info.size.y * image_info.size.z;
        //     debug_gpu_resource_infos.push_back({
        //         .type = "image",
        //         .name = image_info.name.data(),
        //         .size = size,
        //     });
        //     result_size += size;
        // };
        auto buffer_size = [&device, &result_size, &debug_gpu_resource_infos](daxa::BufferId buffer) {
            if (buffer.is_empty()) {
                return;
            }
            auto buffer_info = device.info_buffer(buffer).value();
            debug_gpu_resource_infos.push_back({
                .type = "buffer",
                .name = buffer_info.name.data(),
                .size = buffer_info.size,
            });
            result_size += buffer_info.size;
        };

        buffer_size(gpu_resources.input_buffer);
        buffer_size(gpu_resources.globals_buffer);

        voxel_world.for_each_buffer(buffer_size);

        buffer_size(gpu_resources.gvox_model_buffer);
        buffer_size(gpu_resources.simulated_voxel_particles_buffer);
        buffer_size(gpu_resources.rendered_voxel_particles_buffer);
        buffer_size(gpu_resources.placed_voxel_particles_buffer);

        {
            auto size = task_graph.get_transient_memory_size();
            debug_gpu_resource_infos.push_back({
                .type = "buffer",
                .name = "Transient Memory Buffer",
                .size = size,
            });
            result_size += size;
        }

        needs_vram_calc = false;

        ui_strings.push_back(fmt::format("Est. VRAM usage: {} MB", static_cast<float>(result_size) / 1000000));
    }

    void begin_frame(daxa::Device &device, daxa::TaskGraph &task_graph, AppUi &ui) {
        gpu_input.sampler_nnc = gpu_resources.sampler_nnc;
        gpu_input.sampler_lnc = gpu_resources.sampler_lnc;
        gpu_input.sampler_llc = gpu_resources.sampler_llc;
        gpu_input.sampler_llr = gpu_resources.sampler_llr;

        gpu_input.flags &= ~GAME_FLAG_BITS_PAUSED;
        gpu_input.flags |= GAME_FLAG_BITS_PAUSED * static_cast<daxa_u32>(ui.paused);

        gpu_input.flags &= ~GAME_FLAG_BITS_NEEDS_PHYS_UPDATE;

        gpu_input.sky_settings = ui.settings.sky;

        auto now = Clock::now();
        if (now - prev_phys_update_time > std::chrono::duration<float>(GAME_PHYS_UPDATE_DT)) {
            gpu_input.flags |= GAME_FLAG_BITS_NEEDS_PHYS_UPDATE;
            prev_phys_update_time = now;
        }

        if (needs_vram_calc) {
            calc_vram_usage(device, task_graph);
        }

        bool should_realloc = voxel_world.check_for_realloc(device, gpu_output.voxel_world);

        if (should_realloc) {
            dynamic_buffers_realloc(device);
        }
    }

    void end_frame() {
        gbuffer_renderer.next_frame();
        ssao_renderer.next_frame();
#if ENABLE_TAA
        taa_renderer.next_frame();
#endif
        shadow_denoiser.next_frame();
    }

    void dynamic_buffers_realloc(daxa::Device &device) {
        auto temp_task_graph = daxa::TaskGraph({
            .device = device,
            .name = "temp_task_graph",
        });

        voxel_world.dynamic_buffers_realloc(temp_task_graph, needs_vram_calc);

        temp_task_graph.submit({});
        temp_task_graph.complete({});
        temp_task_graph.execute({});
    }

    void record_startup(RecordContext &record_ctx) {
        record_ctx.task_graph.use_persistent_buffer(task_input_buffer);
        record_ctx.task_graph.use_persistent_buffer(task_globals_buffer);

        voxel_world.use_buffers(record_ctx);

        voxel_world.record_startup(record_ctx);
        record_ctx.task_graph.add_task({
            .uses = {
                daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{task_globals_buffer},
            },
            .task = [this](daxa::TaskInterface ti) {
                auto &recorder = ti.get_recorder();
                recorder.clear_buffer({
                    .buffer = task_globals_buffer.get_state().buffers[0],
                    .offset = 0,
                    .size = sizeof(GpuGlobals),
                    .clear_value = 0,
                });
            },
            .name = "StartupTask (Globals Clear)",
        });
        record_ctx.task_graph.add_task(StartupComputeTask{
            .uses = {
                .gpu_input = task_input_buffer,
                .globals = task_globals_buffer,
                VOXELS_BUFFER_USES_ASSIGN(voxel_world.buffers),
            },
            .state = &startup_task_state,
            .thread_count = {1, 1, 1},
        });
    }

    void record_frame(RecordContext &record_ctx) {
        record_ctx.task_graph.use_persistent_image(task_value_noise_image);
        record_ctx.task_graph.use_persistent_image(task_blue_noise_vec2_image);
        record_ctx.task_graph.use_persistent_image(task_debug_texture);

        record_ctx.task_graph.use_persistent_buffer(task_input_buffer);
        record_ctx.task_graph.use_persistent_buffer(task_output_buffer);
        record_ctx.task_graph.use_persistent_buffer(task_staging_output_buffer);
        record_ctx.task_graph.use_persistent_buffer(task_globals_buffer);
        record_ctx.task_graph.use_persistent_buffer(task_gvox_model_buffer);

        record_ctx.task_graph.use_persistent_buffer(task_simulated_voxel_particles_buffer);
        record_ctx.task_graph.use_persistent_buffer(task_rendered_voxel_particles_buffer);
        record_ctx.task_graph.use_persistent_buffer(task_placed_voxel_particles_buffer);

        voxel_world.use_buffers(record_ctx);

        record_ctx.task_blue_noise_vec2_image = task_blue_noise_vec2_image;
        record_ctx.task_debug_texture = task_debug_texture;
        record_ctx.task_input_buffer = task_input_buffer;
        record_ctx.task_globals_buffer = task_globals_buffer;

        record_ctx.task_graph.add_task({
            .uses = {
                daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_WRITE>{task_input_buffer},
            },
            .task = [this](daxa::TaskInterface ti) {
                auto &recorder = ti.get_recorder();
                auto staging_input_buffer = ti.get_device().create_buffer({
                    .size = sizeof(GpuInput),
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "staging_input_buffer",
                });
                recorder.destroy_buffer_deferred(staging_input_buffer);
                auto *buffer_ptr = ti.get_device().get_host_address_as<GpuInput>(staging_input_buffer).value();
                *buffer_ptr = gpu_input;
                recorder.copy_buffer_to_buffer({
                    .src_buffer = staging_input_buffer,
                    .dst_buffer = task_input_buffer.get_state().buffers[0],
                    .size = sizeof(GpuInput),
                });
            },
            .name = "GpuInputUploadTransferTask",
        });

        record_ctx.task_graph.add_task(PerframeComputeTask{
            .uses = {
                .gpu_input = task_input_buffer,
                .gpu_output = task_output_buffer,
                .globals = task_globals_buffer,
                .simulated_voxel_particles = task_simulated_voxel_particles_buffer,
                VOXELS_BUFFER_USES_ASSIGN(voxel_world.buffers),
            },
            .state = &perframe_task_state,
            .thread_count = {1, 1, 1},
        });

#if MAX_RENDERED_VOXEL_PARTICLES > 0
        record_ctx.task_graph.add_task(VoxelParticleSimComputeTask{
            .uses = {
                .gpu_input = task_input_buffer,
                .globals = task_globals_buffer,
                VOXELS_BUFFER_USES_ASSIGN(voxel_world.buffers),
                .simulated_voxel_particles = task_simulated_voxel_particles_buffer,
                .rendered_voxel_particles = task_rendered_voxel_particles_buffer,
                .placed_voxel_particles = task_placed_voxel_particles_buffer,
            },
            .state = &voxel_particle_sim_task_state,
        });
#endif

        voxel_world.record_frame(record_ctx, task_gvox_model_buffer, task_value_noise_image);

        auto raster_color_image = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R32G32B32A32_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "raster_color_image",
        });

#if MAX_RENDERED_VOXEL_PARTICLES > 0
        auto raster_depth_image = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::D32_SFLOAT,
            .size = {record_ctx.render_resolution.x, record_ctx.render_resolution.y, 1},
            .name = "raster_depth_image",
        });
        record_ctx.task_graph.add_task(VoxelParticleRasterTask{
            .uses = {
                .gpu_input = task_input_buffer,
                .globals = task_globals_buffer,
                .simulated_voxel_particles = task_simulated_voxel_particles_buffer,
                .rendered_voxel_particles = task_rendered_voxel_particles_buffer,
                .render_image = raster_color_image,
                .depth_image_id = raster_depth_image,
            },
            .state = &voxel_particle_raster_task_state,
        });
#endif

        auto [sky_lut, transmittance_lut] = sky_renderer.render(record_ctx);

        auto [gbuffer_depth, velocity_image] = gbuffer_renderer.render(record_ctx, sky_lut, voxel_world.buffers);
        auto reprojection_map = reprojection_renderer.calculate_reprojection_map(record_ctx, gbuffer_depth, velocity_image);
        auto ssao_image = ssao_renderer.render(record_ctx, gbuffer_depth, reprojection_map);
        auto shadow_bitmap = shadow_renderer.render(record_ctx, gbuffer_depth, voxel_world.buffers);
        auto denoised_shadows = shadow_denoiser.denoise_shadow_bitmap(record_ctx, gbuffer_depth, shadow_bitmap, reprojection_map);

        auto irradiance = ssao_image;

        auto composited_image = compositor.render(record_ctx, gbuffer_depth, sky_lut, transmittance_lut, irradiance, denoised_shadows, raster_color_image);

        auto final_image = [&]() {
#if ENABLE_TAA
            return taa_renderer.render(record_ctx, composited_image, gbuffer_depth.depth.task_resources.output_resource, reprojection_map);
#else
            return composited_image;
#endif
        }();

        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "[final]"});

        auto &dbg_disp = *AppUi::DebugDisplay::s_instance;
        auto pass_iter = std::find_if(dbg_disp.passes.begin(), dbg_disp.passes.end(), [&](auto &pass) { return pass.name == dbg_disp.selected_pass_name; });
        if (pass_iter == dbg_disp.passes.end() || dbg_disp.selected_pass_name == "[final]") {
            record_ctx.task_graph.add_task(PostprocessingRasterTask{
                .uses = {
                    .gpu_input = task_input_buffer,
                    .composited_image_id = final_image,
                    .g_buffer_image_id = gbuffer_depth.gbuffer,
                    .render_image = record_ctx.task_swapchain_image,
                },
                .state = &postprocessing_task_state,
            });
        } else {
            auto const &pass = *pass_iter;
            debug_image_task_state.type = pass.type;
            record_ctx.task_graph.add_task(DebugImageRasterTask{
                .uses = {
                    .gpu_input = task_input_buffer,
                    .image_id = pass.task_image_id,
                    .render_image = record_ctx.task_swapchain_image,
                },
                .state = &debug_image_task_state,
            });
        }

        record_ctx.task_graph.add_task({
            .uses = {
                daxa::TaskBufferUse<daxa::TaskBufferAccess::TRANSFER_READ>{task_output_buffer},
                daxa::TaskBufferUse<daxa::TaskBufferAccess::HOST_TRANSFER_WRITE>{task_staging_output_buffer},
            },
            .task = [this](daxa::TaskInterface ti) {
                auto &recorder = ti.get_recorder();
                auto output_buffer = task_output_buffer.get_state().buffers[0];
                auto staging_output_buffer = gpu_resources.staging_output_buffer;
                auto frame_index = gpu_input.frame_index + 1;
                auto *buffer_ptr = ti.get_device().get_host_address_as<std::array<GpuOutput, (FRAMES_IN_FLIGHT + 1)>>(staging_output_buffer).value();
                daxa_u32 const offset = frame_index % (FRAMES_IN_FLIGHT + 1);
                gpu_output = (*buffer_ptr)[offset];
                recorder.copy_buffer_to_buffer({
                    .src_buffer = output_buffer,
                    .dst_buffer = staging_output_buffer,
                    .size = sizeof(GpuOutput) * (FRAMES_IN_FLIGHT + 1),
                });
            },
            .name = "GpuOutputDownloadTransferTask",
        });

        // record_ctx.task_graph.use_persistent_buffer(task_test_buffer);
        // record_ctx.task_graph.add_task(TestComputeTask{
        //     .uses = {
        //         .data = task_test_buffer,
        //     },
        //     .state = &test_task_state,
        // });

        needs_vram_calc = true;
    }
};

#endif
