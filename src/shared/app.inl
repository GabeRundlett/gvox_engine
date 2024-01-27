#pragma once

#include <shared/core.inl>

#include <shared/input.inl>
#include <shared/globals.inl>

#include <shared/voxels/voxels.inl>
#include <shared/voxels/impl/voxel_world.inl>
#include <shared/voxels/voxel_particles.inl>
#include <shared/voxels/gvox_model.inl>

#include <shared/renderer/downscale.inl>
#include <shared/renderer/trace_primary.inl>
#include <shared/renderer/calculate_reprojection_map.inl>
#include <shared/renderer/ssao.inl>
#include <shared/renderer/trace_secondary.inl>
#include <shared/renderer/shadow_denoiser.inl>
#include <shared/renderer/taa.inl>
#include <shared/renderer/postprocessing.inl>
#include <shared/renderer/sky.inl>
#include <shared/renderer/blur.inl>
#include <shared/renderer/fsr.inl>

#if StartupComputeShader || defined(__cplusplus)
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

#if PerframeComputeShader || defined(__cplusplus)
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

#if TestComputeShader || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(TestCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32), data)
DAXA_DECL_TASK_HEAD_END
struct TestComputePush {
    TestCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(TestComputePush, push)
daxa_RWBufferPtr(daxa_u32) data = push.uses.data;
#endif
#endif

#if defined(__cplusplus)

static_assert(IsVoxelWorld<VoxelWorld>);

#include <minizip/unzip.h>

inline void test_compute(RecordContext &record_ctx) {
    auto test_buffer = record_ctx.task_graph.create_transient_buffer({
        .size = static_cast<daxa_u32>(sizeof(uint32_t) * 8 * 8 * 8 * 64 * 64 * 64),
        .name = "test_buffer",
    });

    record_ctx.add(ComputeTask<TestCompute, TestComputePush, NoTaskInfo>{
        .source = daxa::ShaderFile{"test.comp.glsl"},
        .uses = {
            .data = test_buffer,
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TestCompute::Uses &, TestComputePush &push, NoTaskInfo const &) {
            ti.get_recorder().set_pipeline(pipeline);
            ti.get_recorder().push_constant(push);
            auto volume_size = uint32_t(8 * 64);
            ti.get_recorder().dispatch({(volume_size + 7) / 8, (volume_size + 7) / 8, (volume_size + 7) / 8});
        },
    });
}

struct GpuResources {
    daxa::ImageId value_noise_image;
    daxa::ImageId blue_noise_vec2_image;
    daxa::ImageId debug_texture;

    daxa::BufferId input_buffer;
    daxa::BufferId output_buffer;
    daxa::BufferId staging_output_buffer;
    daxa::BufferId globals_buffer;
    daxa::BufferId gvox_model_buffer;

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
        device.destroy_sampler(sampler_nnc);
        device.destroy_sampler(sampler_lnc);
        device.destroy_sampler(sampler_llc);
        device.destroy_sampler(sampler_llr);
    }
};

struct GpuApp : AppUi::DebugDisplayProvider {
    GbufferRenderer gbuffer_renderer;
    SsaoRenderer ssao_renderer;
    TaaRenderer taa_renderer;
    ShadowDenoiser shadow_denoiser;
    PostProcessor post_processor;
    std::unique_ptr<Fsr2Renderer> fsr2_renderer;
    SkyRenderer sky;

    VoxelWorld voxel_world;
    VoxelParticles particles;

    GpuResources gpu_resources;
    daxa::BufferId prev_gvox_model_buffer{};

    daxa::TaskImage task_value_noise_image{{.name = "task_value_noise_image"}};
    daxa::TaskImage task_blue_noise_vec2_image{{.name = "task_blue_noise_vec2_image"}};
    daxa::TaskImage task_debug_texture{{.name = "task_debug_texture"}};

    daxa::TaskBuffer task_input_buffer{{.name = "task_input_buffer"}};
    daxa::TaskBuffer task_output_buffer{{.name = "task_output_buffer"}};
    daxa::TaskBuffer task_staging_output_buffer{{.name = "task_staging_output_buffer"}};
    daxa::TaskBuffer task_globals_buffer{{.name = "task_globals_buffer"}};
    daxa::TaskBuffer task_gvox_model_buffer{{.name = "task_gvox_model_buffer"}};

    GpuInput gpu_input{};
    GpuOutput gpu_output{};
    std::vector<std::string> ui_strings;

    bool needs_vram_calc = true;

    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point prev_phys_update_time = Clock::now();
    daxa::Format swapchain_format;

    GpuApp(daxa::Device &device, daxa::Format a_swapchain_format)
        : post_processor{device},
          gpu_resources{},
          swapchain_format{a_swapchain_format} {

        sky.create(device);
        gpu_resources.create(device);
        voxel_world.create(device);
        particles.create(device);

        task_input_buffer.set_buffers({.buffers = std::array{gpu_resources.input_buffer}});
        task_output_buffer.set_buffers({.buffers = std::array{gpu_resources.output_buffer}});
        task_staging_output_buffer.set_buffers({.buffers = std::array{gpu_resources.staging_output_buffer}});
        task_globals_buffer.set_buffers({.buffers = std::array{gpu_resources.globals_buffer}});
        task_gvox_model_buffer.set_buffers({.buffers = std::array{gpu_resources.gvox_model_buffer}});

        task_value_noise_image.set_images({.images = std::array{gpu_resources.value_noise_image}});
        task_blue_noise_vec2_image.set_images({.images = std::array{gpu_resources.blue_noise_vec2_image}});

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
        if (ImGui::TreeNode("Player")) {
            ImGui::Text("pos: %.2f, %.2f, %.2f", static_cast<double>(gpu_output.player_pos.x), static_cast<double>(gpu_output.player_pos.y), static_cast<double>(gpu_output.player_pos.z));
            ImGui::Text("y/p/r: %.2f, %.2f, %.2f", static_cast<double>(gpu_output.player_rot.x), static_cast<double>(gpu_output.player_rot.y), static_cast<double>(gpu_output.player_rot.z));
            ImGui::Text("unit offs: %.2f, %.2f, %.2f", static_cast<double>(gpu_output.player_unit_offset.x), static_cast<double>(gpu_output.player_unit_offset.y), static_cast<double>(gpu_output.player_unit_offset.z));
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Auto-Exposure")) {
            ImGui::Text("Exposure multiple: %.2f", static_cast<double>(gpu_input.pre_exposure));
            auto hist_float = std::array<float, LUMINANCE_HISTOGRAM_BIN_COUNT>{};
            auto hist_min = static_cast<float>(post_processor.histogram[0]);
            auto hist_max = static_cast<float>(post_processor.histogram[0]);
            auto first_bin_with_value = -1;
            auto last_bin_with_value = -1;
            for (uint32_t i = 0; i < LUMINANCE_HISTOGRAM_BIN_COUNT; ++i) {
                if (first_bin_with_value == -1 && post_processor.histogram[i] != 0) {
                    first_bin_with_value = i;
                }
                if (post_processor.histogram[i] != 0) {
                    last_bin_with_value = i;
                }
                hist_float[i] = static_cast<float>(post_processor.histogram[i]);
                hist_min = std::min(hist_min, hist_float[i]);
                hist_max = std::max(hist_max, hist_float[i]);
            }
            ImGui::PlotHistogram("Histogram", hist_float.data(), static_cast<int>(hist_float.size()), 0, "hist", hist_min, hist_max, ImVec2(0, 120.0f));
            ImGui::Text("min %.2f | max %.2f", static_cast<double>(hist_min), static_cast<double>(hist_max));
            auto a = double(first_bin_with_value) / 256.0 * (LUMINANCE_HISTOGRAM_MAX_LOG2 - LUMINANCE_HISTOGRAM_MIN_LOG2) + LUMINANCE_HISTOGRAM_MIN_LOG2;
            auto b = double(last_bin_with_value) / 256.0 * (LUMINANCE_HISTOGRAM_MAX_LOG2 - LUMINANCE_HISTOGRAM_MIN_LOG2) + LUMINANCE_HISTOGRAM_MIN_LOG2;
            ImGui::Text("first bin %d (%.2f) | last bin %d (%.2f)", first_bin_with_value, exp2(a), last_bin_with_value, exp2(b));
            ImGui::TreePop();
        }
    }

    void destroy(daxa::Device &device) {
        sky.destroy(device);
        gpu_resources.destroy(device);
        voxel_world.destroy(device);
        particles.destroy(device);
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
        buffer_size(particles.simulated_voxel_particles_buffer);
        buffer_size(particles.rendered_voxel_particles_buffer);
        buffer_size(particles.placed_voxel_particles_buffer);

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
        gpu_input.pre_exposure = post_processor.exposure_state.pre_mult;
        gpu_input.pre_exposure_prev = post_processor.exposure_state.pre_mult_prev;
        gpu_input.pre_exposure_delta = post_processor.exposure_state.pre_mult_delta;

        if constexpr (!ENABLE_TAA) {
            fsr2_renderer->next_frame();
            fsr2_renderer->state.delta_time = gpu_input.delta_time;
            gpu_input.halton_jitter = fsr2_renderer->state.jitter;
        }

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

    void end_frame(AppUi &ui) {
        gbuffer_renderer.next_frame();
        ssao_renderer.next_frame();
        post_processor.next_frame(ui.settings.auto_exposure, gpu_input.delta_time);
        if constexpr (ENABLE_TAA) {
            taa_renderer.next_frame();
        }
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

        record_ctx.add(ComputeTask<StartupCompute, StartupComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"app.comp.glsl"},
            .uses = {
                .gpu_input = task_input_buffer,
                .globals = task_globals_buffer,
                VOXELS_BUFFER_USES_ASSIGN(voxel_world.buffers),
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, StartupCompute::Uses &, StartupComputePush &push, NoTaskInfo const &) {
                ti.get_recorder().set_pipeline(pipeline);
                ti.get_recorder().push_constant(push);
                ti.get_recorder().dispatch({1, 1, 1});
            },
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

        voxel_world.use_buffers(record_ctx);
        particles.use_buffers(record_ctx);

        record_ctx.task_blue_noise_vec2_image = task_blue_noise_vec2_image;
        record_ctx.task_debug_texture = task_debug_texture;
        record_ctx.task_input_buffer = task_input_buffer;
        record_ctx.task_globals_buffer = task_globals_buffer;

#if IMMEDIATE_SKY
        auto [sky_cube, ibl_cube] = generate_procedural_sky(record_ctx);
#else
        sky.use_images(record_ctx);
        auto sky_cube = sky.task_sky_cube.view().view({.layer_count = 6});
        auto ibl_cube = sky.task_ibl_cube.view().view({.layer_count = 6});
#endif
        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "sky_cube", .task_image_id = sky_cube, .type = DEBUG_IMAGE_TYPE_CUBEMAP});
        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "ibl_cube", .task_image_id = ibl_cube, .type = DEBUG_IMAGE_TYPE_CUBEMAP});

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

        // TODO: Refactor this. I hate that due to these tasks, I have to call this "use_buffers" thing above.
        record_ctx.add(ComputeTask<PerframeCompute, PerframeComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"app.comp.glsl"},
            .uses = {
                .gpu_input = task_input_buffer,
                .gpu_output = task_output_buffer,
                .globals = task_globals_buffer,
                .simulated_voxel_particles = particles.task_simulated_voxel_particles_buffer,
                VOXELS_BUFFER_USES_ASSIGN(voxel_world.buffers),
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, PerframeCompute::Uses &, PerframeComputePush &push, NoTaskInfo const &) {
                ti.get_recorder().set_pipeline(pipeline);
                ti.get_recorder().push_constant(push);
                ti.get_recorder().dispatch({1, 1, 1});
            },
        });

        particles.simulate(record_ctx, voxel_world.buffers);
        voxel_world.record_frame(record_ctx, task_gvox_model_buffer, task_value_noise_image);

        auto [particles_color_image, particles_depth_image] = particles.render(record_ctx);
        auto [gbuffer_depth, velocity_image] = gbuffer_renderer.render(record_ctx, voxel_world.buffers, particles.task_simulated_voxel_particles_buffer, particles_color_image, particles_depth_image);
        auto reprojection_map = calculate_reprojection_map(record_ctx, gbuffer_depth, velocity_image);
        auto ssao_image = ssao_renderer.render(record_ctx, gbuffer_depth, reprojection_map);
        auto shadow_bitmap = trace_shadows(record_ctx, gbuffer_depth, voxel_world.buffers);
        auto denoised_shadows = shadow_denoiser.denoise_shadow_bitmap(record_ctx, gbuffer_depth, shadow_bitmap, reprojection_map);
        auto composited_image = composite(record_ctx, gbuffer_depth, sky_cube, ibl_cube, ssao_image, denoised_shadows);
        fsr2_renderer = std::make_unique<Fsr2Renderer>(record_ctx.device, Fsr2Info{.render_resolution = record_ctx.render_resolution, .display_resolution = record_ctx.output_resolution});

        auto antialiased_image = [&]() {
            if constexpr (ENABLE_TAA) {
                return taa_renderer.render(record_ctx, composited_image, gbuffer_depth.depth.task_resources.output_resource, reprojection_map);
            } else {
                return fsr2_renderer->upscale(record_ctx, gbuffer_depth, composited_image, reprojection_map);
            }
        }();

        auto post_processed_image = post_processor.process(record_ctx, antialiased_image, record_ctx.output_resolution);

        AppUi::DebugDisplay::s_instance->passes.push_back({.name = "[final]"});

        auto &dbg_disp = *AppUi::DebugDisplay::s_instance;
        auto pass_iter = std::find_if(dbg_disp.passes.begin(), dbg_disp.passes.end(), [&](auto &pass) { return pass.name == dbg_disp.selected_pass_name; });
        if (pass_iter == dbg_disp.passes.end() || dbg_disp.selected_pass_name == "[final]") {
            tonemap_raster(record_ctx, antialiased_image, record_ctx.task_swapchain_image, swapchain_format);
        } else {
            debug_pass(record_ctx, *pass_iter, record_ctx.task_swapchain_image, swapchain_format);
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

        // test_compute(record_ctx);
        

        needs_vram_calc = true;
    }
};

#endif
