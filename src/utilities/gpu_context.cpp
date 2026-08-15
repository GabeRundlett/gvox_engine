#include "gpu_context.hpp"
#include "daxa/device.hpp"

#include <application/input.inl>
#include <application/settings.inl>

#include <minizip/unzip.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <base/format.hpp>

#include <random>

GpuContext::GpuContext() {
    daxa_instance = daxa::create_instance({});
    auto required_implicit =
        daxa::ImplicitFeatureFlagBits::RAY_TRACING_PIPELINE |
        daxa::ImplicitFeatureFlagBits::SHADER_CLOCK |
        daxa::ImplicitFeatureFlagBits::SHADER_ATOMIC_INT64 |
        daxa::ImplicitFeatureFlagBits::SHADER_INT8 |
        daxa::ImplicitFeatureFlagBits::SHADER_INT16 |
        daxa::ImplicitFeatureFlagBits::SWAPCHAIN;

    auto device_info = daxa::DeviceInfo2{};
    device_info.name = "gvox_engine";
    device_info.max_allowed_buffers = 150'000;
    device_info.explicit_features = daxa::ExplicitFeatureFlagBits::ROBUSTNESS_2;
    device_info = daxa_instance.choose_device(required_implicit, device_info);
    device = daxa_instance.create_device_2(device_info);

    // IMPORTANT: these must be the very first sampler / image resources created
    // on this device, in exactly this order. src/renderer/globals.glsl hard-codes
    // the resulting bindless IDs, which is only valid because daxa hands out
    // sequential slots and nothing else can allocate before GpuContext does.
    // The assertions below catch any future reordering. See globals.glsl.
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
    value_noise_image = device.create_image({
        .dimensions = 2,
        .format = daxa::Format::R8_UNORM,
        .size = {256, 256, 1},
        .array_layer_count = 256,
        .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
        .name = "value_noise_image",
    });
    value_noise_image_view = device.create_image_view({
        .type = daxa::ImageViewType::REGULAR_2D_ARRAY,
        .format = daxa::Format::R8_UNORM,
        .image = value_noise_image,
        .slice = {.layer_count = 256},
        .name = "value_noise_image_view",
    });

    // Must match the literals in src/renderer/globals.glsl.
    static constexpr uint64_t GLOBALS_GLSL_SAMPLER_NNC = 2097152;
    static constexpr uint64_t GLOBALS_GLSL_SAMPLER_LNC = 2097153;
    static constexpr uint64_t GLOBALS_GLSL_SAMPLER_LLC = 2097154;
    static constexpr uint64_t GLOBALS_GLSL_SAMPLER_LLR = 2097155;
    static constexpr uint64_t GLOBALS_GLSL_VALUE_NOISE_TEX = 1;
    assert(std::bit_cast<uint64_t>(sampler_nnc) == GLOBALS_GLSL_SAMPLER_NNC && "globals.glsl g_sampler_nnc out of sync; something allocated a sampler before GpuContext");
    assert(std::bit_cast<uint64_t>(sampler_lnc) == GLOBALS_GLSL_SAMPLER_LNC && "globals.glsl g_sampler_lnc out of sync");
    assert(std::bit_cast<uint64_t>(sampler_llc) == GLOBALS_GLSL_SAMPLER_LLC && "globals.glsl g_sampler_llc out of sync");
    assert(std::bit_cast<uint64_t>(sampler_llr) == GLOBALS_GLSL_SAMPLER_LLR && "globals.glsl g_sampler_llr out of sync");
    // NOTE: .index is a u64 bitfield, so compare the value directly (it can't be bit_cast).
    assert(static_cast<uint64_t>(value_noise_image_view.index) == GLOBALS_GLSL_VALUE_NOISE_TEX && "globals.glsl g_value_noise_tex out of sync; something allocated an image before GpuContext");

    // Shader #include roots live in the pipeline manager itself (SHADER_ROOTS
    // in pipeline_manager.cpp).
    pipeline_manager = create_pipeline_manager();

    // NOTE: the 4 samplers and value_noise_image/_view are created at the very
    // top of this constructor -- see the comment there and globals.glsl.
    blue_noise_vec2_image = device.create_image({
        .flags = daxa::ImageCreateFlagBits::COMPATIBLE_2D_ARRAY,
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
    task_input_buffer.set_buffer(input_buffer);

    task_value_noise_image.set_image(value_noise_image);
    task_value_noise_image_view = task_value_noise_image.view().layers(0, 256);

    task_blue_noise_vec2_image.set_image(blue_noise_vec2_image);

    {
        daxa::TaskGraph temp_task_graph = daxa::TaskGraph({
            .device = device,
            .name = "temp_task_graph",
        });
        temp_task_graph.register_image(task_blue_noise_vec2_image);
        temp_task_graph.add_task(
            daxa::InlineTask::Transfer("upload_blue_noise")
                .transfer.writes(daxa::ImageViewType::REGULAR_3D, task_blue_noise_vec2_image)
                .executes([this](daxa::TaskInterface ti) {
                    auto staging_buffer = ti.device.create_buffer({
                        .size = 128 * 128 * 4 * 64 * 1,
                        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                        .name = "staging_buffer",
                    });
                    auto *buffer_ptr = ti.device.buffer_host_address_as<uint8_t>(staging_buffer).value();
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

                            int channels = 0;
                            auto *temp_data = stbi_load_from_memory(file_data.data(), file_data.size(), &size_x, &size_y, &channels, 4);
                            assert(temp_data != nullptr && "Failed to load image");

                            if (temp_data != nullptr) {
                                assert(size_x == 128 && size_y == 128);
                                std::copy(temp_data + 0, temp_data + 128 * 128 * 4, buffer_out_ptr);
                            }

                            stbi_image_free(temp_data);
                        };
                        auto vec2_name = std::string{"STBN/stbn_vec2_2Dx1D_128x128x64_"} + std::to_string(i) + ".png";
                        load_image(vec2_name.c_str(), buffer_ptr + (128 * 128 * 4) * i + (128 * 128 * 4 * 64) * 0);
                    }

                    ti.recorder.pipeline_barrier({
                        .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
                    });
                    ti.recorder.destroy_buffer_deferred(staging_buffer);
                    ti.recorder.copy_buffer_to_image({
                        .src_buffer = staging_buffer,
                        .buffer_offset = (size_t{128} * 128 * 4 * 64) * 0,
                        .dst_image = task_blue_noise_vec2_image.id(),
                        .image_extent = {128, 128, 64},
                    });
                }));
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
        int size_x = 0, size_y = 0, channels = 0;
        auto *temp_data = stbi_load(texture_path, &size_x, &size_y, &channels, 4);
        assert(temp_data != nullptr && "Failed to load image");
        auto size = static_cast<daxa_u32>(size_x) * static_cast<daxa_u32>(size_y) * 4 * 1;

        debug_texture = device.create_image({
            .dimensions = 2,
            .format = daxa::Format::R8G8B8A8_UNORM,
            .size = {static_cast<daxa_u32>(size_x), static_cast<daxa_u32>(size_y), 1},
            .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
            .name = "debug_texture",
        });

        task_debug_texture.set_image(debug_texture);
        temp_task_graph.register_image(task_debug_texture);
        temp_task_graph.add_task(
            daxa::InlineTask::Transfer("upload_debug_texture")
                .transfer.writes(daxa::ImageViewType::REGULAR_2D, task_debug_texture)
                .executes([&, this](daxa::TaskInterface ti) {
                    auto staging_buffer = ti.device.create_buffer({
                        .size = size,
                        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                        .name = "staging_buffer",
                    });
                    auto *buffer_ptr = ti.device.buffer_host_address_as<uint8_t>(staging_buffer).value();
                    std::copy(temp_data + 0, temp_data + size, buffer_ptr);
                    stbi_image_free(temp_data);

                    ti.recorder.pipeline_barrier({
                        .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
                    });
                    ti.recorder.destroy_buffer_deferred(staging_buffer);
                    ti.recorder.copy_buffer_to_image({
                        .src_buffer = staging_buffer,
                        .dst_image = task_debug_texture.id(),
                        .image_extent = {static_cast<daxa_u32>(size_x), static_cast<daxa_u32>(size_y), 1},
                    });
                }));
        temp_task_graph.submit({});
        temp_task_graph.complete({});
        temp_task_graph.execute({});
    }

    if (false) {
#if 0
        daxa::TaskGraph temp_task_graph = daxa::TaskGraph({
            .device = device,
            .name = "temp_task_graph",
        });

        auto texture_path = "C:/Users/gabe/Downloads/Rugged Terrain with Rocky Peaks/Rugged Terrain with Rocky Peaks Height Map EXR.exr";
        auto fi_file_desc = FreeImage_GetFileType(texture_path, 0);
        FIBITMAP *fi_bitmap = FreeImage_Load(fi_file_desc, texture_path);
        auto size_x = static_cast<uint32_t>(FreeImage_GetWidth(fi_bitmap));
        auto size_y = static_cast<uint32_t>(FreeImage_GetHeight(fi_bitmap));
        auto *temp_data = FreeImage_GetBits(fi_bitmap);
        assert(temp_data != nullptr && "Failed to load image");
        // auto pixel_size = FreeImage_GetBPP(fi_bitmap);
        // if (pixel_size != 32) {
        //     auto *temp = FreeImage_ConvertTo32Bits(fi_bitmap);
        //     FreeImage_Unload(fi_bitmap);
        //     fi_bitmap = temp;
        // }
        auto size = static_cast<daxa_u32>(size_x) * static_cast<daxa_u32>(size_y) * 1 * 4;

        test_texture = device.create_image({
            .dimensions = 2,
            .format = daxa::Format::R32_SFLOAT,
            .size = {static_cast<daxa_u32>(size_x), static_cast<daxa_u32>(size_y), 1},
            .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
            .name = "test_texture",
        });

        auto texture_path2 = "C:/Users/gabe/Downloads/Rugged Terrain with Rocky Peaks/Rugged Terrain with Rocky Peaks Diffuse EXR.exr";
        auto fi_file_desc2 = FreeImage_GetFileType(texture_path2, 0);
        FIBITMAP *fi_bitmap2 = FreeImage_Load(fi_file_desc2, texture_path2);
        auto size_x2 = static_cast<uint32_t>(FreeImage_GetWidth(fi_bitmap2));
        auto size_y2 = static_cast<uint32_t>(FreeImage_GetHeight(fi_bitmap2));
        {
            auto *temp = FreeImage_ConvertToRGBAF(fi_bitmap2);
            FreeImage_Unload(fi_bitmap2);
            fi_bitmap2 = temp;
        }
        auto *temp_data2 = FreeImage_GetBits(fi_bitmap2);
        assert(temp_data2 != nullptr && "Failed to load image");
        auto size2 = static_cast<daxa_u32>(size_x2) * static_cast<daxa_u32>(size_y2) * 4 * 4;

        test_texture2 = device.create_image({
            .dimensions = 2,
            .format = daxa::Format::R32G32B32A32_SFLOAT,
            .size = {static_cast<daxa_u32>(size_x2), static_cast<daxa_u32>(size_y2), 1},
            .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
            .name = "test_texture",
        });

        task_test_texture.set_image(test_texture);
        task_test_texture2.set_image(test_texture2);
        temp_task_graph.register_image(task_test_texture);
        temp_task_graph.register_image(task_test_texture2);
        temp_task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, daxa::ImageViewType::REGULAR_2D, task_test_texture),
                daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, daxa::ImageViewType::REGULAR_2D, task_test_texture2),
            },
            .task = [&, this](daxa::TaskInterface const &ti) {
                {
                    auto staging_buffer = ti.device.create_buffer({
                        .size = size,
                        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                        .name = "staging_buffer",
                    });
                    auto *buffer_ptr = ti.device.buffer_host_address_as<uint8_t>(staging_buffer).value();
                    std::copy(temp_data + 0, temp_data + size, buffer_ptr);
                    FreeImage_Unload(fi_bitmap);
                    ti.recorder.destroy_buffer_deferred(staging_buffer);
                    ti.recorder.copy_buffer_to_image({
                        .buffer = staging_buffer,
                        .image = task_test_texture.get_state().images[0],
                        .image_extent = {static_cast<daxa_u32>(size_x), static_cast<daxa_u32>(size_y), 1},
                    });
                }
                {
                    auto staging_buffer = ti.device.create_buffer({
                        .size = size2,
                        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                        .name = "staging_buffer",
                    });
                    auto *buffer_ptr = ti.device.buffer_host_address_as<uint8_t>(staging_buffer).value();
                    std::copy(temp_data2 + 0, temp_data2 + size2, buffer_ptr);
                    FreeImage_Unload(fi_bitmap2);
                    ti.recorder.destroy_buffer_deferred(staging_buffer);
                    ti.recorder.copy_buffer_to_image({
                        .buffer = staging_buffer,
                        .image = task_test_texture2.get_state().images[0],
                        .image_extent = {static_cast<daxa_u32>(size_x2), static_cast<daxa_u32>(size_y2), 1},
                    });
                }
            },
            .name = "upload_test_texture",
        });
        temp_task_graph.submit({});
        temp_task_graph.complete({});
        temp_task_graph.execute({});
#endif
    } else {
        task_test_texture.set_image(debug_texture);
        task_test_texture2.set_image(debug_texture);
    }
}

GpuContext::~GpuContext() {
    device.destroy_image(value_noise_image);
    device.destroy_image_view(value_noise_image_view);
    device.destroy_image(blue_noise_vec2_image);
    if (!debug_texture.is_empty()) {
        device.destroy_image(debug_texture);
    }
    if (!test_texture.is_empty()) {
        device.destroy_image(test_texture);
    }
    if (!test_texture2.is_empty()) {
        device.destroy_image(test_texture2);
    }
    device.destroy_buffer(input_buffer);
    device.destroy_sampler(sampler_nnc);
    device.destroy_sampler(sampler_lnc);
    device.destroy_sampler(sampler_llc);
    device.destroy_sampler(sampler_llr);

    for (auto const &slot : temporal_buffers) {
        device.destroy_buffer(slot.value.task_resource.id());
    }
    for (auto const &slot : temporal_images) {
        device.destroy_image(slot.value.task_resource.id());
    }
    // Pipelines are individually heap-allocated so their addresses stay stable
    // for recorded task-graph closures; free them here.
    for (auto &slot : ray_tracing_pipelines) {
        if (slot.value->sbt_storage.has_value()) {
            device.destroy_buffer(slot.value->sbt_storage.value().buffer);
        }
        delete slot.value;
    }
    for (auto &slot : compute_pipelines) {
        delete slot.value;
    }
    for (auto &slot : raster_pipelines) {
        delete slot.value;
    }

    destroy_pipeline_manager(pipeline_manager);
    pipeline_manager = nullptr;
}

void GpuContext::create_swapchain(daxa::SwapchainInfo const &info) {
    swapchain = device.create_swapchain(info);
}

void GpuContext::use_resources() {
    frame_task_graph.register_image(task_swapchain_image);

    auto use_shared_resources = [this](daxa::TaskGraph &task_graph) {
        task_graph.register_image(task_value_noise_image);
        task_graph.register_image(task_blue_noise_vec2_image);
        task_graph.register_image(task_debug_texture);
        task_graph.register_image(task_test_texture);
        task_graph.register_image(task_test_texture2);

        task_graph.register_buffer(task_input_buffer);
    };

    use_shared_resources(frame_task_graph);
    use_shared_resources(startup_task_graph);
}

void GpuContext::update_seeded_value_noise(uint64_t seed) {
    daxa::TaskGraph temp_task_graph = daxa::TaskGraph({
        .device = device,
        .name = "temp_task_graph",
    });
    temp_task_graph.register_image(task_value_noise_image);
    temp_task_graph.add_task(
        daxa::InlineTask::Transfer("upload_value_noise")
            .transfer.writes(daxa::ImageViewType::REGULAR_2D_ARRAY, task_value_noise_image_view)
            .executes([this, seed](daxa::TaskInterface ti) {
                auto staging_buffer = ti.device.create_buffer({
                    .size = 256 * 256 * 256 * 1,
                    .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "staging_buffer",
                });
                auto *buffer_ptr = ti.device.buffer_host_address_as<uint8_t>(staging_buffer).value();
                std::mt19937_64 rng(seed);
                std::uniform_int_distribution<std::mt19937::result_type> dist(0, 255);
                for (daxa_u32 i = 0; i < (256 * 256 * 256 * 1); ++i) {
                    buffer_ptr[i] = dist(rng) & 0xff;
                }
                ti.recorder.pipeline_barrier({
                    .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
                });
                ti.recorder.destroy_buffer_deferred(staging_buffer);
                for (daxa_u32 i = 0; i < 256; ++i) {
                    ti.recorder.copy_buffer_to_image({
                        .src_buffer = staging_buffer,
                        .buffer_offset = 256 * 256 * i,
                        .dst_image = task_value_noise_image.id(),
                        .image_slice{
                            .base_array_layer = i,
                            .layer_count = 1,
                        },
                        .image_extent = {256, 256, 1},
                    });
                }
            }));
    temp_task_graph.submit({});
    temp_task_graph.complete({});
    temp_task_graph.execute({});
}

auto GpuContext::find_or_add_temporal_buffer(daxa::BufferInfo const &info) -> TemporalBuffer {
    auto id = Str{info.name.c_str()};
    auto *existing = temporal_buffers.get(id);

    if (existing == nullptr) {
        auto result = TemporalBuffer{};
        auto buffer_id = device.create_buffer(info);
        result.task_resource = daxa::ExternalTaskBuffer(daxa::ExternalTaskBufferInfo{.buffer = buffer_id, .name = info.name.c_str()});
        temporal_buffers.set(id, result);
        existing = temporal_buffers.get(id);
    } else {
        auto existing_info = device.buffer_info(existing->task_resource.id()).value();
        if (existing_info.size != info.size) {
            debug_utils::Console::add_log(format("TemporalBuffer \"%s\" recreated with bad size... This should NEVER happen!!!", id.c_str()).data);
        }
    }

    return *existing;
}

auto GpuContext::find_or_add_temporal_image(daxa::ImageInfo const &info) -> TemporalImage {
    auto id = Str{info.name.c_str()};
    auto *existing = temporal_images.get(id);

    if (existing == nullptr) {
        auto result = TemporalImage{};
        auto image_id = device.create_image(info);
        result.task_resource = daxa::ExternalTaskImage(daxa::ExternalTaskImageInfo{.image = image_id, .name = info.name.c_str()});
        temporal_images.set(id, result);
        existing = temporal_images.get(id);
    } else {
        auto existing_info = device.image_info(existing->task_resource.id()).value();
        if (existing_info.size != info.size) {
            debug_utils::Console::add_log(format("TemporalImage \"%s\" recreated with bad size... This should NEVER happen!!!", id.c_str()).data);
        }
    }

    return *existing;
}

void GpuContext::remove_temporal_buffer(char const *id) {
    auto key = Str{id};
    if (temporal_buffers.get(key) != nullptr) {
        device.destroy_buffer(temporal_buffers.get(key)->task_resource.id());
        temporal_buffers.remove(key);
    }
}

void GpuContext::remove_temporal_image(char const *id) {
    auto key = Str{id};
    if (temporal_images.get(key) != nullptr) {
        device.destroy_image(temporal_images.get(key)->task_resource.id());
        temporal_images.remove(key);
    }
}

void GpuContext::remove_temporal_buffer(daxa::BufferId id) {
    remove_temporal_buffer(device.buffer_info(id).value().name.c_str());
}

void GpuContext::remove_temporal_image(daxa::ImageId id) {
    remove_temporal_image(device.image_info(id).value().name.c_str());
}
