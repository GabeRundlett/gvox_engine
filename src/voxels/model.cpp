#include "model.hpp"

#include <utilities/mesh/mesh_model.hpp>

#include <gvox/adapters/input/byte_buffer.h>
#include <gvox/adapters/output/byte_buffer.h>
#include <gvox/adapters/parse/voxlap.h>

#include <voxels/gvox_model.inl>

#include <fstream>
#include <filesystem>
using namespace std::chrono_literals;

void VoxelModelLoader::create(daxa::Device &device) {
    this->device = device;
    gvox_ctx = gvox_create_context();
    gvox_model_buffer = device.create_buffer({
        .size = static_cast<uint32_t>(offsetof(GpuGvoxModel, data)),
        .name = "gvox_model_buffer",
    });
    task_gvox_model_buffer.set_buffers({.buffers = std::array{gvox_model_buffer}});
}

void VoxelModelLoader::destroy() {
    gvox_destroy_context(gvox_ctx);
    if (!gvox_model_buffer.is_empty()) {
        device.destroy_buffer(gvox_model_buffer);
    }
}

auto VoxelModelLoader::open_mesh_model() -> GvoxModelData {
    MeshModel mesh_model;
    ::open_mesh_model(device, mesh_model, gvox_model_path, "test");
    if (mesh_model.meshes.size() == 0) {
        debug_utils::Console::add_log("[error] Failed to load the mesh model");
        should_upload_gvox_model = false;
        return {};
    }

    auto mesh_gpu_input = MeshGpuInput{};
    mesh_gpu_input.size = {768, 768, 768};
    mesh_gpu_input.bound_min = mesh_model.bound_min;
    mesh_gpu_input.bound_max = mesh_model.bound_max;

    daxa::SamplerId texture_sampler = device.create_sampler({
        .magnification_filter = daxa::Filter::LINEAR,
        .minification_filter = daxa::Filter::LINEAR,
        .address_mode_u = daxa::SamplerAddressMode::REPEAT,
        .address_mode_v = daxa::SamplerAddressMode::REPEAT,
        .address_mode_w = daxa::SamplerAddressMode::REPEAT,
        .name = "texture_sampler",
    });
    daxa::BufferId mesh_gpu_input_buffer = device.create_buffer(daxa::BufferInfo{
        .size = sizeof(MeshGpuInput),
        .name = "mesh_gpu_input_buffer",
    });
    daxa::BufferId voxel_buffer = device.create_buffer(daxa::BufferInfo{
        .size = static_cast<uint32_t>(sizeof(uint32_t) * mesh_gpu_input.size.x * mesh_gpu_input.size.y * mesh_gpu_input.size.z),
        .name = "voxel_buffer",
    });
    daxa::BufferId staging_voxel_buffer = device.create_buffer({
        .size = static_cast<uint32_t>(sizeof(uint32_t) * mesh_gpu_input.size.x * mesh_gpu_input.size.y * mesh_gpu_input.size.z),
        .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
        .name = "staging_voxel_buffer",
    });
    auto preprocess_pipeline = main_pipeline_manager->add_compute_pipeline({
        .shader_info = {
            .source = daxa::ShaderFile{"mesh/preprocess.comp.glsl"},
        },
        .push_constant_size = sizeof(MeshPreprocessPush),
        .name = "preprocess_pipeline",
    });
    auto raster_pipeline = main_pipeline_manager->add_raster_pipeline({
        .vertex_shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"mesh/voxelize.raster.glsl"},
            .compile_options = {.defines = {{"RASTER_VERT", "1"}}},
        },
        .fragment_shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"mesh/voxelize.raster.glsl"},
            .compile_options = {.defines = {{"RASTER_FRAG", "1"}}},
        },
        .raster = {
            // .samples = 8,
            .conservative_raster_info = daxa::ConservativeRasterInfo{
                .mode = daxa::ConservativeRasterizationMode::OVERESTIMATE,
                .size = 0.0f,
            },
        },
        .push_constant_size = sizeof(MeshRasterPush),
        .name = "raster_pipeline",
    });
    daxa::CommandSubmitInfo submit_info;
    daxa::TaskGraph task_list = daxa::TaskGraph({
        .device = device,
        .name = "mesh voxel conv",
    });
    auto task_gpu_input_buffer = daxa::TaskBuffer(daxa::TaskBufferInfo{.initial_buffers = {.buffers = std::array{mesh_gpu_input_buffer}}, .name = "task_gpu_input_buffer"});
    task_list.use_persistent_buffer(task_gpu_input_buffer);
    auto task_vertex_buffer_ids = std::vector<daxa::BufferId>{};
    auto task_image_ids = std::vector<daxa::ImageId>{};
    for (auto const &mesh : mesh_model.meshes) {
        task_vertex_buffer_ids.push_back(mesh.vertex_buffer);
        task_image_ids.push_back(mesh.textures[0]->image_id);
        // task_list.add_runtime_buffer(task_vertex_buffer, mesh.vertex_buffer);
        // task_list.add_runtime_image(task_image_id, mesh.textures[0]->image_id);
    }
    auto task_vertex_buffer = daxa::TaskBuffer(daxa::TaskBufferInfo{.initial_buffers = {.buffers = task_vertex_buffer_ids}, .name = "task_vertex_buffer"});
    auto task_image_id = daxa::TaskImage(daxa::TaskImageInfo{.initial_images = {.images = task_image_ids}, .name = "task_image_id"});
    auto task_voxel_buffer = daxa::TaskBuffer(daxa::TaskBufferInfo{.initial_buffers = {.buffers = std::array{voxel_buffer}}, .name = "task_voxel_buffer"});
    auto task_staging_voxel_buffer = daxa::TaskBuffer(daxa::TaskBufferInfo{.initial_buffers = {.buffers = std::array{staging_voxel_buffer}}, .name = "task_staging_voxel_buffer"});
    task_list.use_persistent_buffer(task_vertex_buffer);
    task_list.use_persistent_image(task_image_id);
    task_list.use_persistent_buffer(task_voxel_buffer);
    task_list.use_persistent_buffer(task_staging_voxel_buffer);
    task_list.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, task_gpu_input_buffer),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, task_vertex_buffer),
        },
        .task = [&](daxa::TaskInterface const &ti) {
            {
                auto staging_gpu_input_buffer = device.create_buffer({
                    .size = sizeof(MeshGpuInput),
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "staging_gpu_input_buffer",
                });
                ti.recorder.destroy_buffer_deferred(staging_gpu_input_buffer);
                auto *buffer_ptr = device.get_host_address_as<MeshGpuInput>(staging_gpu_input_buffer).value();
                *buffer_ptr = mesh_gpu_input;
                ti.recorder.copy_buffer_to_buffer({
                    .src_buffer = staging_gpu_input_buffer,
                    .dst_buffer = mesh_gpu_input_buffer,
                    .size = sizeof(MeshGpuInput),
                });
            }
            {
                size_t vert_n = 0;
                for (auto const &mesh : mesh_model.meshes) {
                    vert_n += mesh.verts.size();
                }
                auto staging_vertex_buffer = device.create_buffer({
                    .size = static_cast<uint32_t>(sizeof(MeshVertex) * vert_n),
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "staging_vertex_buffer",
                });
                ti.recorder.destroy_buffer_deferred(staging_vertex_buffer);
                auto *buffer_ptr = device.get_host_address_as<MeshVertex>(staging_vertex_buffer).value();
                size_t vert_offset = 0;
                for (auto const &mesh : mesh_model.meshes) {
                    std::memcpy(buffer_ptr + vert_offset, mesh.verts.data(), sizeof(MeshVertex) * mesh.verts.size());
                    ti.recorder.copy_buffer_to_buffer({
                        .src_buffer = staging_vertex_buffer,
                        .dst_buffer = mesh.vertex_buffer,
                        .src_offset = sizeof(MeshVertex) * vert_offset,
                        .size = sizeof(MeshVertex) * mesh.verts.size(),
                    });
                    vert_offset += mesh.verts.size();
                }
            }
        },
        .name = "Input Transfer",
    });
    task_list.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, task_voxel_buffer),
        },
        .task = [&](daxa::TaskInterface const &ti) {
            ti.recorder.clear_buffer({
                .buffer = voxel_buffer,
                .offset = 0,
                .size = sizeof(uint32_t) * mesh_gpu_input.size.x * mesh_gpu_input.size.y * mesh_gpu_input.size.z,
                .clear_value = {},
            });
        },
        .name = "Clear Output",
    });
    task_list.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::COMPUTE_SHADER_READ, task_gpu_input_buffer),
            daxa::inl_attachment(daxa::TaskBufferAccess::COMPUTE_SHADER_WRITE, task_vertex_buffer),
        },
        .task = [&](daxa::TaskInterface const &ti) {
            for (auto const &mesh : mesh_model.meshes) {
                if (!preprocess_pipeline.is_valid()) {
                    return;
                }
                ti.recorder.set_pipeline(preprocess_pipeline.get());
                set_push_constant(
                    ti,
                    MeshPreprocessPush{
                        .modl_mat = mesh.modl_mat,
                        .gpu_input = device.get_device_address(mesh_gpu_input_buffer).value(),
                        .vertex_buffer = device.get_device_address(mesh.vertex_buffer).value(),
                        .normal_buffer = device.get_device_address(mesh.normal_buffer).value(),
                        .triangle_count = static_cast<uint32_t>(mesh.verts.size() / 3),
                    });
                ti.recorder.dispatch({.x = static_cast<uint32_t>(mesh.verts.size() / 3)});
            }
        },
        .name = "Preprocess Verts",
    });
    task_list.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::FRAGMENT_SHADER_READ, task_gpu_input_buffer),
            daxa::inl_attachment(daxa::TaskBufferAccess::FRAGMENT_SHADER_READ, task_vertex_buffer),
            daxa::inl_attachment(daxa::TaskBufferAccess::FRAGMENT_SHADER_WRITE, task_voxel_buffer),
            daxa::inl_attachment(daxa::TaskImageAccess::FRAGMENT_SHADER_SAMPLED, daxa::ImageViewType::REGULAR_2D, task_image_id.view().view({.base_mip_level = 0, .level_count = 4})),
        },
        .task = [&](daxa::TaskInterface const &ti) {
            auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({.render_area = {.width = mesh_gpu_input.size.x, .height = mesh_gpu_input.size.y}});
            if (!raster_pipeline.is_valid()) {
                return;
            }
            renderpass_recorder.set_pipeline(raster_pipeline.get());
            for (auto const &mesh : mesh_model.meshes) {
                set_push_constant(
                    ti, renderpass_recorder,
                    MeshRasterPush{
                        .gpu_input = device.get_device_address(mesh_gpu_input_buffer).value(),
                        .vertex_buffer = device.get_device_address(mesh.vertex_buffer).value(),
                        .normal_buffer = device.get_device_address(mesh.normal_buffer).value(),
                        .voxel_buffer = device.get_device_address(voxel_buffer).value(),
                        .texture_id = {.value = static_cast<uint32_t>(mesh.textures[0]->image_id.default_view().index)},
                        .texture_sampler = texture_sampler,
                    });
                renderpass_recorder.draw({.vertex_count = static_cast<uint32_t>(mesh.verts.size())});
            }
            ti.recorder = std::move(renderpass_recorder).end_renderpass();
        },
        .name = "Raster to Voxels",
    });
    task_list.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_READ, task_voxel_buffer),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, task_staging_voxel_buffer),
        },
        .task = [&](daxa::TaskInterface const &ti) {
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer = voxel_buffer,
                .dst_buffer = staging_voxel_buffer,
                .size = sizeof(uint32_t) * mesh_gpu_input.size.x * mesh_gpu_input.size.y * mesh_gpu_input.size.z,
            });
        },
        .name = "Output Transfer",
    });
    task_list.submit({});
    task_list.complete({});
    task_list.execute({});
    device.wait_idle();
    auto cleanup = [&]() {
        device.wait_idle();
        device.destroy_buffer(mesh_gpu_input_buffer);
        for (auto &mesh : mesh_model.meshes) {
            device.destroy_buffer(mesh.vertex_buffer);
            device.destroy_buffer(mesh.normal_buffer);
        }
        for (auto &[key, value] : mesh_model.textures) {
            device.destroy_image(value->image_id);
        }
        device.destroy_buffer(voxel_buffer);
        device.destroy_buffer(staging_voxel_buffer);
        device.destroy_sampler(texture_sampler);
    };
    auto *buffer_ptr = device.get_host_address_as<uint32_t>(staging_voxel_buffer).value();
    if (buffer_ptr == nullptr) {
        cleanup();
        debug_utils::Console::add_log("[error] Failed to voxelize the mesh model");
        should_upload_gvox_model = false;
        return {};
    }

    struct GpuOutputState {
        uint32_t *buffer_ptr;
        daxa_u32vec3 size;
    };

    auto gpu_output_state = GpuOutputState{
        .buffer_ptr = buffer_ptr,
        .size = mesh_gpu_input.size,
    };

    GvoxParseAdapterInfo procedural_adapter_info = {
        .base_info = {
            .name_str = "gpu_result",
            .create = [](GvoxAdapterContext *ctx, void const *user_state_ptr) -> void {
                gvox_adapter_set_user_pointer(ctx, const_cast<void *>(user_state_ptr));
            },
            .destroy = [](GvoxAdapterContext *) -> void {},
            .blit_begin = [](GvoxBlitContext * /*unused*/, GvoxAdapterContext * /*unused*/, GvoxRegionRange const * /*unused*/, uint32_t /*unused*/) {},
            .blit_end = [](GvoxBlitContext * /*unused*/, GvoxAdapterContext * /*unused*/) {},
        },
        .query_details = []() -> GvoxParseAdapterDetails { return {.preferred_blit_mode = GVOX_BLIT_MODE_SERIALIZE_DRIVEN}; },
        .query_parsable_range = [](GvoxBlitContext * /*unused*/, GvoxAdapterContext * /*unused*/) -> GvoxRegionRange { return {{0, 0, 0}, {0, 0, 0}}; },
        .sample_region = [](GvoxBlitContext * /*unused*/, GvoxAdapterContext *ctx, GvoxRegion const * /*unused*/, GvoxOffset3D const *offset, uint32_t channel_id) -> GvoxSample {
            auto &[buffer_ptr_, size] = *static_cast<GpuOutputState *>(gvox_adapter_get_user_pointer(ctx));
            auto voxel_i = static_cast<size_t>(offset->x) + size.x * offset->y + size.x * size.y * offset->z;
            auto uint32_t_voxel = buffer_ptr_[voxel_i];
            // float r = static_cast<float>((uint32_t_voxel >> 0x00) & 0xff) / 255.0f;
            // float g = static_cast<float>((uint32_t_voxel >> 0x08) & 0xff) / 255.0f;
            // float b = static_cast<float>((uint32_t_voxel >> 0x10) & 0xff) / 255.0f;
            // uint32_t const id = (uint32_t_voxel >> 0x18) & 0xff;
            switch (channel_id) {
            case GVOX_CHANNEL_ID_COLOR: return {uint32_t_voxel, 1u};
            // case GVOX_CHANNEL_ID_NORMAL: return {0x0, 1u};
            // case GVOX_CHANNEL_ID_MATERIAL_ID: return {id, 1u};
            default:
                gvox_adapter_push_error(ctx, GVOX_RESULT_ERROR_PARSE_ADAPTER_INVALID_INPUT, "Tried sampling something other than color or normal");
                return {0u, 0u};
            }
            return {};
        },
        .query_region_flags = [](GvoxBlitContext * /*unused*/, GvoxAdapterContext * /*unused*/, GvoxRegionRange const * /*unused*/, uint32_t /*unused*/) -> uint32_t { return 0; },
        .load_region = [](GvoxBlitContext * /*unused*/, GvoxAdapterContext *ctx, GvoxRegionRange const *range, uint32_t channel_flags) -> GvoxRegion {
            GvoxRegion const region = {.range = *range, .channels = channel_flags, .flags = 0u, .data = nullptr};
            return region;
        },
        .unload_region = [](GvoxBlitContext * /*unused*/, GvoxAdapterContext * /*unused*/, GvoxRegion * /*unused*/) {},
        .parse_region = [](GvoxBlitContext *blit_ctx, GvoxAdapterContext *ctx, GvoxRegionRange const *range, uint32_t channel_flags) -> void {
            GvoxRegion const region = {.range = *range, .channels = channel_flags, .flags = 0u, .data = nullptr};
            gvox_emit_region(blit_ctx, &region);
        },
    };

    static auto parse_adapter = gvox_register_parse_adapter(gvox_ctx, &procedural_adapter_info);

    GvoxAdapterContext *p_ctx = gvox_create_adapter_context(gvox_ctx, parse_adapter, &gpu_output_state);
    GvoxRegionRange region_range = {
        .offset = {0, 0, 0},
        .extent = {
            mesh_gpu_input.size.x,
            mesh_gpu_input.size.y,
            mesh_gpu_input.size.z,
        },
    };
    auto result = load_gvox_data_from_parser(nullptr, p_ctx, &region_range);
    gvox_destroy_adapter_context(p_ctx);
    cleanup();
    if (result.size == 0) {
        should_upload_gvox_model = false;
    }
    return result;
}

void VoxelModelLoader::update(AppUi &ui) {
    if (ui.should_upload_gvox_model) {
        should_upload_gvox_model = ui.should_upload_gvox_model;
        ui.should_upload_gvox_model = false;
        gvox_model_path = ui.gvox_model_path;
    }
    if (should_upload_gvox_model) {
        if (false) {
            // async model loading.. broken with mesh import
            if (!model_is_loading) {
                gvox_model_data_future = std::async(std::launch::async, &VoxelModelLoader::load_gvox_data, this);
                model_is_loading = true;
            }
            if (model_is_loading && gvox_model_data_future.wait_for(0.01s) == std::future_status::ready) {
                model_is_ready = true;
                model_is_loading = false;
                gvox_model_data = gvox_model_data_future.get();
                prev_gvox_model_buffer = gvox_model_buffer;
                gvox_model_buffer = device.create_buffer({
                    .size = static_cast<uint32_t>(gvox_model_data.size),
                    .name = "gvox_model_buffer",
                });
                task_gvox_model_buffer.set_buffers({.buffers = std::array{gvox_model_buffer}});
            }
        } else {
            if (!model_is_loading) {
                model_is_loading = true;
            }
            if (model_is_loading) {
                model_is_loading = false;
                gvox_model_data = load_gvox_data();
                if (gvox_model_data.size != 0) {
                    model_is_ready = true;
                    prev_gvox_model_buffer = gvox_model_buffer;
                    gvox_model_buffer = device.create_buffer({
                        .size = static_cast<uint32_t>(gvox_model_data.size),
                        .name = "gvox_model_buffer",
                    });
                    task_gvox_model_buffer.set_buffers({.buffers = std::array{gvox_model_buffer}});
                }
            }
        }
    }

    if (model_is_ready) {
        upload_model();
    }
}

void VoxelModelLoader::upload_model() {
    auto temp_task_graph = daxa::TaskGraph({
        .device = device,
        .name = "temp_task_graph",
    });
    temp_task_graph.use_persistent_buffer(task_gvox_model_buffer);
    temp_task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, task_gvox_model_buffer),
        },
        .task = [this](daxa::TaskInterface const &ti) {
            if (!prev_gvox_model_buffer.is_empty()) {
                ti.recorder.destroy_buffer_deferred(prev_gvox_model_buffer);
            }
            ti.recorder.pipeline_barrier({
                .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
            });
            auto staging_gvox_model_buffer = device.create_buffer({
                .size = static_cast<uint32_t>(gvox_model_data.size),
                .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = "staging_gvox_model_buffer",
            });
            ti.recorder.destroy_buffer_deferred(staging_gvox_model_buffer);
            char *buffer_ptr = device.get_host_address_as<char>(staging_gvox_model_buffer).value();
            std::copy(gvox_model_data.ptr, gvox_model_data.ptr + gvox_model_data.size, buffer_ptr);
            if (gvox_model_data.ptr != nullptr) {
                free(gvox_model_data.ptr);
            }
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer = staging_gvox_model_buffer,
                .dst_buffer = gvox_model_buffer,
                .size = static_cast<uint32_t>(gvox_model_data.size),
            });
            should_upload_gvox_model = false;
            has_model = true;
        },
        .name = "upload_model",
    });
    temp_task_graph.submit({});
    temp_task_graph.complete({});
    temp_task_graph.execute({});
    model_is_ready = false;
}

auto VoxelModelLoader::load_gvox_data_from_parser(GvoxAdapterContext *i_ctx, GvoxAdapterContext *p_ctx, GvoxRegionRange const *region_range) -> GvoxModelData {
    auto result = GvoxModelData{};
    GvoxByteBufferOutputAdapterConfig o_config = {
        .out_size = &result.size,
        .out_byte_buffer_ptr = &result.ptr,
        .allocate = nullptr,
    };
    GvoxAdapterContext *o_ctx = gvox_create_adapter_context(gvox_ctx, gvox_get_output_adapter(gvox_ctx, "byte_buffer"), &o_config);
    GvoxAdapterContext *s_ctx = gvox_create_adapter_context(gvox_ctx, gvox_get_serialize_adapter(gvox_ctx, "gvox_palette"), nullptr);

    {
        // time_t start = clock();
        gvox_blit_region(
            i_ctx, o_ctx, p_ctx, s_ctx,
            region_range,
            // GVOX_CHANNEL_BIT_COLOR | GVOX_CHANNEL_BIT_MATERIAL_ID | GVOX_CHANNEL_BIT_EMISSIVITY);
            GVOX_CHANNEL_BIT_COLOR);
        // time_t end = clock();
        // double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        // debug_utils::Console::add_log("{}s, new size: {} bytes", cpu_time_used, result.size);

        GvoxResult res = gvox_get_result(gvox_ctx);
        // int error_count = 0;
        while (res != GVOX_RESULT_SUCCESS) {
            size_t size = 0;
            gvox_get_result_message(gvox_ctx, nullptr, &size);
            char *str = new char[size + 1];
            gvox_get_result_message(gvox_ctx, str, nullptr);
            str[size] = '\0';
            debug_utils::Console::add_log(fmt::format("ERROR loading model: {}", str));
            gvox_pop_result(gvox_ctx);
            delete[] str;
            res = gvox_get_result(gvox_ctx);
            // ++error_count;
            return {};
        }
    }

    gvox_destroy_adapter_context(o_ctx);
    gvox_destroy_adapter_context(s_ctx);
    return result;
}

auto VoxelModelLoader::load_gvox_data() -> GvoxModelData {
    auto result = GvoxModelData{};
    auto file = std::ifstream(gvox_model_path, std::ios::binary);
    if (!file.is_open()) {
        debug_utils::Console::add_log("[error] Failed to load the model");
        should_upload_gvox_model = false;
        return result;
    }
    file.seekg(0, std::ios_base::end);
    auto temp_gvox_model_size = static_cast<uint32_t>(file.tellg());
    auto temp_gvox_model = std::vector<uint8_t>{};
    temp_gvox_model.resize(temp_gvox_model_size);
    {
        // time_t start = clock();
        file.seekg(0, std::ios_base::beg);
        file.read(reinterpret_cast<char *>(temp_gvox_model.data()), static_cast<std::streamsize>(temp_gvox_model_size));
        file.close();
        // time_t end = clock();
        // double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        // debug_utils::Console::add_log("(pulling file into memory: {}s)", cpu_time_used);
    }
    GvoxByteBufferInputAdapterConfig i_config = {
        .data = temp_gvox_model.data(),
        .size = temp_gvox_model_size,
    };
    void *i_config_ptr = nullptr;
    auto voxlap_config = GvoxVoxlapParseAdapterConfig{
        .size_x = 512,
        .size_y = 512,
        .size_z = 64,
        .make_solid = 1,
        .is_ace_of_spades = 1,
    };
    char const *gvox_model_type = "gvox_palette";
    if (gvox_model_path.has_extension()) {
        auto ext = gvox_model_path.extension();
        if (ext == ".vox") {
            gvox_model_type = "magicavoxel";
        } else if (ext == ".rle") {
            gvox_model_type = "gvox_run_length_encoding";
        } else if (ext == ".oct") {
            gvox_model_type = "gvox_octree";
        } else if (ext == ".glp") {
            gvox_model_type = "gvox_global_palette";
        } else if (ext == ".brk") {
            gvox_model_type = "gvox_brickmap";
        } else if (ext == ".gvr") {
            gvox_model_type = "gvox_raw";
        } else if (ext == ".vxl") {
            i_config_ptr = &voxlap_config;
            gvox_model_type = "voxlap";
        } else {
            return open_mesh_model();
        }
    } else {
        return open_mesh_model();
    }
    GvoxAdapterContext *i_ctx = gvox_create_adapter_context(gvox_ctx, gvox_get_input_adapter(gvox_ctx, "byte_buffer"), &i_config);
    GvoxAdapterContext *p_ctx = gvox_create_adapter_context(gvox_ctx, gvox_get_parse_adapter(gvox_ctx, gvox_model_type), i_config_ptr);
    result = load_gvox_data_from_parser(i_ctx, p_ctx, nullptr);
    gvox_destroy_adapter_context(i_ctx);
    gvox_destroy_adapter_context(p_ctx);
    return result;
}
