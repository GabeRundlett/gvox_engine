#pragma once

#include <core.inl>
#include <gvox/gvox.h>
#include <application/ui.hpp>
#include <future>

struct GvoxModelData {
    size_t size = 0;
    uint8_t *ptr = nullptr;
};

struct VoxelModelLoader {
    GvoxContext *gvox_ctx;
    GpuContext *gpu_context;

    bool has_model = false;
    std::future<GvoxModelData> gvox_model_data_future;
    GvoxModelData gvox_model_data;
    bool should_upload_gvox_model = false;
    bool model_is_loading = false;
    bool model_is_ready = false;
    std::filesystem::path gvox_model_path{};

    daxa::BufferId gvox_model_buffer;
    daxa::BufferId prev_gvox_model_buffer{};
    daxa::TaskBuffer task_gvox_model_buffer{{.name = "task_gvox_model_buffer"}};

    void create(GpuContext &gpu_context);
    void destroy();

    void update(AppUi &ui);
    void upload_model();
    auto load_gvox_data_from_parser(GvoxAdapterContext *i_ctx, GvoxAdapterContext *p_ctx, GvoxRegionRange const *region_range) -> GvoxModelData;
    auto load_gvox_data() -> GvoxModelData;
    auto voxelize_mesh_model() -> GvoxModelData;
};
