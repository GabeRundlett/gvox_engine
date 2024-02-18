#pragma once

#include <daxa/daxa.hpp>
#include <daxa/utils/task_graph.hpp>
#include "async_pipeline_manager.hpp"
#include "gpu_task.hpp"

struct TemporalBuffer {
    daxa::BufferId resource_id;
    daxa::TaskBuffer task_resource;
};
struct TemporalImage {
    daxa::ImageId resource_id;
    daxa::TaskImage task_resource;
};

using TemporalBuffers = std::unordered_map<std::string, TemporalBuffer>;
using TemporalImages = std::unordered_map<std::string, TemporalImage>;

struct RecordContext;

struct GpuContext {
    daxa::Instance daxa_instance;
    daxa::Device device;

    daxa::ImageId value_noise_image;
    daxa::ImageId blue_noise_vec2_image;
    daxa::ImageId debug_texture;
    daxa::ImageId test_texture;
    daxa::ImageId test_texture2;

    daxa::BufferId input_buffer;
    daxa::BufferId output_buffer;
    daxa::BufferId staging_output_buffer;
    daxa::BufferId globals_buffer;

    daxa::SamplerId sampler_nnc;
    daxa::SamplerId sampler_lnc;
    daxa::SamplerId sampler_llc;
    daxa::SamplerId sampler_llr;

    daxa::TaskImage task_value_noise_image{{.name = "task_value_noise_image"}};
    daxa::TaskImage task_blue_noise_vec2_image{{.name = "task_blue_noise_vec2_image"}};
    daxa::TaskImage task_debug_texture{{.name = "task_debug_texture"}};
    daxa::TaskImage task_test_texture{{.name = "task_test_texture"}};
    daxa::TaskImage task_test_texture2{{.name = "task_test_texture2"}};

    daxa::TaskBuffer task_input_buffer{{.name = "task_input_buffer"}};
    daxa::TaskBuffer task_output_buffer{{.name = "task_output_buffer"}};
    daxa::TaskBuffer task_staging_output_buffer{{.name = "task_staging_output_buffer"}};
    daxa::TaskBuffer task_globals_buffer{{.name = "task_globals_buffer"}};

    std::shared_ptr<AsyncPipelineManager> pipeline_manager;
    TemporalBuffers temporal_buffers;
    TemporalImages temporal_images;

    GpuContext();
    ~GpuContext();

    void use_resources(RecordContext &record_ctx);
    void update_seeded_value_noise(daxa::Device &device, uint64_t seed);

    auto find_or_add_temporal_buffer(daxa::BufferInfo const &info) -> TemporalBuffer;
    auto find_or_add_temporal_image(daxa::ImageInfo const &info) -> TemporalImage;
    void remove_temporal_buffer(std::string const &id);
    void remove_temporal_image(std::string const &id);
    void remove_temporal_buffer(daxa::BufferId id);
    void remove_temporal_image(daxa::ImageId id);

    std::unordered_map<std::string, std::shared_ptr<AsyncManagedComputePipeline>> compute_pipelines;
    std::unordered_map<std::string, std::shared_ptr<AsyncManagedRasterPipeline>> raster_pipelines;

    template <typename TaskHeadT, typename PushT, typename InfoT, typename PipelineT>
    auto find_or_add_pipeline(Task<TaskHeadT, PushT, InfoT, PipelineT> &task, std::string const &shader_id) {
        auto push_constant_size = static_cast<uint32_t>(::push_constant_size<PushT>() + TaskHeadT::attachment_shader_data_size());
        if constexpr (std::is_same_v<PipelineT, AsyncManagedComputePipeline>) {
            auto pipe_iter = compute_pipelines.find(shader_id);
            if (pipe_iter == compute_pipelines.end()) {
                task.extra_defines.push_back({std::string{TaskHeadT::name()} + "Shader", "1"});
                auto emplace_result = compute_pipelines.emplace(
                    shader_id,
                    std::make_shared<AsyncManagedComputePipeline>(pipeline_manager->add_compute_pipeline({
                        .shader_info = {
                            .source = task.source,
                            .compile_options = {.defines = task.extra_defines},
                        },
                        .push_constant_size = push_constant_size,
                        .name = std::string{TaskHeadT::name()},
                    })));
                pipe_iter = emplace_result.first;
            }
            return pipe_iter;
        } else if constexpr (std::is_same_v<PipelineT, AsyncManagedRasterPipeline>) {
            auto pipe_iter = raster_pipelines.find(shader_id);
            // TODO: if we found a pipeline, but it has differing info such as attachments or raster info,
            // we should destroy that old one and create a new one.
            if (pipe_iter == raster_pipelines.end()) {
                task.extra_defines.push_back({std::string{TaskHeadT::name()} + "Shader", "1"});
                auto emplace_result = raster_pipelines.emplace(
                    shader_id,
                    std::make_shared<AsyncManagedRasterPipeline>(pipeline_manager->add_raster_pipeline({
                        .vertex_shader_info = daxa::ShaderCompileInfo{
                            .source = task.vert_source,
                            .compile_options = {.defines = task.extra_defines},
                        },
                        .fragment_shader_info = daxa::ShaderCompileInfo{
                            .source = task.frag_source,
                            .compile_options = {.defines = task.extra_defines},
                        },
                        .color_attachments = task.color_attachments,
                        .depth_test = task.depth_test,
                        .raster = task.raster,
                        .push_constant_size = push_constant_size,
                        .name = std::string{TaskHeadT::name()},
                    })));
                pipe_iter = emplace_result.first;
            }
            return pipe_iter;
        }
    }
};
