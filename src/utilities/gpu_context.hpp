#pragma once

#include <daxa/daxa.hpp>
#include <daxa/utils/task_graph.hpp>
#include "async_pipeline_manager.hpp"
#include "gpu_task.hpp"
#include <any>

struct TemporalBuffer {
    daxa::ExternalTaskBuffer task_resource;
    operator daxa::ExternalTaskBuffer &() { return task_resource; }
    operator daxa::BufferId() const { return !task_resource.is_valid() || task_resource.info().buffer.is_empty() ? daxa::BufferId{} : task_resource.info().buffer; }
    operator daxa::ExternalTaskBuffer const &() const { return task_resource; }
    operator daxa::TaskBufferView() { return task_resource; }
};
struct TemporalImage {
    daxa::ExternalTaskImage task_resource;
    operator daxa::ExternalTaskImage &() { return task_resource; }
    operator daxa::ImageId() const { return !task_resource.is_valid() || task_resource.info().image.is_empty() ? daxa::ImageId{} : task_resource.info().image; }
    operator daxa::ExternalTaskImage const &() const { return task_resource; }
    operator daxa::TaskImageView() { return task_resource; }
};

using TemporalBuffers = std::unordered_map<std::string, TemporalBuffer>;
using TemporalImages = std::unordered_map<std::string, TemporalImage>;

struct GpuContext {
    daxa::Instance daxa_instance;
    daxa::Device device;

    daxa::Swapchain swapchain;
    daxa::ImageId swapchain_image{};
    daxa::ExternalTaskImage task_swapchain_image{daxa::ExternalTaskImageInfo{.is_swapchain_image = true}};

    daxa::ImageId value_noise_image;
    daxa::ImageViewId value_noise_image_view;
    daxa::ImageId blue_noise_vec2_image;
    daxa::ImageId debug_texture;
    daxa::ImageId test_texture;
    daxa::ImageId test_texture2;

    daxa::BufferId input_buffer;

    daxa::SamplerId sampler_nnc;
    daxa::SamplerId sampler_lnc;
    daxa::SamplerId sampler_llc;
    daxa::SamplerId sampler_llr;

    daxa::ExternalTaskImage task_value_noise_image{{.name = "task_value_noise_image"}};
    daxa::TaskImageView task_value_noise_image_view{};
    daxa::ExternalTaskImage task_blue_noise_vec2_image{{.name = "task_blue_noise_vec2_image"}};
    daxa::ExternalTaskImage task_debug_texture{{.name = "task_debug_texture"}};
    daxa::ExternalTaskImage task_test_texture{{.name = "task_test_texture"}};
    daxa::ExternalTaskImage task_test_texture2{{.name = "task_test_texture2"}};

    daxa::ExternalTaskBuffer task_input_buffer{{.name = "task_input_buffer"}};

    std::shared_ptr<AsyncPipelineManager> pipeline_manager;
    TemporalBuffers temporal_buffers;
    TemporalImages temporal_images;

    daxa::TaskGraph startup_task_graph;
    daxa::TaskGraph frame_task_graph;
    daxa_u32vec2 render_resolution;
    daxa_u32vec2 output_resolution;

    GpuContext();
    ~GpuContext();

    void create_swapchain(daxa::SwapchainInfo const &info);

    void use_resources();
    void update_seeded_value_noise(uint64_t seed);

    auto find_or_add_temporal_buffer(daxa::BufferInfo const &info) -> TemporalBuffer;
    auto find_or_add_temporal_image(daxa::ImageInfo const &info) -> TemporalImage;
    void remove_temporal_buffer(std::string const &id);
    void remove_temporal_image(std::string const &id);
    void remove_temporal_buffer(daxa::BufferId id);
    void remove_temporal_image(daxa::ImageId id);

    std::unordered_map<std::string, std::shared_ptr<AsyncManagedComputePipeline>> compute_pipelines;
    std::unordered_map<std::string, std::shared_ptr<AsyncManagedRayTracingPipeline>> ray_tracing_pipelines;
    std::unordered_map<std::string, std::shared_ptr<AsyncManagedRasterPipeline>> raster_pipelines;

    std::vector<std::any> task_states;

    template <typename TaskHeadT, typename PushT, typename InfoT, typename PipelineT>
    auto find_or_add_pipeline(Task<TaskHeadT, PushT, InfoT, PipelineT> &task, std::string const &shader_id) {
        auto push_constant_size = static_cast<uint32_t>(::push_constant_size<PushT>());
        if constexpr (std::is_same_v<PipelineT, AsyncManagedComputePipeline>) {
            auto pipe_iter = compute_pipelines.find(shader_id);
            if (pipe_iter == compute_pipelines.end()) {
                task.extra_defines.push_back({std::string{TaskHeadT::NAME} + "Shader", "1"});
                auto emplace_result = compute_pipelines.emplace(
                    shader_id,
                    std::make_shared<AsyncManagedComputePipeline>(pipeline_manager->add_compute_pipeline({
                        .source = task.source,
                        .defines = task.extra_defines,
                        .required_subgroup_size = task.required_subgroup_size,
                        .push_constant_size = push_constant_size,
                        .name = std::string{TaskHeadT::NAME},
                    })));
                pipe_iter = emplace_result.first;
            }
            return pipe_iter;
        } else if constexpr (std::is_same_v<PipelineT, AsyncManagedRayTracingPipeline>) {
            auto pipe_iter = ray_tracing_pipelines.find(shader_id);
            if (pipe_iter == ray_tracing_pipelines.end()) {
                task.extra_defines.push_back({std::string{TaskHeadT::NAME} + "Shader", "1"});
                auto emplace_result = ray_tracing_pipelines.emplace(
                    shader_id,
                    std::make_shared<AsyncManagedRayTracingPipeline>(pipeline_manager->add_ray_tracing_pipeline({
                        .ray_gen_infos = {daxa::ShaderCompileInfo2{
                            .source = task.source,
                            .defines = task.extra_defines,
                        }},
                        .intersection_infos = {daxa::ShaderCompileInfo2{
                            .source = task.source,
                            .defines = task.extra_defines,
                        }},
                        .closest_hit_infos = {daxa::ShaderCompileInfo2{
                            .source = task.source,
                            .defines = task.extra_defines,
                        }},
                        .miss_hit_infos = {daxa::ShaderCompileInfo2{
                            .source = task.source,
                            .defines = task.extra_defines,
                        }},
                        // Groups are in order of their shader indices.
                        // NOTE: The order of the groups is important! raygen, miss, hit, callable
                        .shader_groups_infos = {
                            daxa::RayTracingShaderGroupInfo{
                                .type = daxa::ShaderGroup::GENERAL,
                                .general_shader_index = 0,
                            },
                            daxa::RayTracingShaderGroupInfo{
                                .type = daxa::ShaderGroup::GENERAL,
                                .general_shader_index = 3,
                            },
                            daxa::RayTracingShaderGroupInfo{
                                .type = daxa::ShaderGroup::PROCEDURAL_HIT_GROUP,
                                .closest_hit_shader_index = 2,
                                .intersection_shader_index = 1,
                            },
                        },
                        .max_ray_recursion_depth = task.max_ray_recursion_depth,
                        .push_constant_size = push_constant_size,
                        .name = std::string{TaskHeadT::NAME},
                    })));
                pipe_iter = emplace_result.first;
            }
            return pipe_iter;
        } else if constexpr (std::is_same_v<PipelineT, AsyncManagedRasterPipeline>) {
            auto pipe_iter = raster_pipelines.find(shader_id);
            // TODO: if we found a pipeline, but it has differing info such as attachments or raster info,
            // we should destroy that old one and create a new one.
            if (pipe_iter == raster_pipelines.end()) {
                task.extra_defines.push_back({std::string{TaskHeadT::NAME} + "Shader", "1"});
                auto emplace_result = raster_pipelines.emplace(
                    shader_id,
                    std::make_shared<AsyncManagedRasterPipeline>(pipeline_manager->add_raster_pipeline({
                        // .mesh_shader_info = daxa::holds_alternative<daxa::Monostate>(task.mesh_source)
                        //                         ? daxa::Optional<daxa::ShaderCompileInfo2>{}
                        //                         : daxa::Optional<daxa::ShaderCompileInfo2>{daxa::ShaderCompileInfo2{
                        //                               .source = task.mesh_source,
                        //                               .defines = task.extra_defines,
                        //                               .required_subgroup_size = task.required_subgroup_size,
                        //                           }},
                        .vertex_shader_info = daxa::holds_alternative<daxa::Monostate>(task.vert_source)
                                                  ? daxa::Optional<daxa::ShaderCompileInfo2>{}
                                                  : daxa::Optional<daxa::ShaderCompileInfo2>{daxa::ShaderCompileInfo2{
                                                        .source = task.vert_source,
                                                        .defines = task.extra_defines,
                                                    }},
                        .fragment_shader_info = daxa::ShaderCompileInfo2{
                            .source = task.frag_source,
                            .defines = task.extra_defines,
                        },
                        .color_attachments = task.color_attachments,
                        .depth_test = task.depth_test,
                        .raster = task.raster,
                        .push_constant_size = push_constant_size,
                        .name = std::string{TaskHeadT::NAME},
                    })));
                pipe_iter = emplace_result.first;
            }
            return pipe_iter;
        }
    }

    template <typename TaskHeadT, typename PushT, typename InfoT, typename PipelineT>
    void add(Task<TaskHeadT, PushT, InfoT, PipelineT> &&task) {
        using MetaTaskT = Task<TaskHeadT, PushT, InfoT, PipelineT>;
        auto shader_id = std::string{TaskHeadT::NAME};
        for (auto const &define : task.extra_defines) {
            shader_id.append(define.name);
            shader_id.append(define.value);
        }
        shader_id.append("_");
        if constexpr (requires(MetaTaskT t) { t.required_subgroup_size; }) {
            if (task.required_subgroup_size.has_value()) {
                shader_id.append(std::to_string(*task.required_subgroup_size));
            }
        }
        shader_id.append("_");
        if constexpr (requires(MetaTaskT t) { t.max_ray_recursion_depth; }) {
            shader_id.append(std::to_string(task.max_ray_recursion_depth));
        }
        auto pipe_iter = find_or_add_pipeline<TaskHeadT, PushT, InfoT, PipelineT>(task, shader_id);
        task.pipeline = pipe_iter->second;
        task_states.push_back(std::make_any<MetaTaskT>(task));
        auto *task_ptr = std::any_cast<MetaTaskT>(&task_states.back());
        if (task.task_graph_ptr == nullptr) {
            task.task_graph_ptr = &frame_task_graph;
        }
        task.task_graph_ptr->add_task(task.create().template uses_head<TaskHeadT>().head_views(task.views).executes([task_ptr](daxa::TaskInterface const &ti) {
            MetaTaskT::callback(ti, *task_ptr);
        }));
    }
};
