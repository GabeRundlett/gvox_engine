#pragma once

#include <daxa/daxa.hpp>
#include <daxa/utils/task_graph.hpp>
#include <renderer/pipeline_manager.hpp>
#include <base/hash_map.hpp>
#include <base/str.hpp>
#include <base/vec.hpp>
#include <base/profiler.hpp>
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

using TemporalBuffers = HashMap<Str, TemporalBuffer>;
using TemporalImages = HashMap<Str, TemporalImage>;

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

    PipelineManager *pipeline_manager = nullptr;
    TemporalBuffers temporal_buffers;
    TemporalImages temporal_images;

    daxa::TaskGraph startup_task_graph;
    daxa::TaskGraph frame_task_graph;
    daxa_u32vec2 render_resolution;
    daxa_u32vec2 output_resolution;
    
    daxa::TimelineQueryPool timeline_query_pool;
    uint32_t timeline_query_frame_offset = 0;
    uint32_t timeline_query_index = 0;
    uint32_t timeline_query_index_begin = 0;
    uint32_t timeline_query_index_count[FRAMES_IN_FLIGHT] = {};
    Vec<Str> timestamp_names_storage[FRAMES_IN_FLIGHT] = {};
    Vec<Str> timestamp_names;
    HashMap<Str, const char*> dynamic_timestamp_name_storage;
    std::vector<daxa_u64> timeline_query_results;

    GpuContext();
    ~GpuContext();

    void create_swapchain(daxa::SwapchainInfo const &info);

    void update_timestamps();
    void finalize_timestamps();
    bool supports_timestamps() const { return device.properties().limits.timestamp_period > 0 && device.properties().limits.timestamp_compute_and_graphics != 0; }
    void get_timestamps(Vec<struct ProfileTimestamp> &out_timestamps);
    void begin_task_timestamp(const daxa::TaskInterface& ti);
    void end_task_timestamp(const daxa::TaskInterface& ti);

    void use_resources();
    void update_seeded_value_noise(uint64_t seed);

    auto find_or_add_temporal_buffer(daxa::BufferInfo const &info) -> TemporalBuffer;
    auto find_or_add_temporal_image(daxa::ImageInfo const &info) -> TemporalImage;
    void remove_temporal_buffer(char const *id);
    void remove_temporal_image(char const *id);
    void remove_temporal_buffer(daxa::BufferId id);
    void remove_temporal_image(daxa::ImageId id);

    // Pipelines are individually heap-allocated (never stored by value in a
    // container) so their addresses stay stable: recorded task-graph closures
    // hold raw pointers to them, and hot-reload overwrites them in place.
    // Owned here; freed in ~GpuContext.
    HashMap<Str, daxa::ComputePipeline *> compute_pipelines;
    HashMap<Str, RayTracingPipelineAndSbt *> ray_tracing_pipelines;
    HashMap<Str, daxa::RasterPipeline *> raster_pipelines;

    Vec<std::any> task_states;

    // Builds the key that identifies a unique pipeline variant: task name plus
    // every define, subgroup size and recursion depth that affects compilation.
    template <typename TaskHeadT, typename PushT, typename InfoT, typename PipelineT>
    static auto make_shader_id(Task<TaskHeadT, PushT, InfoT, PipelineT> const &task) -> Str {
        using MetaTaskT = Task<TaskHeadT, PushT, InfoT, PipelineT>;
        auto shader_id = Str{TaskHeadT::NAME};
        for (auto const &define : task.extra_defines) {
            shader_id.append(define.name);
            shader_id.append(define.value);
        }
        shader_id.append("_");
        if constexpr (requires(MetaTaskT t) { t.required_subgroup_size; }) {
            if (task.required_subgroup_size.has_value()) {
                shader_id.append(static_cast<unsigned long long>(*task.required_subgroup_size));
            }
        }
        shader_id.append("_");
        if constexpr (requires(MetaTaskT t) { t.max_ray_recursion_depth; }) {
            shader_id.append(static_cast<unsigned long long>(task.max_ray_recursion_depth));
        }
        return shader_id;
    }

    // Registers the pipeline for this task with the pipeline manager (no
    // compilation happens here -- that's compile_all_shaders/create_all_pipelines,
    // called once after the whole task graph has been recorded) and returns the
    // stable pointer the task will dereference.
    template <typename TaskHeadT, typename PushT, typename InfoT, typename PipelineT>
    auto find_or_add_pipeline(Task<TaskHeadT, PushT, InfoT, PipelineT> &task, Str const &shader_id) -> PipelineT * {
        PROFILE_FUNC();

        auto push_constant_size = static_cast<uint32_t>(::push_constant_size<PushT>());
        if constexpr (std::is_same_v<PipelineT, daxa::ComputePipeline>) {
            if (auto **existing = compute_pipelines.get(shader_id)) {
                return *existing;
            }
            task.extra_defines.push_back({Str{TaskHeadT::NAME} + "Shader", "1"});
            auto *pipeline = new daxa::ComputePipeline{};
            compute_pipelines.set(shader_id, pipeline);
            register_pipeline(
                pipeline_manager,
                ComputePipelineCompileInfo{
                    .out_pipeline = pipeline,
                    .source_path = task.source,
                    .defines = task.extra_defines,
                    .required_subgroup_size = task.required_subgroup_size.has_value() ? static_cast<int>(*task.required_subgroup_size) : -1,
                    .push_constant_size = push_constant_size,
                    .name = Str{TaskHeadT::NAME},
                });
            return pipeline;
        } else if constexpr (std::is_same_v<PipelineT, RayTracingPipelineAndSbt>) {
            if (auto **existing = ray_tracing_pipelines.get(shader_id)) {
                return *existing;
            }
            task.extra_defines.push_back({Str{TaskHeadT::NAME} + "Shader", "1"});
            auto *pipeline = new RayTracingPipelineAndSbt{};
            ray_tracing_pipelines.set(shader_id, pipeline);

            auto stage_info = ShaderCompileInfo{.source_path = task.source, .defines = task.extra_defines};
            auto rt_info = RayTracingPipelineCompileInfo{.out_pipeline = &pipeline->pipeline};
            rt_info.ray_gen_infos.push_back(stage_info);
            rt_info.intersection_infos.push_back(stage_info);
            rt_info.closest_hit_infos.push_back(stage_info);
            rt_info.miss_hit_infos.push_back(stage_info);
            // Groups are in order of their shader indices.
            // NOTE: The order of the groups is important! raygen, miss, hit, callable
            rt_info.shader_groups_infos.push_back(daxa::RayTracingShaderGroupInfo{
                .type = daxa::ShaderGroup::GENERAL,
                .general_shader_index = 0,
            });
            rt_info.shader_groups_infos.push_back(daxa::RayTracingShaderGroupInfo{
                .type = daxa::ShaderGroup::GENERAL,
                .general_shader_index = 3,
            });
            rt_info.shader_groups_infos.push_back(daxa::RayTracingShaderGroupInfo{
                .type = daxa::ShaderGroup::PROCEDURAL_HIT_GROUP,
                .closest_hit_shader_index = 2,
                .intersection_shader_index = 1,
            });
            rt_info.max_ray_recursion_depth = task.max_ray_recursion_depth;
            rt_info.push_constant_size = push_constant_size;
            rt_info.name = Str{TaskHeadT::NAME};
            register_pipeline(pipeline_manager, rt_info);
            return pipeline;
        } else if constexpr (std::is_same_v<PipelineT, daxa::RasterPipeline>) {
            // TODO: if we found a pipeline, but it has differing info such as attachments or raster info,
            // we should destroy that old one and create a new one.
            if (auto **existing = raster_pipelines.get(shader_id)) {
                return *existing;
            }
            task.extra_defines.push_back({Str{TaskHeadT::NAME} + "Shader", "1"});
            auto *pipeline = new daxa::RasterPipeline{};
            raster_pipelines.set(shader_id, pipeline);

            auto raster_info = RasterPipelineCompileInfo{.out_pipeline = pipeline};
            if (task.vert_source != nullptr) {
                raster_info.vert_info = ShaderCompileInfo{.source_path = task.vert_source, .defines = task.extra_defines};
            }
            raster_info.frag_info = ShaderCompileInfo{.source_path = task.frag_source, .defines = task.extra_defines};
            raster_info.color_attachments = task.color_attachments;
            raster_info.depth_test = task.depth_test;
            raster_info.raster = task.raster;
            raster_info.push_constant_size = push_constant_size;
            raster_info.name = Str{TaskHeadT::NAME};
            register_pipeline(pipeline_manager, raster_info);
            return pipeline;
        }
    }

    template <typename TaskHeadT, typename PushT, typename InfoT, typename PipelineT>
    void add(Task<TaskHeadT, PushT, InfoT, PipelineT> &&task) {
        using MetaTaskT = Task<TaskHeadT, PushT, InfoT, PipelineT>;
        auto shader_id = make_shader_id(task);
        task.pipeline = find_or_add_pipeline<TaskHeadT, PushT, InfoT, PipelineT>(task, shader_id);
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
