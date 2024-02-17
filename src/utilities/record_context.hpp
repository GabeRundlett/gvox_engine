#pragma once

#include <unordered_map>
#include <daxa/daxa.hpp>
#include <daxa/utils/task_graph.hpp>

#include "debug.hpp"
#include "async_pipeline_manager.hpp"

template <typename TaskHeadT, typename PushT, typename InfoT, typename PipelineT>
using TaskCallback = void(daxa::TaskInterface const &ti, typename PipelineT::PipelineT &pipeline, PushT &push, InfoT const &info);

struct NoTaskInfo {
};

template <typename PushT>
constexpr auto push_constant_size() -> uint32_t {
    return static_cast<uint32_t>(((sizeof(PushT) & ~0x3u) + 7u) & ~7u);
}

template <typename PushT>
void set_push_constant(daxa::TaskInterface const &ti, PushT push) {
    uint32_t offset = 0;
    if constexpr (push_constant_size<PushT>() != 0) {
        ti.recorder.push_constant(push);
        offset = push_constant_size<PushT>();
    }
    // ti.copy_task_head_to(&push.views);
    ti.recorder.push_constant_vptr({ti.attachment_shader_data.data(), ti.attachment_shader_data.size(), offset});
}

template <typename PushT>
void set_push_constant(daxa::TaskInterface const &ti, daxa::RenderCommandRecorder &render_recorder, PushT push) {
    uint32_t offset = 0;
    if constexpr (sizeof(PushT) >= 4) {
        render_recorder.push_constant(push);
        offset = sizeof(PushT);
    }
    render_recorder.push_constant_vptr({ti.attachment_shader_data.data(), ti.attachment_shader_data.size(), offset});
}

template <typename TaskHeadT, typename PushT, typename InfoT, typename PipelineT>
struct Task : TaskHeadT {
    daxa::ShaderSource source;
    std::vector<daxa::ShaderDefine> extra_defines{};
    TaskHeadT::AttachmentViews views{};
    TaskCallback<TaskHeadT, PushT, InfoT, PipelineT> *callback_{};
    InfoT info{};
    // Not set by user
    // std::string_view name = TaskHeadT::NAME;
    std::shared_ptr<PipelineT> pipeline;
    void callback(daxa::TaskInterface const &ti) {
        auto push = PushT{};
        if (!pipeline->is_valid()) {
            return;
        }
        callback_(ti, pipeline->get(), push, info);
    }
};

template <typename TaskHeadT, typename PushT, typename InfoT>
struct Task<TaskHeadT, PushT, InfoT, AsyncManagedRasterPipeline> : TaskHeadT {
    daxa::ShaderSource vert_source;
    daxa::ShaderSource frag_source;
    std::vector<daxa::RenderAttachment> color_attachments{};
    daxa::Optional<daxa::DepthTestInfo> depth_test{};
    daxa::RasterizerInfo raster{};
    std::vector<daxa::ShaderDefine> extra_defines{};
    TaskHeadT::AttachmentViews views{};
    TaskCallback<TaskHeadT, PushT, InfoT, AsyncManagedRasterPipeline> *callback_{};
    InfoT info{};
    // Not set by user
    // std::string_view name = TaskHeadT::NAME;
    std::shared_ptr<AsyncManagedRasterPipeline> pipeline;
    void callback(daxa::TaskInterface const &ti) {
        auto push = PushT{};
        // ti.copy_task_head_to(&push.uses);
        if (!pipeline->is_valid()) {
            return;
        }
        callback_(ti, pipeline->get(), push, info);
    }
};

template <typename TaskHeadT, typename PushT, typename InfoT>
using ComputeTask = Task<TaskHeadT, PushT, InfoT, AsyncManagedComputePipeline>;

template <typename TaskHeadT, typename PushT, typename InfoT>
using RasterTask = Task<TaskHeadT, PushT, InfoT, AsyncManagedRasterPipeline>;

struct TemporalBuffer {
    daxa::BufferId buffer_id;
    daxa::TaskBuffer task_buffer;
};

using TemporalBuffers = std::unordered_map<std::string, TemporalBuffer>;

struct RecordContext {
    daxa::Device device;
    daxa::TaskGraph task_graph;
    AsyncPipelineManager *pipeline_manager;
    daxa_u32vec2 render_resolution;
    daxa_u32vec2 output_resolution;

    daxa::TaskImageView task_swapchain_image;
    daxa::TaskImageView task_blue_noise_vec2_image;
    daxa::TaskImageView task_debug_texture;
    daxa::TaskImageView task_test_texture;
    daxa::TaskImageView task_test_texture2;
    daxa::TaskBufferView task_input_buffer;
    daxa::TaskBufferView task_globals_buffer;

    std::unordered_map<std::string, std::shared_ptr<AsyncManagedComputePipeline>> *compute_pipelines;
    std::unordered_map<std::string, std::shared_ptr<AsyncManagedRasterPipeline>> *raster_pipelines;
    TemporalBuffers *temporal_buffers;

    template <typename TaskHeadT, typename PushT, typename InfoT, typename PipelineT>
    auto find_or_add_pipeline(Task<TaskHeadT, PushT, InfoT, PipelineT> &task, std::string const &shader_id) {
        auto push_constant_size = static_cast<uint32_t>(::push_constant_size<PushT>() + TaskHeadT::attachment_shader_data_size());
        if constexpr (std::is_same_v<PipelineT, AsyncManagedComputePipeline>) {
            auto pipe_iter = compute_pipelines->find(shader_id);
            if (pipe_iter == compute_pipelines->end()) {
                task.extra_defines.push_back({std::string{TaskHeadT::name()} + "Shader", "1"});
                auto emplace_result = compute_pipelines->emplace(
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
            auto pipe_iter = raster_pipelines->find(shader_id);
            // TODO: if we found a pipeline, but it has differing info such as attachments or raster info,
            // we should destroy that old one and create a new one.
            if (pipe_iter == raster_pipelines->end()) {
                task.extra_defines.push_back({std::string{TaskHeadT::name()} + "Shader", "1"});
                auto emplace_result = raster_pipelines->emplace(
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

    template <typename TaskHeadT, typename PushT, typename InfoT, typename PipelineT>
    void add(Task<TaskHeadT, PushT, InfoT, PipelineT> &&task) {
        auto shader_id = std::string{TaskHeadT::name()};
        for (auto const &define : task.extra_defines) {
            shader_id.append(define.name);
            shader_id.append(define.value);
        }
        auto pipe_iter = find_or_add_pipeline<TaskHeadT, PushT, InfoT, PipelineT>(task, shader_id);
        task.pipeline = pipe_iter->second;
        task_graph.add_task(std::move(task));
    }

    auto find_or_add_temporal_buffer(daxa::BufferInfo const &info) -> TemporalBuffer {
        auto id = std::string{info.name.view()};
        auto iter = temporal_buffers->find(id);

        if (iter == temporal_buffers->end()) {
            auto result = TemporalBuffer{};
            result.buffer_id = device.create_buffer(info);
            result.task_buffer = daxa::TaskBuffer(daxa::TaskBufferInfo{.initial_buffers = {.buffers = std::array{result.buffer_id}}, .name = id});
            auto emplace_result = temporal_buffers->emplace(id, result);
            iter = emplace_result.first;
        } else {
            auto existing_info = device.info_buffer(iter->second.buffer_id).value();
            if (existing_info.size != info.size) {
                debug_utils::Console::add_log(std::format("TemporalBuffer \"{}\" recreated with bad size... This should NEVER happen!!!", id));
            }
        }
        task_graph.use_persistent_buffer(iter->second.task_buffer);

        return iter->second;
    }
};
