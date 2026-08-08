#pragma once

#include <daxa/daxa.hpp>
#include <daxa/utils/task_graph.hpp>
#include "async_pipeline_manager.hpp"

template <typename TaskHeadT, typename PushT, typename InfoT, typename PipelineT>
using TaskCallback = void(daxa::TaskInterface const &ti, typename PipelineT::PipelineT &pipeline, PushT &push, InfoT const &info);

struct NoTaskInfo {
};

template <typename TaskHeadT, typename PushT, typename InfoT, typename PipelineT>
struct Task : TaskHeadT {
    daxa::ShaderSource source;
    std::optional<uint32_t> required_subgroup_size{};
    std::vector<daxa::ShaderDefine> extra_defines{};
    TaskHeadT::Views views{};
    TaskCallback<TaskHeadT, PushT, InfoT, PipelineT> *callback_{};
    InfoT info{};
    // Not set by user
    // std::string_view name = TaskHeadT::NAME;
    std::shared_ptr<PipelineT> pipeline;
    daxa::TaskGraph *task_graph_ptr = nullptr;
    static void callback(daxa::TaskInterface const &ti, Task task) {
        auto push = PushT{};
        if (!task.pipeline->is_valid()) {
            return;
        }
        task.callback_(ti, task.pipeline->get(), push, task.info);
    }
    daxa::Task create() {
        return daxa::ComputeTask(TaskHeadT::NAME);
    }
};

template <typename TaskHeadT, typename PushT, typename InfoT>
using RayTracingTaskCallback = void(daxa::TaskInterface const &ti, daxa::RayTracingPipeline &pipeline, daxa::RayTracingShaderBindingTable const &shader_binding_table, PushT &push, InfoT const &info);

template <typename TaskHeadT, typename PushT, typename InfoT>
struct Task<TaskHeadT, PushT, InfoT, AsyncManagedRayTracingPipeline> : TaskHeadT {
    daxa::ShaderSource source;
    uint32_t max_ray_recursion_depth = 1;
    std::vector<daxa::ShaderDefine> extra_defines{};
    TaskHeadT::Views views{};
    RayTracingTaskCallback<TaskHeadT, PushT, InfoT> *callback_{};
    InfoT info{};
    // Not set by user
    std::shared_ptr<AsyncManagedRayTracingPipeline> pipeline;
    daxa::TaskGraph *task_graph_ptr = nullptr;
    static void callback(daxa::TaskInterface const &ti, Task task) {
        auto push = PushT{};
        if (!task.pipeline->is_valid()) {
            return;
        }
        task.callback_(ti, task.pipeline->get(), task.pipeline->sbt().table, push, task.info);
    }
    daxa::Task create() {
        return daxa::RayTracingTask(TaskHeadT::NAME);
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
    TaskHeadT::Views views{};
    TaskCallback<TaskHeadT, PushT, InfoT, AsyncManagedRasterPipeline> *callback_{};
    InfoT info{};
    // Not set by user
    std::shared_ptr<AsyncManagedRasterPipeline> pipeline;
    daxa::TaskGraph *task_graph_ptr = nullptr;
    static void callback(daxa::TaskInterface const &ti, Task task) {
        auto push = PushT{};
        if (!task.pipeline->is_valid()) {
            return;
        }
        task.callback_(ti, task.pipeline->get(), push, task.info);
    }
    daxa::Task create() {
        return daxa::RasterTask(TaskHeadT::NAME);
    }
};

template <typename TaskHeadT, typename PushT, typename InfoT>
using ComputeTask = Task<TaskHeadT, PushT, InfoT, AsyncManagedComputePipeline>;

template <typename TaskHeadT, typename PushT, typename InfoT>
using RayTracingTask = Task<TaskHeadT, PushT, InfoT, AsyncManagedRayTracingPipeline>;

template <typename TaskHeadT, typename PushT, typename InfoT>
using RasterTask = Task<TaskHeadT, PushT, InfoT, AsyncManagedRasterPipeline>;

namespace {
    template <typename PushT>
    constexpr auto push_constant_size() -> uint32_t {
        return static_cast<uint32_t>(((sizeof(PushT) & ~0x3u) + 7u) & ~7u);
    }

    template <typename PushT>
    void set_push_constant(daxa::TaskInterface const &ti, PushT push) {
        if constexpr (requires(PushT p) { p.uses; }) {
            push.uses = ti.attachment_shader_blob;
        }
        ti.recorder.push_constant(push);
    }

    template <typename PushT>
    void set_push_constant(daxa::TaskInterface const &ti, daxa::RenderCommandRecorder &render_recorder, PushT push) {
        if constexpr (requires(PushT p) { p.uses; }) {
            push.uses = ti.attachment_shader_blob;
            // ti.assign_attachment_shader_blob(reinterpret_cast<daxa::u8*>(&push.uses));
        }
        render_recorder.push_constant(push);
    }
} // namespace
