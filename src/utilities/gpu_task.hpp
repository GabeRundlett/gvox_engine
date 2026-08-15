#pragma once

#include <daxa/daxa.hpp>
#include <daxa/utils/task_graph.hpp>
#include <renderer/pipeline_manager.hpp>

// NOTE: `pipeline` is a raw, non-owning pointer into storage owned by
// GpuContext (see gpu_context.hpp). It must stay at a stable address: task
// graph closures capture the Task by pointer and dereference `pipeline` every
// frame, and hot-reload assigns a new pipeline *through* that pointer so
// already-recorded closures pick it up without re-recording the graph.

template <typename TaskHeadT, typename PushT, typename InfoT, typename PipelineT>
using TaskCallback = void(daxa::TaskInterface const &ti, PipelineT &pipeline, PushT &push, InfoT const &info);

struct NoTaskInfo {
};

template <typename TaskHeadT, typename PushT, typename InfoT, typename PipelineT>
struct Task : TaskHeadT {
    const char *source;
    std::optional<uint32_t> required_subgroup_size{};
    Vec<ShaderDefine> extra_defines{};
    TaskHeadT::Views views{};
    TaskCallback<TaskHeadT, PushT, InfoT, PipelineT> *callback_{};
    InfoT info{};
    // Not set by user
    PipelineT *pipeline{};
    daxa::TaskGraph *task_graph_ptr = nullptr;
    static void callback(daxa::TaskInterface const &ti, Task task) {
        auto push = PushT{};
        if (task.pipeline == nullptr || !task.pipeline->is_valid()) {
            return;
        }
        task.callback_(ti, *task.pipeline, push, task.info);
    }
    daxa::Task create() {
        return daxa::ComputeTask(TaskHeadT::NAME);
    }
};

template <typename TaskHeadT, typename PushT, typename InfoT>
using RayTracingTaskCallback = void(daxa::TaskInterface const &ti, daxa::RayTracingPipeline &pipeline, daxa::RayTracingShaderBindingTable const &shader_binding_table, PushT &push, InfoT const &info);

// A ray tracing pipeline plus its lazily-created shader binding table.
// daxa::RayTracingPipelineInfo doesn't carry an SBT, and the buffer returned by
// create_default_sbt() must outlive its use, so we own it alongside the pipeline.
struct RayTracingPipelineAndSbt {
    daxa::RayTracingPipeline pipeline{};
    daxa::Optional<daxa::RayTracingPipeline::SbtPair> sbt_storage{};

    auto is_valid() -> bool { return pipeline.is_valid(); }
    auto sbt() -> daxa::RayTracingPipeline::SbtPair const & {
        if (!sbt_storage.has_value()) {
            sbt_storage = pipeline.create_default_sbt();
        }
        return sbt_storage.value();
    }
    // Called after a hot-reload replaces `pipeline`: the old SBT refers to the
    // old pipeline's shader groups and must not be reused.
    void recreate_sbt() {
        if (sbt_storage.has_value()) {
            sbt_storage = pipeline.create_default_sbt();
        }
    }
};

template <typename TaskHeadT, typename PushT, typename InfoT>
struct Task<TaskHeadT, PushT, InfoT, RayTracingPipelineAndSbt> : TaskHeadT {
    const char *source;
    uint32_t max_ray_recursion_depth = 1;
    Vec<ShaderDefine> extra_defines{};
    TaskHeadT::Views views{};
    RayTracingTaskCallback<TaskHeadT, PushT, InfoT> *callback_{};
    InfoT info{};
    // Not set by user
    RayTracingPipelineAndSbt *pipeline{};
    daxa::TaskGraph *task_graph_ptr = nullptr;
    static void callback(daxa::TaskInterface const &ti, Task task) {
        auto push = PushT{};
        if (task.pipeline == nullptr || !task.pipeline->is_valid()) {
            return;
        }
        task.callback_(ti, task.pipeline->pipeline, task.pipeline->sbt().table, push, task.info);
    }
    daxa::Task create() {
        return daxa::RayTracingTask(TaskHeadT::NAME);
    }
};

template <typename TaskHeadT, typename PushT, typename InfoT>
struct Task<TaskHeadT, PushT, InfoT, daxa::RasterPipeline> : TaskHeadT {
    const char *vert_source;
    const char *frag_source;
    Vec<daxa::RenderAttachment> color_attachments{};
    daxa::Optional<daxa::DepthTestInfo> depth_test{};
    daxa::RasterizerInfo raster{};
    Vec<ShaderDefine> extra_defines{};
    TaskHeadT::Views views{};
    TaskCallback<TaskHeadT, PushT, InfoT, daxa::RasterPipeline> *callback_{};
    InfoT info{};
    // Not set by user
    daxa::RasterPipeline *pipeline{};
    daxa::TaskGraph *task_graph_ptr = nullptr;
    static void callback(daxa::TaskInterface const &ti, Task task) {
        auto push = PushT{};
        if (task.pipeline == nullptr || !task.pipeline->is_valid()) {
            return;
        }
        task.callback_(ti, *task.pipeline, push, task.info);
    }
    daxa::Task create() {
        return daxa::RasterTask(TaskHeadT::NAME);
    }
};

template <typename TaskHeadT, typename PushT, typename InfoT>
using ComputeTask = Task<TaskHeadT, PushT, InfoT, daxa::ComputePipeline>;

template <typename TaskHeadT, typename PushT, typename InfoT>
using RayTracingTask = Task<TaskHeadT, PushT, InfoT, RayTracingPipelineAndSbt>;

template <typename TaskHeadT, typename PushT, typename InfoT>
using RasterTask = Task<TaskHeadT, PushT, InfoT, daxa::RasterPipeline>;

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
