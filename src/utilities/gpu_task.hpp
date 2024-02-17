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

namespace {
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
} // namespace
