#pragma once

#include <unordered_map>
#include <daxa/daxa.hpp>
#include <daxa/utils/task_graph.hpp>

#include "debug.hpp"
#include "gpu_context.hpp"

struct RecordContext {
    daxa::TaskGraph task_graph;
    daxa_u32vec2 render_resolution;
    daxa_u32vec2 output_resolution;

    daxa::TaskImageView task_swapchain_image;
    GpuContext *gpu_context;

    template <typename TaskHeadT, typename PushT, typename InfoT, typename PipelineT>
    void add(Task<TaskHeadT, PushT, InfoT, PipelineT> &&task) {
        auto shader_id = std::string{TaskHeadT::name()};
        for (auto const &define : task.extra_defines) {
            shader_id.append(define.name);
            shader_id.append(define.value);
        }
        auto pipe_iter = gpu_context->find_or_add_pipeline<TaskHeadT, PushT, InfoT, PipelineT>(task, shader_id);
        task.pipeline = pipe_iter->second;
        task_graph.add_task(std::move(task));
    }
};
