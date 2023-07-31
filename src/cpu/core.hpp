#pragma once

#include <memory>

#include <daxa/daxa.hpp>
#include <daxa/utils/pipeline_manager.hpp>
#include <daxa/utils/imgui.hpp>
#include <daxa/utils/task_graph.hpp>
#include <daxa/utils/math_operators.hpp>
using namespace daxa::math_operators;

using BDA = daxa::BufferDeviceAddress;

static inline constexpr usize FRAMES_IN_FLIGHT = 1;

struct RecordContext {
    daxa::Device device;
    daxa::PipelineManager pipeline_manager;
    daxa::TaskGraph task_graph;
    u32vec2 render_resolution;

    daxa::TaskBuffer *task_input_buffer;
    daxa::TaskBuffer *task_globals_buffer;
};
