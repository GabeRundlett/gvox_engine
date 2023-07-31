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
    daxa::TaskGraph task_graph;
    u32vec2 render_resolution;

    daxa::TaskBufferView task_input_buffer;
    daxa::TaskBufferView task_globals_buffer;
};

struct PingPongImage {
    struct Resources {
        Resources() = default;
        Resources(Resources const &) = delete;
        Resources(Resources &&) = delete;
        Resources &operator=(Resources const &) = delete;
        Resources &operator=(Resources &&other) {
            std::swap(this->device, other.device);
            std::swap(this->image_a, other.image_a);
            std::swap(this->image_b, other.image_b);
            return *this;
        }

        daxa::Device device{};
        daxa::ImageId image_a{};
        daxa::ImageId image_b{};

        ~Resources() {
            if (!image_a.is_empty()) {
                device.destroy_image(image_a);
                device.destroy_image(image_b);
            }
        }
    };
    struct TaskResources {
        daxa::TaskImage output_image;
        daxa::TaskImage history_image;
    };
    Resources resources;
    TaskResources task_resources;

    auto get(daxa::Device a_device, daxa::ImageInfo const &a_info) -> std::pair<daxa::TaskImage &, daxa::TaskImage &> {
        if (!resources.device) {
            resources.device = a_device;
        }
        assert(resources.device == a_device);
        if (resources.image_a.is_empty()) {
            auto info_a = a_info;
            auto info_b = a_info;
            info_a.name += "_a";
            info_b.name += "_b";
            resources.image_a = a_device.create_image(info_a);
            resources.image_b = a_device.create_image(info_b);
            task_resources.output_image = daxa::TaskImage(daxa::TaskImageInfo{
                .initial_images = {.images = std::array{resources.image_a}},
                .name = a_info.name,
            });
            task_resources.history_image = daxa::TaskImage(daxa::TaskImageInfo{
                .initial_images = {.images = std::array{resources.image_b}},
                .name = a_info.name + "_history",
            });
        }
        return {task_resources.output_image, task_resources.history_image};
    }
};
