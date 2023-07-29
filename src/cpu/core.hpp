#pragma once

#include <memory>
#include <daxa/daxa.hpp>
#include <daxa/utils/pipeline_manager.hpp>
#include <daxa/utils/task_graph.hpp>

static constexpr usize FRAMES_IN_FLIGHT = 1;

struct GbufferDepth {
    daxa::ImageId geometric_normal;
    daxa::ImageId gbuffer;
    daxa::ImageId depth;
    std::optional<daxa::ImageId> half_view_normal;
    std::optional<daxa::ImageId> half_depth;
};

struct PingPongImage {
    struct Resources {
        daxa::ImageId output_image_id;
        daxa::ImageId history_image_id;
    };
    struct TaskResources {
        daxa::TaskImage task_output_image_id;
        daxa::TaskImage task_history_image_id;
    };
    Resources resources;
    TaskResources task_resources;
    daxa::Device device;

    TaskResources get(daxa::Device a_device, daxa::ImageInfo const &a_info) {
        if (!device) {
            device = a_device;
        }
        assert(device == a_device);
        if (resources.output_image_id.is_empty()) {
            resources.output_image_id = device.create_image(a_info);
            resources.history_image_id = device.create_image(a_info);
            task_resources.task_output_image_id = daxa::TaskImage(daxa::TaskImageInfo{
                .initial_images = {.images = std::array{resources.history_image_id}},
                .name = a_info.name,
            });
            task_resources.task_history_image_id = daxa::TaskImage(daxa::TaskImageInfo{
                .initial_images = {.images = std::array{resources.output_image_id}},
                .name = a_info.name,
            });
        }
        task_resources.task_output_image_id.swap_images(task_resources.task_history_image_id);
        return task_resources;
    }

    ~PingPongImage() {
        if (!resources.output_image_id.is_empty()) {
            device.destroy_image(resources.output_image_id);
            device.destroy_image(resources.history_image_id);
        }
    }
};
