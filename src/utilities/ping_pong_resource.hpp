#pragma once

#include <daxa/daxa.hpp>

struct PingPongImage_impl {
    using ResourceType = daxa::ImageId;
    using ResourceInfoType = daxa::ImageInfo;
    using TaskResourceType = daxa::TaskImage;
    using TaskResourceInfoType = daxa::TaskImageInfo;

    static auto create(daxa::Device &device, ResourceInfoType const &info) -> ResourceType {
        return device.create_image(info);
    }
    static void destroy(daxa::Device &device, ResourceType rsrc_id) {
        device.destroy_image(rsrc_id);
    }
    static auto create_task_resource(ResourceType rsrc_id, std::string const &name) -> TaskResourceType {
        return TaskResourceType(TaskResourceInfoType{.initial_images = {std::array{rsrc_id}}, .name = name});
    }
    static void swap(TaskResourceType &resource_a, TaskResourceType &resource_b) {
        resource_a.swap_images(resource_b);
    }
};

struct PingPongBuffer_impl {
    using ResourceType = daxa::BufferId;
    using ResourceInfoType = daxa::BufferInfo;
    using TaskResourceType = daxa::TaskBuffer;
    using TaskResourceInfoType = daxa::TaskBufferInfo;

    static auto create(daxa::Device &device, ResourceInfoType const &info) -> ResourceType {
        return device.create_buffer(info);
    }
    static void destroy(daxa::Device &device, ResourceType rsrc_id) {
        device.destroy_buffer(rsrc_id);
    }
    static auto create_task_resource(ResourceType rsrc_id, std::string const &name) -> TaskResourceType {
        return TaskResourceType(TaskResourceInfoType{.initial_buffers = {std::array{rsrc_id}}, .name = name});
    }
    static void swap(TaskResourceType &resource_a, TaskResourceType &resource_b) {
        resource_a.swap_buffers(resource_b);
    }
};

template <typename Impl>
struct PingPongResource {
    using ResourceType = typename Impl::ResourceType;
    using ResourceInfoType = typename Impl::ResourceInfoType;
    using TaskResourceType = typename Impl::TaskResourceType;
    using TaskResourceInfoType = typename Impl::TaskResourceInfoType;

    struct Resources {
        Resources() = default;
        Resources(Resources const &) = delete;
        Resources(Resources &&) = delete;
        Resources &operator=(Resources const &) = delete;
        Resources &operator=(Resources &&other) {
            std::swap(this->device, other.device);
            std::swap(this->resource_a, other.resource_a);
            std::swap(this->resource_b, other.resource_b);
            return *this;
        }

        daxa::Device device{};
        ResourceType resource_a{};
        ResourceType resource_b{};

        ~Resources() {
            if (!resource_a.is_empty()) {
                Impl::destroy(device, resource_a);
                Impl::destroy(device, resource_b);
            }
        }
    };
    struct TaskResources {
        TaskResourceType output_resource;
        TaskResourceType history_resource;
    };
    Resources resources;
    TaskResources task_resources;

    void swap() {
        Impl::swap(task_resources.output_resource, task_resources.history_resource);
    }

    auto current() -> TaskResourceType & { return task_resources.output_resource; }
    auto history() -> TaskResourceType & { return task_resources.history_resource; }
    auto current() const -> TaskResourceType const & { return task_resources.output_resource; }
    auto history() const -> TaskResourceType const & { return task_resources.history_resource; }

    auto get(daxa::Device a_device, ResourceInfoType const &a_info) -> std::pair<TaskResourceType &, TaskResourceType &> {
        if (!resources.device.is_valid()) {
            resources.device = a_device;
        }
        // assert(resources.device == a_device);
        if (resources.resource_a.is_empty()) {
            auto info_a = a_info;
            auto info_b = a_info;
            info_a.name = std::string(info_a.name.view()) + "_a";
            info_b.name = std::string(info_b.name.view()) + "_b";
            resources.resource_a = Impl::create(a_device, info_a);
            resources.resource_b = Impl::create(a_device, info_b);
            task_resources.output_resource = Impl::create_task_resource(resources.resource_a, std::string(a_info.name.view()));
            task_resources.history_resource = Impl::create_task_resource(resources.resource_b, std::string(a_info.name.view()) + "_hist");
        }
        return {current(), history()};
    }
};

using PingPongImage = PingPongResource<PingPongImage_impl>;
using PingPongBuffer = PingPongResource<PingPongBuffer_impl>;
