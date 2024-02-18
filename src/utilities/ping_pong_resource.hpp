#pragma once

#include <daxa/daxa.hpp>
#include "gpu_context.hpp"

struct PingPongImage_impl {
    using ResourceType = daxa::ImageId;
    using ResourceInfoType = daxa::ImageInfo;
    using TaskResourceType = daxa::TaskImage;
    using TaskResourceInfoType = daxa::TaskImageInfo;
    using TemporalResourceType = TemporalImage;

    static auto create(GpuContext &gpu_context, ResourceInfoType const &info) -> TemporalResourceType {
        return gpu_context.find_or_add_temporal_image(info);
    }
    static void destroy(GpuContext &gpu_context, TemporalResourceType &rsrc_id) {
        gpu_context.remove_temporal_image(rsrc_id.resource_id);
    }
    static void swap(TemporalResourceType &resource_a, TemporalResourceType &resource_b) {
        resource_a.task_resource.swap_images(resource_b.task_resource);
    }
};

struct PingPongBuffer_impl {
    using ResourceType = daxa::BufferId;
    using ResourceInfoType = daxa::BufferInfo;
    using TaskResourceType = daxa::TaskBuffer;
    using TaskResourceInfoType = daxa::TaskBufferInfo;
    using TemporalResourceType = TemporalBuffer;

    static auto create(GpuContext &gpu_context, ResourceInfoType const &info) -> TemporalResourceType {
        return gpu_context.find_or_add_temporal_buffer(info);
    }
    static void destroy(GpuContext &gpu_context, TemporalResourceType rsrc_id) {
        gpu_context.remove_temporal_buffer(rsrc_id.resource_id);
    }
    static void swap(TemporalResourceType &resource_a, TemporalResourceType &resource_b) {
        resource_a.task_resource.swap_buffers(resource_b.task_resource);
    }
};

template <typename Impl>
struct PingPongResource {
    using ResourceType = typename Impl::ResourceType;
    using ResourceInfoType = typename Impl::ResourceInfoType;
    using TaskResourceType = typename Impl::TaskResourceType;
    using TaskResourceInfoType = typename Impl::TaskResourceInfoType;
    using TemporalResourceType = typename Impl::TemporalResourceType;

    struct Resources {
        Resources() = default;
        Resources(Resources const &) = delete;
        Resources(Resources &&) = delete;
        Resources &operator=(Resources const &) = delete;
        Resources &operator=(Resources &&other) {
            std::swap(this->gpu_context, other.gpu_context);
            std::swap(this->resource_a, other.resource_a);
            std::swap(this->resource_b, other.resource_b);
            return *this;
        }

        GpuContext *gpu_context{};
        TemporalResourceType resource_a;
        TemporalResourceType resource_b;

        ~Resources() {
            if (!resource_a.resource_id.is_empty()) {
                Impl::destroy(*gpu_context, resource_a);
                Impl::destroy(*gpu_context, resource_b);
            }
        }
    };
    Resources resources;

    void swap() {
        Impl::swap(resources.resource_a, resources.resource_b);
    }

    auto current() -> TaskResourceType & { return resources.resource_a.task_resource; }
    auto history() -> TaskResourceType & { return resources.resource_b.task_resource; }
    auto current() const -> TaskResourceType const & { return resources.resource_a.task_resource; }
    auto history() const -> TaskResourceType const & { return resources.resource_b.task_resource; }

    auto get(GpuContext &gpu_context, ResourceInfoType const &a_info) -> std::pair<TaskResourceType &, TaskResourceType &> {
        resources.gpu_context = &gpu_context;
        // assert(resources.device == a_device);
        if (resources.resource_a.resource_id.is_empty()) {
            auto info_a = a_info;
            auto info_b = a_info;
            info_a.name = std::string(info_a.name.view()) + "_a";
            info_b.name = std::string(info_b.name.view()) + "_b";
            resources.resource_a = Impl::create(*resources.gpu_context, info_a);
            resources.resource_b = Impl::create(*resources.gpu_context, info_b);
        }
        return {current(), history()};
    }
};

using PingPongImage = PingPongResource<PingPongImage_impl>;
using PingPongBuffer = PingPongResource<PingPongBuffer_impl>;
