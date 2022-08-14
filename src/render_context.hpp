#pragma once

#include <Daxa.hpp>

static constexpr int RENDER_SCL = 1;

struct RenderContext {
    daxa::DeviceHandle device;
    daxa::PipelineCompilerHandle pipeline_compiler;
    daxa::CommandQueueHandle queue;

    daxa::SwapchainHandle swapchain;
    daxa::SwapchainImage swapchain_image;
    daxa::ImageViewHandle render_col_images[2];
    daxa::ImageViewHandle render_pos_images[2];
    daxa::ImageViewHandle render_nrm_images[2];
    daxa::ImageViewHandle render_depth_image;
    u32 frame_i = 0;

    struct PerFrameData {
        daxa::SignalHandle present_signal;
        daxa::TimelineSemaphoreHandle timeline;
        u64 timeline_counter = 0;
    };
    std::deque<PerFrameData> frames;

    auto create_color_image(glm::ivec2 dim) {
        auto result = device->createImageView({
            .image = device->createImage({
                .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                .extent = {static_cast<u32>(dim.x), static_cast<u32>(dim.y), 1},
                .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                .debugName = "Render Image",
            }),
            .format = VK_FORMAT_R32G32B32A32_SFLOAT,
            .subresourceRange =
                {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            .debugName = "Render Image View",
        });
        return result;
    }

    auto create_depth_image(glm::ivec2 dim) {
        auto result = device->createImageView({
            .image = device->createImage({
                .format = VK_FORMAT_D32_SFLOAT,
                .extent = {static_cast<u32>(dim.x), static_cast<u32>(dim.y), 1},
                .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                .debugName = "Depth Image",
            }),
            .format = VK_FORMAT_D32_SFLOAT,
            .subresourceRange =
                {
                    .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            .debugName = "Depth Image View",
        });
        return result;
    }

    RenderContext(VkSurfaceKHR surface, glm::ivec2 dim)
        : device(daxa::Device::create()),
          pipeline_compiler{this->device->createPipelineCompiler()},
          queue(device->createCommandQueue({.batchCount = 2})),
          swapchain(device->createSwapchain({
              .surface = surface,
              .width = static_cast<u32>(dim.x),
              .height = static_cast<u32>(dim.y),
              .presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR,
              .additionalUses = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
              .debugName = "Swapchain",
          })),
          swapchain_image(swapchain->aquireNextImage()),
          render_col_images{
              create_color_image(dim / RENDER_SCL),
              create_color_image(dim / RENDER_SCL),
          },
          render_pos_images{
              create_color_image(dim / RENDER_SCL),
              create_color_image(dim / RENDER_SCL),
          },
          render_nrm_images{
              create_color_image(dim / RENDER_SCL),
              create_color_image(dim / RENDER_SCL),
          },
          render_depth_image(create_depth_image(dim / RENDER_SCL)) {
        for (int i = 0; i < 3; i++) {
            frames.push_back(PerFrameData{
                .present_signal = device->createSignal({}),
                .timeline = device->createTimelineSemaphore({}),
                .timeline_counter = 0,
            });
        }
        pipeline_compiler->addShaderSourceRootPath("shaders");
    }

    ~RenderContext() {
        wait_idle();
        frames.clear();
    }

    void wait_idle() {
        queue->waitIdle();
        queue->checkForFinishedSubmits();
        device->waitIdle();
    }

    auto begin_frame(glm::ivec2 dim) {
        bool resized = resize(dim);
        auto cmd_list = queue->getCommandList({});
        if (resized) {
            for (size_t i = 0; i < 2; ++i) {
                cmd_list.queueImageBarrier(daxa::ImageBarrier{
                    .barrier = daxa::FULL_MEMORY_BARRIER,
                    .image = render_col_images[i],
                    .layoutBefore = VK_IMAGE_LAYOUT_UNDEFINED,
                    .layoutAfter = VK_IMAGE_LAYOUT_GENERAL,
                });
                cmd_list.queueImageBarrier(daxa::ImageBarrier{
                    .barrier = daxa::FULL_MEMORY_BARRIER,
                    .image = render_pos_images[i],
                    .layoutBefore = VK_IMAGE_LAYOUT_UNDEFINED,
                    .layoutAfter = VK_IMAGE_LAYOUT_GENERAL,
                });
                cmd_list.queueImageBarrier(daxa::ImageBarrier{
                    .barrier = daxa::FULL_MEMORY_BARRIER,
                    .image = render_nrm_images[i],
                    .layoutBefore = VK_IMAGE_LAYOUT_UNDEFINED,
                    .layoutAfter = VK_IMAGE_LAYOUT_GENERAL,
                });
            }
            cmd_list.queueImageBarrier(daxa::ImageBarrier{
                .barrier = daxa::FULL_MEMORY_BARRIER,
                .image = render_depth_image,
                .layoutBefore = VK_IMAGE_LAYOUT_UNDEFINED,
                .layoutAfter = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            });
        } else {
            for (size_t i = 0; i < 2; ++i) {
                cmd_list.queueImageBarrier(daxa::ImageBarrier{
                    .barrier = daxa::FULL_MEMORY_BARRIER,
                    .image = render_col_images[i],
                });
                cmd_list.queueImageBarrier(daxa::ImageBarrier{
                    .barrier = daxa::FULL_MEMORY_BARRIER,
                    .image = render_pos_images[i],
                });
                cmd_list.queueImageBarrier(daxa::ImageBarrier{
                    .barrier = daxa::FULL_MEMORY_BARRIER,
                    .image = render_nrm_images[i],
                });
            }
            cmd_list.queueImageBarrier(daxa::ImageBarrier{
                .barrier = daxa::FULL_MEMORY_BARRIER,
                .image = render_depth_image,
            });
        }
        return cmd_list;
    }

    void begin_rendering(daxa::CommandListHandle cmd_list) {
        std::array framebuffer{daxa::RenderAttachmentInfo{
            .image = swapchain_image.getImageViewHandle(),
            .clearValue = {.color = {.float32 = {1.0f, 0.0f, 1.0f, 1.0f}}},
        }};
        daxa::RenderAttachmentInfo depth_attachment{
            .image = render_depth_image,
            .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .clearValue = {.depthStencil = {.depth = 1.0f}},
        };
        cmd_list.beginRendering(daxa::BeginRenderingInfo{
            .colorAttachments = framebuffer,
            .depthAttachment = &depth_attachment,
        });
    }

    void end_rendering(daxa::CommandListHandle cmd_list) {
        cmd_list.endRendering();
    }

    void submit(daxa::CommandListHandle cmd_list) {
        cmd_list.finalize();
        daxa::SubmitInfo submit_info;
        submit_info.commandLists.push_back(std::move(cmd_list));
        queue->submit(submit_info);
        queue->checkForFinishedSubmits();
        queue->nextBatch();
    }

    void end_frame(daxa::CommandListHandle cmd_list) {
        auto *current_frame = &frames.front();
        cmd_list.finalize();
        daxa::SubmitInfo submit_info;
        submit_info.commandLists.push_back(std::move(cmd_list));
        submit_info.signalOnCompletion = {&current_frame->present_signal, 1};
        queue->submit(submit_info);
        queue->present(std::move(swapchain_image), current_frame->present_signal);
        swapchain_image = swapchain->aquireNextImage();
        auto frame_context = std::move(frames.back());
        frames.pop_back();
        frames.push_front(std::move(frame_context));
        current_frame = &frames.front();
        queue->checkForFinishedSubmits();
        queue->nextBatch();
    }

    bool resize(glm::ivec2 dim) {
        if (dim.x != static_cast<i32>(swapchain->getSize().width) || dim.y != static_cast<i32>(swapchain->getSize().height)) {
            device->waitIdle();
            swapchain->resize(VkExtent2D{.width = static_cast<u32>(dim.x), .height = static_cast<u32>(dim.y)});
            swapchain_image = swapchain->aquireNextImage();
            render_col_images[0] = create_color_image(dim / RENDER_SCL);
            render_col_images[1] = create_color_image(dim / RENDER_SCL);
            render_pos_images[0] = create_color_image(dim / RENDER_SCL);
            render_pos_images[1] = create_color_image(dim / RENDER_SCL);
            render_nrm_images[0] = create_color_image(dim / RENDER_SCL);
            render_nrm_images[1] = create_color_image(dim / RENDER_SCL);
            render_depth_image = create_depth_image(dim / RENDER_SCL);
            return true;
        }
        return false;
    }

    void blit_to_swapchain(daxa::CommandListHandle cmd_list) {
        cmd_list.queueImageBarrier({
            .image = swapchain_image.getImageViewHandle(),
            .layoutBefore = VK_IMAGE_LAYOUT_UNDEFINED,
            .layoutAfter = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        });
        cmd_list.queueImageBarrier({
            .image = render_col_images[frame_i],
            .layoutBefore = VK_IMAGE_LAYOUT_GENERAL,
            .layoutAfter = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        });

        auto render_extent = swapchain_image.getImageViewHandle()->getImageHandle()->getVkExtent3D();
        auto swap_extent = swapchain_image.getImageViewHandle()->getImageHandle()->getVkExtent3D();
        VkImageBlit blit{
            .srcSubresource = VkImageSubresourceLayers{
                .aspectMask = VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .srcOffsets = {
                VkOffset3D{0, 0, 0},
                VkOffset3D{
                    static_cast<int32_t>(render_extent.width / RENDER_SCL),
                    static_cast<int32_t>(render_extent.height / RENDER_SCL),
                    static_cast<int32_t>(render_extent.depth),
                },
            },
            .dstSubresource = VkImageSubresourceLayers{
                .aspectMask = VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .dstOffsets = {
                VkOffset3D{0, 0, 0},
                VkOffset3D{
                    static_cast<int32_t>(swap_extent.width),
                    static_cast<int32_t>(swap_extent.height),
                    static_cast<int32_t>(swap_extent.depth),
                },
            },
        };
        cmd_list.insertQueuedBarriers();
        vkCmdBlitImage(
            cmd_list.getVkCommandBuffer(),
            render_col_images[frame_i]->getImageHandle()->getVkImage(),
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            swapchain_image.getImageViewHandle()->getImageHandle()->getVkImage(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);

        cmd_list.queueImageBarrier({
            .image = swapchain_image.getImageViewHandle(),
            .layoutBefore = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .layoutAfter = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        });
        cmd_list.queueImageBarrier({
            .image = render_col_images[frame_i],
            .layoutBefore = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .layoutAfter = VK_IMAGE_LAYOUT_GENERAL,
        });

        frame_i = 1 - frame_i;
    }
};
