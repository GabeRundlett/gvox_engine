#include <shared/renderer/fsr.inl>
#include <ffx-fsr2-api/vk/ffx_fsr2_vk.h>

#include <daxa/c/daxa.h>

Fsr2Renderer::Fsr2Renderer(daxa::Device a_device, Fsr2Info a_info) : device{std::move(a_device)}, info{a_info} {
    context_description.maxRenderSize.width = info.render_resolution.x;
    context_description.maxRenderSize.height = info.render_resolution.y;
    context_description.displaySize.width = info.display_resolution.x;
    context_description.displaySize.height = info.display_resolution.y;
    auto vk_device = daxa_dvc_get_vk_device(*reinterpret_cast<daxa_Device *>(&device));
    auto vk_physical_device = daxa_dvc_get_vk_physical_device(*reinterpret_cast<daxa_Device *>(&device));

    auto const scratch_buffer_size = ffxFsr2GetScratchMemorySizeVK(vk_physical_device);
    scratch_buffer.resize(scratch_buffer_size);

    {
        FfxErrorCode const err = ffxFsr2GetInterfaceVK(&context_description.callbacks, scratch_buffer.data(), scratch_buffer_size, vk_physical_device, vkGetDeviceProcAddr);
        if (err != FFX_OK) {
            throw std::runtime_error("FSR2 Failed to create Vulkan interface");
        }
    }

    context_description.device = ffxGetDeviceVK(vk_device);
    context_description.flags = FFX_FSR2_ENABLE_DEPTH_INFINITE | FFX_FSR2_ENABLE_DEPTH_INVERTED | FFX_FSR2_ENABLE_HIGH_DYNAMIC_RANGE;

    {
        FfxErrorCode const err = ffxFsr2ContextCreate(&fsr_context, &context_description);
        if (err != FFX_OK) {
            throw std::runtime_error("FSR Failed to create context");
        }
    }
}

Fsr2Renderer::~Fsr2Renderer() {
    if (!scratch_buffer.empty()) {
        FfxErrorCode const err = ffxFsr2ContextDestroy(&fsr_context);
        if (err != FFX_OK) {
            // throw std::runtime_error("FSR2 Failed to destroy context");
        }
    }
}

void Fsr2Renderer::next_frame() {
    daxa_i32 const jitter_phase_count = ffxFsr2GetJitterPhaseCount(
        static_cast<daxa_i32>(info.render_resolution.x),
        static_cast<daxa_i32>(info.display_resolution.x));
    ffxFsr2GetJitterOffset(&state.jitter.x, &state.jitter.y, static_cast<daxa_i32>(jitter_frame_i), jitter_phase_count);
    ++jitter_frame_i;
}

auto Fsr2Renderer::upscale(RecordContext &record_ctx, GbufferDepth const &gbuffer_depth, daxa::TaskImageView color_image, daxa::TaskImageView velocity_image) -> daxa::TaskImageView {
    auto output_image = record_ctx.task_graph.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {record_ctx.output_resolution.x, record_ctx.output_resolution.y, 1},
        .name = "fsr2_output_image",
    });
    auto depth_image = gbuffer_depth.depth.task_resources.output_resource.view();
    record_ctx.task_graph.add_task({
        .uses = {
            daxa::TaskImageUse<daxa::TaskImageAccess::COMPUTE_SHADER_SAMPLED>{color_image},
            daxa::TaskImageUse<daxa::TaskImageAccess::COMPUTE_SHADER_SAMPLED>{depth_image},
            daxa::TaskImageUse<daxa::TaskImageAccess::COMPUTE_SHADER_SAMPLED>{velocity_image},
            daxa::TaskImageUse<daxa::TaskImageAccess::COMPUTE_SHADER_STORAGE_WRITE_ONLY>{output_image},
        },
        .task = [=](daxa::TaskInterface ti) {
            auto const &color_use = ti.uses[color_image];
            auto const &depth_use = ti.uses[depth_image];
            auto const &velocity_use = ti.uses[velocity_image];
            auto const &output_use = ti.uses[output_image];

            VkImage color_vk_image = {}, depth_vk_image = {}, velocity_vk_image = {}, output_vk_image = {};
            daxa_dvc_get_vk_image(*reinterpret_cast<daxa_Device *>(&device), std::bit_cast<daxa_ImageId>(color_use.image()), &color_vk_image);
            daxa_dvc_get_vk_image(*reinterpret_cast<daxa_Device *>(&device), std::bit_cast<daxa_ImageId>(depth_use.image()), &depth_vk_image);
            daxa_dvc_get_vk_image(*reinterpret_cast<daxa_Device *>(&device), std::bit_cast<daxa_ImageId>(velocity_use.image()), &velocity_vk_image);
            daxa_dvc_get_vk_image(*reinterpret_cast<daxa_Device *>(&device), std::bit_cast<daxa_ImageId>(output_use.image()), &output_vk_image);

            VkImageView color_vk_image_view = {}, depth_vk_image_view = {}, velocity_vk_image_view = {}, output_vk_image_view = {};
            daxa_dvc_get_vk_image_view(*reinterpret_cast<daxa_Device *>(&device), std::bit_cast<daxa_ImageViewId>(color_use.view()), &color_vk_image_view);
            daxa_dvc_get_vk_image_view(*reinterpret_cast<daxa_Device *>(&device), std::bit_cast<daxa_ImageViewId>(depth_use.view()), &depth_vk_image_view);
            daxa_dvc_get_vk_image_view(*reinterpret_cast<daxa_Device *>(&device), std::bit_cast<daxa_ImageViewId>(velocity_use.view()), &velocity_vk_image_view);
            daxa_dvc_get_vk_image_view(*reinterpret_cast<daxa_Device *>(&device), std::bit_cast<daxa_ImageViewId>(output_use.view()), &output_vk_image_view);

            auto const &color_extent = device.info_image(color_use.image()).value().size;
            auto const &depth_extent = device.info_image(depth_use.image()).value().size;
            auto const &velocity_extent = device.info_image(velocity_use.image()).value().size;
            auto const &output_extent = device.info_image(output_use.image()).value().size;

            auto const color_format = (VkFormat)device.info_image(color_use.image()).value().format;
            auto const depth_format = (VkFormat)device.info_image(depth_use.image()).value().format;
            auto const velocity_format = (VkFormat)device.info_image(velocity_use.image()).value().format;
            auto const output_format = (VkFormat)device.info_image(output_use.image()).value().format;

            auto &recorder = ti.get_recorder();

            wchar_t fsr_input_color[] = L"FSR2_InputColor";
            wchar_t fsr_input_depth[] = L"FSR2_InputDepth";
            wchar_t fsr_input_velocity[] = L"FSR2_InputMotionVectors";
            wchar_t fsr_input_exposure[] = L"FSR2_InputExposure";
            wchar_t fsr_output_upscaled_color[] = L"FSR2_OutputUpscaledColor";

            FfxFsr2DispatchDescription dispatch_description = {};

            auto cmd_buffer = daxa_cmd_get_vk_command_buffer(*reinterpret_cast<daxa_CommandRecorder *>(&recorder));
            dispatch_description.commandList = ffxGetCommandListVK(cmd_buffer);

            dispatch_description.color = ffxGetTextureResourceVK(
                &fsr_context, color_vk_image, color_vk_image_view,
                color_extent.x, color_extent.y,
                color_format, fsr_input_color);

            dispatch_description.depth = ffxGetTextureResourceVK(
                &fsr_context, depth_vk_image, depth_vk_image_view,
                depth_extent.x, depth_extent.y,
                depth_format, fsr_input_depth);

            dispatch_description.motionVectors = ffxGetTextureResourceVK(
                &fsr_context, velocity_vk_image, velocity_vk_image_view,
                velocity_extent.x, velocity_extent.y,
                velocity_format, fsr_input_velocity);

            dispatch_description.exposure = ffxGetTextureResourceVK(&fsr_context, nullptr, nullptr, 1, 1, VK_FORMAT_UNDEFINED, fsr_input_exposure);

            dispatch_description.output = ffxGetTextureResourceVK(
                &fsr_context, output_vk_image, output_vk_image_view,
                output_extent.x, output_extent.y,
                output_format, fsr_output_upscaled_color, FFX_RESOURCE_STATE_UNORDERED_ACCESS);

            dispatch_description.jitterOffset.x = state.jitter.x;
            dispatch_description.jitterOffset.y = state.jitter.y;
            dispatch_description.motionVectorScale.x = static_cast<daxa_f32>(velocity_extent.x);
            dispatch_description.motionVectorScale.y = static_cast<daxa_f32>(velocity_extent.y);
            dispatch_description.reset = state.should_reset;
            dispatch_description.enableSharpening = state.should_sharpen;
            dispatch_description.sharpness = state.sharpening;
            dispatch_description.frameTimeDelta = state.delta_time * 1000.0f;
            dispatch_description.preExposure = 1.0f;
            dispatch_description.renderSize.width = color_extent.x;
            dispatch_description.renderSize.height = color_extent.y;
            dispatch_description.cameraFar = state.camera_info.far_plane;
            dispatch_description.cameraNear = state.camera_info.near_plane;
            dispatch_description.cameraFovAngleVertical = state.camera_info.vertical_fov;

            FfxErrorCode const err = ffxFsr2ContextDispatch(&fsr_context, &dispatch_description);
            if (err != FFX_OK) {
                throw std::runtime_error("[ERROR][Fsr::Fsr()] FSR Failed to create context");
            }
        },
        .name = "FSR2",
    });
    return output_image;
}
