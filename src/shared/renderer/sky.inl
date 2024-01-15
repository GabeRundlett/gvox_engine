#pragma once

#include <shared/core.inl>

#if SKY_TRANSMITTANCE_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(SkyTransmittanceCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, transmittance_lut)
DAXA_DECL_TASK_HEAD_END
struct SkyTransmittanceComputePush {
    SkyTransmittanceCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(SkyTransmittanceComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewId transmittance_lut = push.uses.transmittance_lut;
#endif
#endif
#if SKY_MULTISCATTERING_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(SkyMultiscatteringCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, multiscattering_lut)
DAXA_DECL_TASK_HEAD_END
struct SkyMultiscatteringComputePush {
    SkyMultiscatteringCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(SkyMultiscatteringComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewId transmittance_lut = push.uses.transmittance_lut;
daxa_ImageViewId multiscattering_lut = push.uses.multiscattering_lut;
#endif
#endif
#if SKY_SKY_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_HEAD_BEGIN(SkySkyCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, multiscattering_lut)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, sky_lut)
DAXA_DECL_TASK_HEAD_END
struct SkySkyComputePush {
    SkySkyCompute uses;
};
#if DAXA_SHADER
DAXA_DECL_PUSH_CONSTANT(SkySkyComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewId transmittance_lut = push.uses.transmittance_lut;
daxa_ImageViewId multiscattering_lut = push.uses.multiscattering_lut;
daxa_ImageViewId sky_lut = push.uses.sky_lut;
#endif
#endif

#if defined(__cplusplus)

#define SKY_DECL_TASK_STATE(Name, NAME, WG_SIZE_X, WG_SIZE_Y)                                                                \
    struct Name##ComputeTaskState {                                                                                          \
        AsyncManagedComputePipeline pipeline;                                                                                \
        Name##ComputeTaskState(AsyncPipelineManager &pipeline_manager) {                                                     \
            pipeline = pipeline_manager.add_compute_pipeline({                                                               \
                .shader_info = {                                                                                             \
                    .source = daxa::ShaderFile{"sky.comp.glsl"},                                                             \
                    .compile_options = {.defines = {{#NAME "_COMPUTE", "1"}}},                                               \
                },                                                                                                           \
                .push_constant_size = sizeof(Name##ComputePush),                                                             \
                .name = "sky_" #NAME,                                                                                        \
            });                                                                                                              \
        }                                                                                                                    \
        void record_commands(Name##ComputePush const &push, daxa::CommandRecorder &recorder) {                               \
            if (!pipeline.is_valid())                                                                                        \
                return;                                                                                                      \
            recorder.set_pipeline(pipeline.get());                                                                           \
            recorder.push_constant(push);                                                                                    \
            recorder.dispatch({(NAME##_RES.x + (WG_SIZE_X - 1)) / WG_SIZE_X, (NAME##_RES.y + (WG_SIZE_Y - 1)) / WG_SIZE_Y}); \
        }                                                                                                                    \
    };                                                                                                                       \
    struct Name##ComputeTask {                                                                                               \
        Name##Compute::Uses uses;                                                                                            \
        std::string name = #Name "Compute";                                                                                  \
        Name##ComputeTaskState *state;                                                                                       \
        void callback(daxa::TaskInterface const &ti) {                                                                       \
            auto &recorder = ti.get_recorder();                                                                              \
            auto push = Name##ComputePush{};                                                                                 \
            ti.copy_task_head_to(&push.uses);                                                                                \
            state->record_commands(push, recorder);                                                                          \
        }                                                                                                                    \
    }

SKY_DECL_TASK_STATE(SkyTransmittance, SKY_TRANSMITTANCE, 8, 4);
SKY_DECL_TASK_STATE(SkyMultiscattering, SKY_MULTISCATTERING, 1, 1);
SKY_DECL_TASK_STATE(SkySky, SKY_SKY, 8, 4);

struct SkyRenderer {
    SkyTransmittanceComputeTaskState sky_transmittance_task_state;
    SkyMultiscatteringComputeTaskState sky_multiscattering_task_state;
    SkySkyComputeTaskState sky_sky_task_state;

    SkyRenderer(AsyncPipelineManager &pipeline_manager)
        : sky_transmittance_task_state{pipeline_manager},
          sky_multiscattering_task_state{pipeline_manager},
          sky_sky_task_state{pipeline_manager} {
    }

    auto render(RecordContext &record_ctx) -> std::pair<daxa::TaskImageView, daxa::TaskImageView> {
        auto transmittance_lut = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {SKY_TRANSMITTANCE_RES.x, SKY_TRANSMITTANCE_RES.y, 1},
            .name = "transmittance_lut",
        });
        auto multiscattering_lut = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {SKY_MULTISCATTERING_RES.x, SKY_MULTISCATTERING_RES.y, 1},
            .name = "multiscattering_lut",
        });
        auto sky_lut = record_ctx.task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {SKY_SKY_RES.x, SKY_SKY_RES.y, 1},
            .name = "sky_lut",
        });

        record_ctx.task_graph.add_task(SkyTransmittanceComputeTask{
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .transmittance_lut = transmittance_lut,
            },
            .state = &sky_transmittance_task_state,
        });

        record_ctx.task_graph.add_task(SkyMultiscatteringComputeTask{
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .transmittance_lut = transmittance_lut,
                .multiscattering_lut = multiscattering_lut,
            },
            .state = &sky_multiscattering_task_state,
        });

        record_ctx.task_graph.add_task(SkySkyComputeTask{
            .uses = {
                .gpu_input = record_ctx.task_input_buffer,
                .transmittance_lut = transmittance_lut,
                .multiscattering_lut = multiscattering_lut,
                .sky_lut = sky_lut,
            },
            .state = &sky_sky_task_state,
        });

        return {sky_lut, transmittance_lut};
    }
};

#endif
