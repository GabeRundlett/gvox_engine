#pragma once

#include <shared/core.inl>

#if SKY_TRANSMITTANCE_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(SkyTransmittanceComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(transmittance_lut, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif
#if SKY_MULTISCATTERING_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(SkyMultiscatteringComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(transmittance_lut, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(multiscattering_lut, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif
#if SKY_SKY_COMPUTE || defined(__cplusplus)
DAXA_DECL_TASK_USES_BEGIN(SkySkyComputeUses, DAXA_UNIFORM_BUFFER_SLOT0)
DAXA_TASK_USE_BUFFER(gpu_input, daxa_BufferPtr(GpuInput), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(transmittance_lut, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(multiscattering_lut, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_IMAGE(sky_lut, REGULAR_2D, COMPUTE_SHADER_STORAGE_WRITE_ONLY)
DAXA_DECL_TASK_USES_END()
#endif

#if defined(__cplusplus)

inline void sky_compile_compute_pipeline(AsyncPipelineManager &pipeline_manager, char const *const name, std::shared_ptr<daxa::ComputePipeline> &pipeline) {
    auto compile_result = pipeline_manager.add_compute_pipeline({
        .shader_info = {
            .source = daxa::ShaderFile{"sky.comp.glsl"},
            .compile_options = {.defines = {{name, "1"}}},
        },
        .name = std::string("sky_") + name,
    });
    if (compile_result.is_err()) {
        AppUi::Console::s_instance->add_log(compile_result.message());
        return;
    }
    pipeline = compile_result.value();
    if (!compile_result.value()->is_valid()) {
        AppUi::Console::s_instance->add_log(compile_result.message());
    }
}

#define SKY_DECL_TASK_STATE(Name, NAME, WG_SIZE_X, WG_SIZE_Y)                                                                                          \
    struct Name##ComputeTaskState {                                                                                                                    \
        std::shared_ptr<daxa::ComputePipeline> pipeline;                                                                                               \
        Name##ComputeTaskState(AsyncPipelineManager &pipeline_manager) { sky_compile_compute_pipeline(pipeline_manager, #NAME "_COMPUTE", pipeline); } \
        auto pipeline_is_valid() -> bool { return pipeline && pipeline->is_valid(); }                                                                  \
        void record_commands(daxa::CommandList &cmd_list) {                                                                                            \
            if (!pipeline_is_valid())                                                                                                                  \
                return;                                                                                                                                \
            cmd_list.set_pipeline(*pipeline);                                                                                                          \
            cmd_list.dispatch((NAME##_RES.x + (WG_SIZE_X - 1)) / WG_SIZE_X, (NAME##_RES.y + (WG_SIZE_Y - 1)) / WG_SIZE_Y);                                 \
        }                                                                                                                                              \
    };                                                                                                                                                 \
    struct Name##ComputeTask : Name##ComputeUses {                                                                                                     \
        Name##ComputeTaskState *state;                                                                                                                 \
        void callback(daxa::TaskInterface const &ti) {                                                                                                 \
            auto cmd_list = ti.get_command_list();                                                                                                     \
            cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());                                                                            \
            state->record_commands(cmd_list);                                                                                                          \
        }                                                                                                                                              \
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

    auto render(RecordContext &record_ctx) -> daxa::TaskImageView {
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
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .transmittance_lut = transmittance_lut,
                },
            },
            &sky_transmittance_task_state,
        });

        record_ctx.task_graph.add_task(SkyMultiscatteringComputeTask{
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .transmittance_lut = transmittance_lut,
                    .multiscattering_lut = multiscattering_lut,
                },
            },
            &sky_multiscattering_task_state,
        });

        record_ctx.task_graph.add_task(SkySkyComputeTask{
            {
                .uses = {
                    .gpu_input = record_ctx.task_input_buffer,
                    .transmittance_lut = transmittance_lut,
                    .multiscattering_lut = multiscattering_lut,
                    .sky_lut = sky_lut,
                },
            },
            &sky_sky_task_state,
        });

        return sky_lut;
    }
};

#endif
