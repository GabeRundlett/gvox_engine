#pragma once

#include <shared/settings.inl>

#if defined(__cplusplus)
#include <cpu/core.hpp>
#include <cpu/app_ui.hpp>
#include <shared/renderer/core.inl>
#define CPU_ONLY(x) x
#define GPU_ONLY(x)
#else
#define CPU_ONLY(x)
#define GPU_ONLY(x) x
#endif

#define DECL_TASK_STATE(shader_file_path, Name, NAME, wg_size_x, wg_size_y, wg_size_z)                                                                                           \
    struct Name##ComputeTaskState {                                                                                                                                              \
        AsyncManagedComputePipeline pipeline;                                                                                                                                    \
        Name##ComputeTaskState(AsyncPipelineManager &pipeline_manager) {                                                                                                         \
            pipeline = pipeline_manager.add_compute_pipeline({                                                                                                                   \
                .shader_info = {                                                                                                                                                 \
                    .source = daxa::ShaderFile{shader_file_path},                                                                                                                \
                    .compile_options = {.defines = {{#NAME "_COMPUTE", "1"}}, .enable_debug_info = true},                                                                        \
                },                                                                                                                                                               \
                .name = #NAME "_COMPUTE",                                                                                                                                        \
            });                                                                                                                                                                  \
        }                                                                                                                                                                        \
        void record_commands(daxa::CommandRecorder &recorder, daxa_u32vec3 thread_count) {                                                                                       \
            if (!pipeline.is_valid())                                                                                                                                            \
                return;                                                                                                                                                          \
            recorder.set_pipeline(pipeline.get());                                                                                                                               \
            recorder.dispatch({(thread_count.x + (wg_size_x - 1)) / wg_size_x, (thread_count.y + (wg_size_y - 1)) / wg_size_y, (thread_count.z + (wg_size_z - 1)) / wg_size_z}); \
        }                                                                                                                                                                        \
    };                                                                                                                                                                           \
    struct Name##ComputeTask : Name##ComputeUses {                                                                                                                               \
        Name##ComputeTaskState *state;                                                                                                                                           \
        daxa_u32vec3 thread_count;                                                                                                                                               \
        void callback(daxa::TaskInterface const &ti) {                                                                                                                           \
            auto &recorder = ti.get_recorder();                                                                                                                                  \
            recorder.set_uniform_buffer(ti.uses.get_uniform_buffer_info());                                                                                                      \
            state->record_commands(recorder, thread_count);                                                                                                                      \
        }                                                                                                                                                                        \
    }

#define DECL_TASK_STATE_WITH_PUSH(shader_file_path, Name, NAME, wg_size_x, wg_size_y, wg_size_z, PushType)                                                                       \
    struct Name##ComputeTaskState {                                                                                                                                              \
        AsyncManagedComputePipeline pipeline;                                                                                                                                    \
        Name##ComputeTaskState(AsyncPipelineManager &pipeline_manager) {                                                                                                         \
            auto compile_result = pipeline_manager.add_compute_pipeline({                                                                                                        \
                .shader_info = {                                                                                                                                                 \
                    .source = daxa::ShaderFile{shader_file_path},                                                                                                                \
                    .compile_options = {.defines = {{#NAME "_COMPUTE", "1"}}, .enable_debug_info = true},                                                                        \
                },                                                                                                                                                               \
                .push_constant_size = sizeof(TaaPush),                                                                                                                           \
                .name = #NAME "_COMPUTE",                                                                                                                                        \
            });                                                                                                                                                                  \
            if (compile_result.is_err()) {                                                                                                                                       \
                AppUi::Console::s_instance->add_log(compile_result.message());                                                                                                   \
                return;                                                                                                                                                          \
            }                                                                                                                                                                    \
            pipeline = compile_result.value();                                                                                                                                   \
            if (!compile_result.value()->is_valid()) {                                                                                                                           \
                AppUi::Console::s_instance->add_log(compile_result.message());                                                                                                   \
            }                                                                                                                                                                    \
        }                                                                                                                                                                        \
        void record_commands(daxa::CommandRecorder &recorder, daxa_u32vec3 thread_count, PushType const &push) {                                                                 \
            if (!pipeline.is_valid())                                                                                                                                            \
                return;                                                                                                                                                          \
            recorder.set_pipeline(pipeline.get());                                                                                                                               \
            recorder.push_constant(push);                                                                                                                                        \
            recorder.dispatch({(thread_count.x + (wg_size_x - 1)) / wg_size_x, (thread_count.y + (wg_size_y - 1)) / wg_size_y, (thread_count.z + (wg_size_z - 1)) / wg_size_z}); \
        }                                                                                                                                                                        \
    };                                                                                                                                                                           \
    struct Name##ComputeTask : Name##ComputeUses {                                                                                                                               \
        Name##ComputeTaskState *state;                                                                                                                                           \
        daxa_u32vec3 thread_count;                                                                                                                                               \
        PushType push;                                                                                                                                                           \
        void callback(daxa::TaskInterface const &ti) {                                                                                                                           \
            auto &recorder = ti.get_recorder();                                                                                                                                  \
            recorder.set_uniform_buffer(ti.uses.get_uniform_buffer_info());                                                                                                      \
            state->record_commands(recorder, thread_count, push);                                                                                                                \
        }                                                                                                                                                                        \
    }
