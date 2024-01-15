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
                .push_constant_size = sizeof(Name##ComputePush),                                                                                                                 \
                .name = #NAME "_COMPUTE",                                                                                                                                        \
            });                                                                                                                                                                  \
        }                                                                                                                                                                        \
        void record_commands(Name##ComputePush const &push, daxa::CommandRecorder &recorder, daxa_u32vec3 thread_count) {                                                        \
            if (!pipeline.is_valid())                                                                                                                                            \
                return;                                                                                                                                                          \
            recorder.set_pipeline(pipeline.get());                                                                                                                               \
            recorder.push_constant(push);                                                                                                                                        \
            recorder.dispatch({(thread_count.x + (wg_size_x - 1)) / wg_size_x, (thread_count.y + (wg_size_y - 1)) / wg_size_y, (thread_count.z + (wg_size_z - 1)) / wg_size_z}); \
        }                                                                                                                                                                        \
    };                                                                                                                                                                           \
    struct Name##ComputeTask {                                                                                                                                                   \
        Name##Compute::Uses uses;                                                                                                                                                \
        std::string name = #Name;                                                                                                                                                \
        Name##ComputeTaskState *state;                                                                                                                                           \
        daxa_u32vec3 thread_count;                                                                                                                                               \
        void callback(daxa::TaskInterface const &ti) {                                                                                                                           \
            auto &recorder = ti.get_recorder();                                                                                                                                  \
            auto push = Name##ComputePush{};                                                                                                                                     \
            ti.copy_task_head_to(&push.uses);                                                                                                                                    \
            state->record_commands(push, recorder, thread_count);                                                                                                                \
        }                                                                                                                                                                        \
    }
