#pragma once

#define DAXA_ENABLE_SHADER_NO_NAMESPACE 1
#define DAXA_ENABLE_IMAGE_OVERLOADS_BASIC 1
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#undef DAXA_ENABLE_SHADER_NO_NAMESPACE
#undef DAXA_ENABLE_IMAGE_OVERLOADS_BASIC

#define MAX_SIMULATED_VOXEL_PARTICLES 0 // (1 << 14)
#define MAX_RENDERED_VOXEL_PARTICLES 0  // (1 << 14)

#define PREPASS_SCL 2
#define SHADING_SCL 2

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

#define ENABLE_DIFFUSE_GI false
#define ENABLE_TAA true

#define DECL_TASK_STATE(shader_file_path, Name, NAME, wg_size_x, wg_size_y, wg_size_z)                                                                                         \
    struct Name##ComputeTaskState {                                                                                                                                            \
        std::shared_ptr<daxa::ComputePipeline> pipeline;                                                                                                                       \
        Name##ComputeTaskState(daxa::PipelineManager &pipeline_manager) {                                                                                                      \
            auto compile_result = pipeline_manager.add_compute_pipeline({                                                                                                      \
                .shader_info = {                                                                                                                                               \
                    .source = daxa::ShaderFile{shader_file_path},                                                                                                              \
                    .compile_options = {.defines = {{#NAME "_COMPUTE", "1"}}, .enable_debug_info = true},                                                                      \
                },                                                                                                                                                             \
                .name = #NAME "_COMPUTE",                                                                                                                                      \
            });                                                                                                                                                                \
            if (compile_result.is_err()) {                                                                                                                                     \
                AppUi::Console::s_instance->add_log(compile_result.message());                                                                                                 \
                return;                                                                                                                                                        \
            }                                                                                                                                                                  \
            pipeline = compile_result.value();                                                                                                                                 \
            if (!compile_result.value()->is_valid()) {                                                                                                                         \
                AppUi::Console::s_instance->add_log(compile_result.message());                                                                                                 \
            }                                                                                                                                                                  \
        }                                                                                                                                                                      \
        auto pipeline_is_valid() -> bool { return pipeline && pipeline->is_valid(); }                                                                                          \
        void record_commands(daxa::CommandList &cmd_list, u32vec3 thread_count) {                                                                                              \
            if (!pipeline_is_valid())                                                                                                                                          \
                return;                                                                                                                                                        \
            cmd_list.set_pipeline(*pipeline);                                                                                                                                  \
            cmd_list.dispatch((thread_count.x + (wg_size_x - 1)) / wg_size_x, (thread_count.y + (wg_size_y - 1)) / wg_size_y, (thread_count.z + (wg_size_z - 1)) / wg_size_z); \
        }                                                                                                                                                                      \
    };                                                                                                                                                                         \
    struct Name##ComputeTask : Name##ComputeUses {                                                                                                                             \
        Name##ComputeTaskState *state;                                                                                                                                         \
        u32vec3 thread_count;                                                                                                                                                  \
        void callback(daxa::TaskInterface const &ti) {                                                                                                                         \
            auto cmd_list = ti.get_command_list();                                                                                                                             \
            cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());                                                                                                    \
            state->record_commands(cmd_list, thread_count);                                                                                                                    \
        }                                                                                                                                                                      \
    }

#define DECL_TASK_STATE_WITH_PUSH(shader_file_path, Name, NAME, wg_size_x, wg_size_y, wg_size_z, PushType)                                                                     \
    struct Name##ComputeTaskState {                                                                                                                                            \
        std::shared_ptr<daxa::ComputePipeline> pipeline;                                                                                                                       \
        Name##ComputeTaskState(daxa::PipelineManager &pipeline_manager) {                                                                                                      \
            auto compile_result = pipeline_manager.add_compute_pipeline({                                                                                                      \
                .shader_info = {                                                                                                                                               \
                    .source = daxa::ShaderFile{shader_file_path},                                                                                                              \
                    .compile_options = {.defines = {{#NAME "_COMPUTE", "1"}}, .enable_debug_info = true},                                                                      \
                },                                                                                                                                                             \
                .push_constant_size = sizeof(TaaPush),                                                                                                                         \
                .name = #NAME "_COMPUTE",                                                                                                                                      \
            });                                                                                                                                                                \
            if (compile_result.is_err()) {                                                                                                                                     \
                AppUi::Console::s_instance->add_log(compile_result.message());                                                                                                 \
                return;                                                                                                                                                        \
            }                                                                                                                                                                  \
            pipeline = compile_result.value();                                                                                                                                 \
            if (!compile_result.value()->is_valid()) {                                                                                                                         \
                AppUi::Console::s_instance->add_log(compile_result.message());                                                                                                 \
            }                                                                                                                                                                  \
        }                                                                                                                                                                      \
        auto pipeline_is_valid() -> bool { return pipeline && pipeline->is_valid(); }                                                                                          \
        void record_commands(daxa::CommandList &cmd_list, u32vec3 thread_count, PushType const &push) {                                                                        \
            if (!pipeline_is_valid())                                                                                                                                          \
                return;                                                                                                                                                        \
            cmd_list.set_pipeline(*pipeline);                                                                                                                                  \
            cmd_list.push_constant(push);                                                                                                                                      \
            cmd_list.dispatch((thread_count.x + (wg_size_x - 1)) / wg_size_x, (thread_count.y + (wg_size_y - 1)) / wg_size_y, (thread_count.z + (wg_size_z - 1)) / wg_size_z); \
        }                                                                                                                                                                      \
    };                                                                                                                                                                         \
    struct Name##ComputeTask : Name##ComputeUses {                                                                                                                             \
        Name##ComputeTaskState *state;                                                                                                                                         \
        u32vec3 thread_count;                                                                                                                                                  \
        PushType push;                                                                                                                                                         \
        void callback(daxa::TaskInterface const &ti) {                                                                                                                         \
            auto cmd_list = ti.get_command_list();                                                                                                                             \
            cmd_list.set_uniform_buffer(ti.uses.get_uniform_buffer_info());                                                                                                    \
            state->record_commands(cmd_list, thread_count, push);                                                                                                              \
        }                                                                                                                                                                      \
    }
