#pragma once

#include <memory>
#include <array>

#include "debug.hpp"
#include "thread_pool.hpp"

#include <daxa/daxa.hpp>
#include <daxa/utils/pipeline_manager.hpp>

#include <future>

template <typename PipelineType>
struct AsyncManagedPipeline {
    using PipelineT = PipelineType;
    std::shared_ptr<PipelineT> pipeline;
    std::shared_ptr<std::promise<std::shared_ptr<PipelineT>>> pipeline_promise;
    std::future<std::shared_ptr<PipelineT>> pipeline_future;

    auto is_valid() -> bool {
        if (pipeline_future.valid()) {
            pipeline_future.wait();
            pipeline = pipeline_future.get();
        }
        return pipeline && pipeline->is_valid();
    }
    auto get() -> PipelineT & {
        return *pipeline;
    }
};

using AsyncManagedComputePipeline = AsyncManagedPipeline<daxa::ComputePipeline>;
using AsyncManagedRayTracingPipeline = AsyncManagedPipeline<daxa::RayTracingPipeline>;
using AsyncManagedRasterPipeline = AsyncManagedPipeline<daxa::RasterPipeline>;

struct AsyncPipelineManager {
    std::array<daxa::PipelineManager, 8> pipeline_managers;
    std::array<std::mutex, 8> mutexes{};
    std::atomic_uint64_t current_index = 0;

    std::vector<thread_pool::Task> tasks;

    AsyncPipelineManager(daxa::PipelineManagerInfo2 info) {
        pipeline_managers = {
            daxa::PipelineManager(info),
            daxa::PipelineManager(info),
            daxa::PipelineManager(info),
            daxa::PipelineManager(info),
            daxa::PipelineManager(info),
            daxa::PipelineManager(info),
            daxa::PipelineManager(info),
            daxa::PipelineManager(info),
        };
    }

    AsyncPipelineManager(AsyncPipelineManager const &) = delete;
    AsyncPipelineManager(AsyncPipelineManager &&) noexcept = delete;
    AsyncPipelineManager &operator=(AsyncPipelineManager const &) = delete;
    AsyncPipelineManager &operator=(AsyncPipelineManager &&) noexcept = delete;

    template <typename PipelineT, typename PipelineInfoT>
    struct TaskState {
        AsyncPipelineManager *self;
        std::shared_ptr<std::promise<std::shared_ptr<PipelineT>>> pipeline_promise;
        PipelineInfoT info_copy;
    };

    using ComputeTaskState = TaskState<daxa::ComputePipeline, daxa::ComputePipelineCompileInfo2>;
    using RasterTaskState = TaskState<daxa::RasterPipeline, daxa::RasterPipelineCompileInfo2>;
    using RaytraceTaskState = TaskState<daxa::RayTracingPipeline, daxa::RayTracingPipelineCompileInfo2>;

    auto add_compute_pipeline(daxa::ComputePipelineCompileInfo2 const &info) -> AsyncManagedComputePipeline {
        auto pipeline_promise = std::make_shared<std::promise<std::shared_ptr<daxa::ComputePipeline>>>();
        auto result = AsyncManagedComputePipeline{};
        result.pipeline_promise = pipeline_promise;
        result.pipeline_future = pipeline_promise->get_future();

        auto *task_state = new ComputeTaskState{
            .self = this,
            .pipeline_promise = pipeline_promise,
            .info_copy = info,
        };
        auto task = thread_pool::create_task(
            [](void *state_ptr) {
                auto &state = *(ComputeTaskState *)state_ptr;
                auto [pipeline_manager, lock] = state.self->get_pipeline_manager();
                auto compile_result = pipeline_manager.add_compute_pipeline2(state.info_copy);
                if (compile_result.is_err()) {
                    // debug_utils::add_log(g_console, compile_result.message().c_str());
                    return;
                }
                if (!compile_result.value()->is_valid()) {
                    // debug_utils::add_log(g_console, compile_result.message().c_str());
                    return;
                }
                state.pipeline_promise->set_value(compile_result.value());
                delete &state;
            },
            task_state);
        thread_pool::async_dispatch(task);
        tasks.push_back(task);

        return result;
    }
    auto add_ray_tracing_pipeline(daxa::RayTracingPipelineCompileInfo2 const &info) -> AsyncManagedRayTracingPipeline {
        auto pipeline_promise = std::make_shared<std::promise<std::shared_ptr<daxa::RayTracingPipeline>>>();
        auto result = AsyncManagedRayTracingPipeline{};
        result.pipeline_promise = pipeline_promise;
        result.pipeline_future = pipeline_promise->get_future();

        auto *task_state = new RaytraceTaskState{
            .self = this,
            .pipeline_promise = pipeline_promise,
            .info_copy = info,
        };
        auto task = thread_pool::create_task(
            [](void *state_ptr) {
                auto &state = *(RaytraceTaskState *)state_ptr;
                auto [pipeline_manager, lock] = state.self->get_pipeline_manager();
                auto compile_result = pipeline_manager.add_ray_tracing_pipeline2(state.info_copy);
                if (compile_result.is_err()) {
                    // debug_utils::add_log(g_console, compile_result.message().c_str());
                    return;
                }
                if (!compile_result.value()->is_valid()) {
                    // debug_utils::add_log(g_console, compile_result.message().c_str());
                    return;
                }
                state.pipeline_promise->set_value(compile_result.value());
                delete &state;
            },
            task_state);
        thread_pool::async_dispatch(task);
        tasks.push_back(task);

        return result;
    }
    auto add_raster_pipeline(daxa::RasterPipelineCompileInfo2 const &info) -> AsyncManagedRasterPipeline {
        auto pipeline_promise = std::make_shared<std::promise<std::shared_ptr<daxa::RasterPipeline>>>();
        auto result = AsyncManagedRasterPipeline{};
        result.pipeline_promise = pipeline_promise;
        result.pipeline_future = pipeline_promise->get_future();
        auto info_copy = info;

        auto *task_state = new RasterTaskState{
            .self = this,
            .pipeline_promise = pipeline_promise,
            .info_copy = info,
        };
        auto task = thread_pool::create_task(
            [](void *state_ptr) {
                auto &state = *(RasterTaskState *)state_ptr;
                auto [pipeline_manager, lock] = state.self->get_pipeline_manager();
                auto compile_result = pipeline_manager.add_raster_pipeline2(state.info_copy);
                if (compile_result.is_err()) {
                    // debug_utils::add_log(g_console, compile_result.message().c_str());
                    return;
                }
                if (!compile_result.value()->is_valid()) {
                    // debug_utils::add_log(g_console, compile_result.message().c_str());
                    return;
                }
                state.pipeline_promise->set_value(compile_result.value());
                delete &state;
            },
            task_state);
        thread_pool::async_dispatch(task);
        tasks.push_back(task);

        return result;
    }
    void remove_compute_pipeline(std::shared_ptr<daxa::ComputePipeline> const &pipeline) {
        auto [pipeline_manager, lock] = get_pipeline_manager();
        pipeline_manager.remove_compute_pipeline(pipeline);
    }
    void remove_raster_pipeline(std::shared_ptr<daxa::RasterPipeline> const &pipeline) {
        auto [pipeline_manager, lock] = get_pipeline_manager();
        pipeline_manager.remove_raster_pipeline(pipeline);
    }
    void add_virtual_file(daxa::VirtualFileInfo const &info) {
        for (auto &pipeline_manager : pipeline_managers) {
            pipeline_manager.add_virtual_file(info);
        }
    }
    void wait() {
        for (auto const &task : tasks) {
            thread_pool::wait(task);
        }
        tasks.clear();
    }
    auto reload_all() -> daxa::PipelineReloadResult {
        std::array<daxa::PipelineReloadResult, 8> results;
        for (daxa_u32 i = 0; i < pipeline_managers.size(); ++i) {
            auto &pipeline_manager = pipeline_managers[i];
            auto lock = std::lock_guard{mutexes[i]};
            results[i] = pipeline_manager.reload_all();
        }
        for (auto const &result : results) {
            if (daxa::holds_alternative<daxa::PipelineReloadError>(result)) {
                return result;
            }
        }
        return results[0];
    }

  private:
    auto get_pipeline_manager() -> std::pair<daxa::PipelineManager &, std::unique_lock<std::mutex>> {
        auto index = current_index.fetch_add(1);
        index = (index / 4) % pipeline_managers.size();

#if NDEBUG // Instead of this, Pipeline manager really needs to be internally thread-safe
        // try to find one that's not locked, otherwise we'll fall back on the index above.
        for (daxa_u32 i = 0; i < pipeline_managers.size(); ++i) {
            auto &mtx = this->mutexes[i];
            if (mtx.try_lock()) {
                index = i;
                mtx.unlock();
                break;
            }
        }
#endif
        return {
            pipeline_managers[index],
            std::unique_lock(mutexes[index]),
        };
    }
};
