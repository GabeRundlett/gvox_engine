#pragma once

#include <memory>
#include <array>

#include "debug.hpp"
#include "thread_pool.hpp"

#include <daxa/daxa.hpp>
#include <daxa/utils/pipeline_manager.hpp>

struct AsyncManagedComputePipeline {
    using PipelineT = daxa::ComputePipeline;
    std::shared_ptr<daxa::ComputePipeline> pipeline;
#if ENABLE_THREAD_POOL
    std::shared_ptr<std::promise<std::shared_ptr<daxa::ComputePipeline>>> pipeline_promise;
    std::future<std::shared_ptr<daxa::ComputePipeline>> pipeline_future;
#endif

    auto is_valid() -> bool {
#if ENABLE_THREAD_POOL
        if (pipeline_future.valid()) {
            pipeline_future.wait();
            pipeline = pipeline_future.get();
        }
#endif
        return pipeline && pipeline->is_valid();
    }
    auto get() -> daxa::ComputePipeline & {
        return *pipeline;
    }
};
struct AsyncManagedRasterPipeline {
    using PipelineT = daxa::RasterPipeline;
    std::shared_ptr<daxa::RasterPipeline> pipeline;
#if ENABLE_THREAD_POOL
    std::shared_ptr<std::promise<std::shared_ptr<daxa::RasterPipeline>>> pipeline_promise;
    std::future<std::shared_ptr<daxa::RasterPipeline>> pipeline_future;
#endif

    auto is_valid() -> bool {
#if ENABLE_THREAD_POOL
        if (pipeline_future.valid()) {
            pipeline_future.wait();
            pipeline = pipeline_future.get();
        }
#endif
        return pipeline && pipeline->is_valid();
    }
    auto get() -> daxa::RasterPipeline & {
        return *pipeline;
    }
};

struct AsyncPipelineManager {
    std::array<daxa::PipelineManager, 8> pipeline_managers;
    std::array<std::mutex, 8> mutexes{};
    std::atomic_uint64_t current_index = 0;
    ThreadPool thread_pool{};

    AsyncPipelineManager(daxa::PipelineManagerInfo info) {
        pipeline_managers = {
            daxa::PipelineManager(info),
            daxa::PipelineManager(info),
            daxa::PipelineManager(info),
            daxa::PipelineManager(info),
            daxa::PipelineManager(info),
            daxa::PipelineManager(info),
            daxa::PipelineManager(info),
            daxa::PipelineManager(info),

            // daxa::PipelineManager(info),
            // daxa::PipelineManager(info),
            // daxa::PipelineManager(info),
            // daxa::PipelineManager(info),
            // daxa::PipelineManager(info),
            // daxa::PipelineManager(info),
            // daxa::PipelineManager(info),
            // daxa::PipelineManager(info),
        };

        thread_pool.start();
    }

    ~AsyncPipelineManager() {
        thread_pool.stop();
    }

    AsyncPipelineManager(AsyncPipelineManager const &) = delete;
    AsyncPipelineManager(AsyncPipelineManager &&) noexcept = delete;
    AsyncPipelineManager &operator=(AsyncPipelineManager const &) = delete;
    AsyncPipelineManager &operator=(AsyncPipelineManager &&) noexcept = delete;

    auto add_compute_pipeline(daxa::ComputePipelineCompileInfo const &info) -> AsyncManagedComputePipeline {
#if ENABLE_THREAD_POOL
        auto pipeline_promise = std::make_shared<std::promise<std::shared_ptr<daxa::ComputePipeline>>>();
        auto result = AsyncManagedComputePipeline{};
        result.pipeline_promise = pipeline_promise;
        result.pipeline_future = pipeline_promise->get_future();
        auto info_copy = info;

        thread_pool.enqueue([this, pipeline_promise, info_copy]() {
            auto [pipeline_manager, lock] = get_pipeline_manager();
            auto compile_result = pipeline_manager.add_compute_pipeline(info_copy);
            if (compile_result.is_err()) {
                debug_utils::Console::add_log(compile_result.message());
                return;
            }
            if (!compile_result.value()->is_valid()) {
                debug_utils::Console::add_log(compile_result.message());
                return;
            }
            pipeline_promise->set_value(compile_result.value());
        });

        return result;
#else
        auto [pipeline_manager, lock] = get_pipeline_manager();
        auto compile_result = pipeline_manager.add_compute_pipeline(info);
        if (compile_result.is_err()) {
            debug_utils::Console::add_log(compile_result.message());
            return {};
        }
        auto result = AsyncManagedComputePipeline{};
        result.pipeline = compile_result.value();
        if (!compile_result.value()->is_valid()) {
            debug_utils::Console::add_log(compile_result.message());
        }
        return result;
#endif
    }
    auto add_raster_pipeline(daxa::RasterPipelineCompileInfo const &info) -> AsyncManagedRasterPipeline {
#if ENABLE_THREAD_POOL
        auto pipeline_promise = std::make_shared<std::promise<std::shared_ptr<daxa::RasterPipeline>>>();
        auto result = AsyncManagedRasterPipeline{};
        result.pipeline_promise = pipeline_promise;
        result.pipeline_future = pipeline_promise->get_future();
        auto info_copy = info;

        thread_pool.enqueue([this, pipeline_promise, info_copy]() {
            auto [pipeline_manager, lock] = get_pipeline_manager();
            auto compile_result = pipeline_manager.add_raster_pipeline(info_copy);
            if (compile_result.is_err()) {
                debug_utils::Console::add_log(compile_result.message());
                return;
            }
            if (!compile_result.value()->is_valid()) {
                debug_utils::Console::add_log(compile_result.message());
                return;
            }
            pipeline_promise->set_value(compile_result.value());
        });

        return result;
#else
        auto [pipeline_manager, lock] = get_pipeline_manager();
        auto compile_result = pipeline_manager.add_raster_pipeline(info);
        if (compile_result.is_err()) {
            debug_utils::Console::add_log(compile_result.message());
            return {};
        }
        auto result = AsyncManagedRasterPipeline{};
        result.pipeline = compile_result.value();
        if (!compile_result.value()->is_valid()) {
            debug_utils::Console::add_log(compile_result.message());
        }
        return result;
#endif
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
#if ENABLE_THREAD_POOL
        while (thread_pool.busy()) {
        }
#endif
    }
    auto reload_all() -> daxa::PipelineReloadResult {
        std::array<daxa::PipelineReloadResult, 8> results;
        for (daxa_u32 i = 0; i < pipeline_managers.size(); ++i) {
            // #if ENABLE_THREAD_POOL
            //             thread_pool.enqueue([this, i, &results]() {
            //                 auto &pipeline_manager = this->pipeline_managers[i];
            //                 auto lock = std::lock_guard{this->mutexes[i]};
            //                 (results)[i] = pipeline_manager.reload_all();
            //             });
            // #else
            auto &pipeline_manager = pipeline_managers[i];
            auto lock = std::lock_guard{mutexes[i]};
            results[i] = pipeline_manager.reload_all();
            // #endif
        }
        // #if ENABLE_THREAD_POOL
        //         while (thread_pool.busy()) {
        //         }
        // #endif
        for (auto const &result : results) {
            if (daxa::holds_alternative<daxa::PipelineReloadError>(result)) {
                return result;
            }
        }
        return results[0];
    }

  private:
    auto get_pipeline_manager() -> std::pair<daxa::PipelineManager &, std::unique_lock<std::mutex>> {
#if ENABLE_THREAD_POOL
        auto index = current_index.fetch_add(1);
        index = (index / 4) % pipeline_managers.size();

#if NDEBUG // Pipeline manager really needs to be internally thread-safe
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
#else
        auto index = 0;
#endif
        return {
            pipeline_managers[index],
            std::unique_lock(mutexes[index]),
        };
    }
};
