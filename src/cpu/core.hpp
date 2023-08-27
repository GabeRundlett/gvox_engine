#pragma once

#include <memory>
#include <thread>
#include <functional>

#include <ftl/task_counter.h>
#include <ftl/task_scheduler.h>

#include <daxa/daxa.hpp>
#include <daxa/utils/pipeline_manager.hpp>
#include <daxa/utils/imgui.hpp>
#include <daxa/utils/task_graph.hpp>
#include <daxa/utils/math_operators.hpp>
using namespace daxa::math_operators;

using BDA = daxa::BufferDeviceAddress;

static inline constexpr usize FRAMES_IN_FLIGHT = 1;

struct RecordContext {
    daxa::Device device;
    daxa::TaskGraph task_graph;
    u32vec2 render_resolution;
    u32vec2 output_resolution;

    daxa::TaskImageView task_swapchain_image;
    daxa::TaskImageView task_blue_noise_vec2_image;
    daxa::TaskImageView task_debug_texture;
    daxa::TaskBufferView task_input_buffer;
    daxa::TaskBufferView task_globals_buffer;
};

struct PingPongImage {
    struct Resources {
        Resources() = default;
        Resources(Resources const &) = delete;
        Resources(Resources &&) = delete;
        Resources &operator=(Resources const &) = delete;
        Resources &operator=(Resources &&other) {
            std::swap(this->device, other.device);
            std::swap(this->image_a, other.image_a);
            std::swap(this->image_b, other.image_b);
            return *this;
        }

        daxa::Device device{};
        daxa::ImageId image_a{};
        daxa::ImageId image_b{};

        ~Resources() {
            if (!image_a.is_empty()) {
                device.destroy_image(image_a);
                device.destroy_image(image_b);
            }
        }
    };
    struct TaskResources {
        daxa::TaskImage output_image;
        daxa::TaskImage history_image;
    };
    Resources resources;
    TaskResources task_resources;

    auto get(daxa::Device a_device, daxa::ImageInfo const &a_info) -> std::pair<daxa::TaskImage &, daxa::TaskImage &> {
        if (!resources.device) {
            resources.device = a_device;
        }
        assert(resources.device == a_device);
        if (resources.image_a.is_empty()) {
            auto info_a = a_info;
            auto info_b = a_info;
            info_a.name += "_a";
            info_b.name += "_b";
            resources.image_a = a_device.create_image(info_a);
            resources.image_b = a_device.create_image(info_b);
            task_resources.output_image = daxa::TaskImage(daxa::TaskImageInfo{
                .initial_images = {.images = std::array{resources.image_a}},
                .name = a_info.name,
            });
            task_resources.history_image = daxa::TaskImage(daxa::TaskImageInfo{
                .initial_images = {.images = std::array{resources.image_b}},
                .name = a_info.name + "_history",
            });
        }
        return {task_resources.output_image, task_resources.history_image};
    }
};

#define ENABLE_THREAD_POOL 0

#if ENABLE_THREAD_POOL
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#endif

namespace {
    struct ThreadPool {
        void start() {
#if ENABLE_THREAD_POOL
            uint32_t const num_threads = std::thread::hardware_concurrency();
            threads.resize(num_threads);
            for (uint32_t i = 0; i < num_threads; i++) {
                threads.at(i) = std::thread(&ThreadPool::thread_loop, this);
            }
#endif
        }
        void enqueue(std::function<void()> const &job) {
#if ENABLE_THREAD_POOL
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                jobs.push(job);
            }
            mutex_condition.notify_one();
#else
            job();
#endif
        }
        void stop() {
#if ENABLE_THREAD_POOL
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                should_terminate = true;
            }
            mutex_condition.notify_all();
            for (std::thread &active_thread : threads) {
                active_thread.join();
            }
            threads.clear();
#endif
        }
        auto busy() -> bool {
#if ENABLE_THREAD_POOL
            bool pool_busy;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                pool_busy = !jobs.empty();
            }
            return pool_busy;
#else
            return false;
#endif
        }

      private:
        void thread_loop() {
#if ENABLE_THREAD_POOL
            while (true) {
                std::function<void()> job;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    mutex_condition.wait(lock, [this] {
                        return !jobs.empty() || should_terminate;
                    });
                    if (should_terminate) {
                        return;
                    }
                    job = jobs.front();
                    jobs.pop();
                }
                job();
            }
#endif
        }
#if ENABLE_THREAD_POOL
        bool should_terminate = false;
        std::mutex queue_mutex;
        std::condition_variable mutex_condition;
        std::vector<std::thread> threads;
        std::queue<std::function<void()>> jobs;
#endif
    };
} // namespace

struct AsyncPipelineManager {
    std::array<daxa::PipelineManager, 8> pipeline_managers;
    struct Atomics {
        std::array<std::mutex, 8> mutexes{};
        std::atomic_uint64_t current_index = 0;
        ThreadPool thread_pool{};
    };
    std::unique_ptr<Atomics> atomics;

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
        };
        atomics = std::make_unique<Atomics>();

        atomics->thread_pool.start();
    }

    ~AsyncPipelineManager() {
        atomics->thread_pool.stop();
    }

    AsyncPipelineManager(AsyncPipelineManager const &) = delete;
    AsyncPipelineManager(AsyncPipelineManager &&) noexcept = default;
    AsyncPipelineManager &operator=(AsyncPipelineManager const &) = delete;
    AsyncPipelineManager &operator=(AsyncPipelineManager &&) noexcept = default;

    auto add_compute_pipeline(daxa::ComputePipelineCompileInfo const &info) -> daxa::Result<std::shared_ptr<daxa::ComputePipeline>> {
        auto [pipeline_manager, lock] = get_pipeline_manager();
        return pipeline_manager.add_compute_pipeline(info);
    }
    auto add_raster_pipeline(daxa::RasterPipelineCompileInfo const &info) -> daxa::Result<std::shared_ptr<daxa::RasterPipeline>> {
        auto [pipeline_manager, lock] = get_pipeline_manager();
        return pipeline_manager.add_raster_pipeline(info);
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
    auto reload_all() -> daxa::PipelineReloadResult {
#if ENABLE_THREAD_POOL

#endif
        std::array<daxa::PipelineReloadResult, 8> results;
        for (u32 i = 0; i < pipeline_managers.size(); ++i) {
#if ENABLE_THREAD_POOL
            atomics->thread_pool.enqueue([this, i, &results]() {
                auto &pipeline_manager = this->pipeline_managers[i];
                auto lock = std::lock_guard{this->atomics->mutexes[i]};
                (results)[i] = pipeline_manager.reload_all();
            });
#else
            auto &pipeline_manager = pipeline_managers[i];
            auto lock = std::lock_guard{atomics->mutexes[i]};
            results[i] = pipeline_manager.reload_all();
#endif
        }
#if ENABLE_THREAD_POOL
        while (atomics->thread_pool.busy()) {
        }
#endif
        for (auto const &result : results) {
            if (std::holds_alternative<daxa::PipelineReloadError>(result)) {
                return result;
            }
        }
        return results[0];
    }

  private:
    auto get_pipeline_manager() -> std::pair<daxa::PipelineManager &, std::unique_lock<std::mutex>> {
#if ENABLE_THREAD_POOL
        auto index = atomics->current_index.fetch_add(1);
        index = index % pipeline_managers.size();
#else
        auto index = 0;
#endif
        return {
            pipeline_managers[index],
            std::unique_lock(atomics->mutexes[index]),
        };
    }
};
