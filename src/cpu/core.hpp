#pragma once

#include <memory>
#include <thread>
#include <functional>
#include <unordered_map>

#include <daxa/daxa.hpp>
#include <daxa/utils/pipeline_manager.hpp>
#include <daxa/utils/imgui.hpp>
#include <daxa/utils/task_graph.hpp>

#include <cpu/app_ui.hpp>

using BDA = daxa::DeviceAddress;

static inline constexpr size_t FRAMES_IN_FLIGHT = 1;

struct PingPongImage_impl {
    using ResourceType = daxa::ImageId;
    using ResourceInfoType = daxa::ImageInfo;
    using TaskResourceType = daxa::TaskImage;
    using TaskResourceInfoType = daxa::TaskImageInfo;

    static auto create(daxa::Device &device, ResourceInfoType const &info) -> ResourceType {
        return device.create_image(info);
    }
    static void destroy(daxa::Device &device, ResourceType rsrc_id) {
        device.destroy_image(rsrc_id);
    }
    static auto create_task_resource(ResourceType rsrc_id, std::string const &name) -> TaskResourceType {
        return TaskResourceType(TaskResourceInfoType{.initial_images = {std::array{rsrc_id}}, .name = name});
    }
};

struct PingPongBuffer_impl {
    using ResourceType = daxa::BufferId;
    using ResourceInfoType = daxa::BufferInfo;
    using TaskResourceType = daxa::TaskBuffer;
    using TaskResourceInfoType = daxa::TaskBufferInfo;

    static auto create(daxa::Device &device, ResourceInfoType const &info) -> ResourceType {
        return device.create_buffer(info);
    }
    static void destroy(daxa::Device &device, ResourceType rsrc_id) {
        device.destroy_buffer(rsrc_id);
    }
    static auto create_task_resource(ResourceType rsrc_id, std::string const &name) -> TaskResourceType {
        return TaskResourceType(TaskResourceInfoType{.initial_buffers = {std::array{rsrc_id}}, .name = name});
    }
};

template <typename Impl>
struct PingPongResource {
    using ResourceType = typename Impl::ResourceType;
    using ResourceInfoType = typename Impl::ResourceInfoType;
    using TaskResourceType = typename Impl::TaskResourceType;
    using TaskResourceInfoType = typename Impl::TaskResourceInfoType;

    struct Resources {
        Resources() = default;
        Resources(Resources const &) = delete;
        Resources(Resources &&) = delete;
        Resources &operator=(Resources const &) = delete;
        Resources &operator=(Resources &&other) {
            std::swap(this->device, other.device);
            std::swap(this->resource_a, other.resource_a);
            std::swap(this->resource_b, other.resource_b);
            return *this;
        }

        daxa::Device device{};
        ResourceType resource_a{};
        ResourceType resource_b{};

        ~Resources() {
            if (!resource_a.is_empty()) {
                Impl::destroy(device, resource_a);
                Impl::destroy(device, resource_b);
            }
        }
    };
    struct TaskResources {
        TaskResourceType output_resource;
        TaskResourceType history_resource;
    };
    Resources resources;
    TaskResources task_resources;

    auto get(daxa::Device a_device, ResourceInfoType const &a_info) -> std::pair<TaskResourceType &, TaskResourceType &> {
        if (!resources.device.is_valid()) {
            resources.device = a_device;
        }
        // assert(resources.device == a_device);
        if (resources.resource_a.is_empty()) {
            auto info_a = a_info;
            auto info_b = a_info;
            info_a.name = std::string(info_a.name.view()) + "_a";
            info_b.name = std::string(info_b.name.view()) + "_b";
            resources.resource_a = Impl::create(a_device, info_a);
            resources.resource_b = Impl::create(a_device, info_b);
            task_resources.output_resource = Impl::create_task_resource(resources.resource_a, std::string(a_info.name.view()));
            task_resources.history_resource = Impl::create_task_resource(resources.resource_b, std::string(a_info.name.view()) + "_hist");
        }
        return {task_resources.output_resource, task_resources.history_resource};
    }
};

using PingPongImage = PingPongResource<PingPongImage_impl>;
using PingPongBuffer = PingPongResource<PingPongBuffer_impl>;

#define ENABLE_THREAD_POOL true

#if ENABLE_THREAD_POOL
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#endif

namespace {
    struct ThreadPool {
        void start() {
#if ENABLE_THREAD_POOL
            uint32_t const num_threads = 8; // std::thread::hardware_concurrency();
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

            // daxa::PipelineManager(info),
            // daxa::PipelineManager(info),
            // daxa::PipelineManager(info),
            // daxa::PipelineManager(info),
            // daxa::PipelineManager(info),
            // daxa::PipelineManager(info),
            // daxa::PipelineManager(info),
            // daxa::PipelineManager(info),
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

    auto add_compute_pipeline(daxa::ComputePipelineCompileInfo const &info) -> AsyncManagedComputePipeline {
#if ENABLE_THREAD_POOL
        auto pipeline_promise = std::make_shared<std::promise<std::shared_ptr<daxa::ComputePipeline>>>();
        auto result = AsyncManagedComputePipeline{};
        result.pipeline_promise = pipeline_promise;
        result.pipeline_future = pipeline_promise->get_future();
        auto info_copy = info;

        atomics->thread_pool.enqueue([this, pipeline_promise, info_copy]() {
            auto [pipeline_manager, lock] = get_pipeline_manager();
            auto compile_result = pipeline_manager.add_compute_pipeline(info_copy);
            if (compile_result.is_err()) {
                AppUi::Console::s_instance->add_log(compile_result.message());
                return;
            }
            if (!compile_result.value()->is_valid()) {
                AppUi::Console::s_instance->add_log(compile_result.message());
                return;
            }
            pipeline_promise->set_value(compile_result.value());
        });

        return result;
#else
        auto [pipeline_manager, lock] = get_pipeline_manager();
        auto compile_result = pipeline_manager.add_compute_pipeline(info);
        if (compile_result.is_err()) {
            AppUi::Console::s_instance->add_log(compile_result.message());
            return {};
        }
        auto result = AsyncManagedComputePipeline{};
        result.pipeline = compile_result.value();
        if (!compile_result.value()->is_valid()) {
            AppUi::Console::s_instance->add_log(compile_result.message());
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

        atomics->thread_pool.enqueue([this, pipeline_promise, info_copy]() {
            auto [pipeline_manager, lock] = get_pipeline_manager();
            auto compile_result = pipeline_manager.add_raster_pipeline(info_copy);
            if (compile_result.is_err()) {
                AppUi::Console::s_instance->add_log(compile_result.message());
                return;
            }
            if (!compile_result.value()->is_valid()) {
                AppUi::Console::s_instance->add_log(compile_result.message());
                return;
            }
            pipeline_promise->set_value(compile_result.value());
        });

        return result;
#else
        auto [pipeline_manager, lock] = get_pipeline_manager();
        auto compile_result = pipeline_manager.add_raster_pipeline(info);
        if (compile_result.is_err()) {
            AppUi::Console::s_instance->add_log(compile_result.message());
            return {};
        }
        auto result = AsyncManagedRasterPipeline{};
        result.pipeline = compile_result.value();
        if (!compile_result.value()->is_valid()) {
            AppUi::Console::s_instance->add_log(compile_result.message());
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
        while (atomics->thread_pool.busy()) {
        }
#endif
    }
    auto reload_all() -> daxa::PipelineReloadResult {
        std::array<daxa::PipelineReloadResult, 8> results;
        for (daxa_u32 i = 0; i < pipeline_managers.size(); ++i) {
            // #if ENABLE_THREAD_POOL
            //             atomics->thread_pool.enqueue([this, i, &results]() {
            //                 auto &pipeline_manager = this->pipeline_managers[i];
            //                 auto lock = std::lock_guard{this->atomics->mutexes[i]};
            //                 (results)[i] = pipeline_manager.reload_all();
            //             });
            // #else
            auto &pipeline_manager = pipeline_managers[i];
            auto lock = std::lock_guard{atomics->mutexes[i]};
            results[i] = pipeline_manager.reload_all();
            // #endif
        }
        // #if ENABLE_THREAD_POOL
        //         while (atomics->thread_pool.busy()) {
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
        auto index = atomics->current_index.fetch_add(1);
        index = (index / 4) % pipeline_managers.size();

#if NDEBUG // Pipeline manager really needs to be internally thread-safe
        // try to find one that's not locked, otherwise we'll fall back on the index above.
        for (daxa_u32 i = 0; i < pipeline_managers.size(); ++i) {
            auto &mtx = this->atomics->mutexes[i];
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
            std::unique_lock(atomics->mutexes[index]),
        };
    }
};

template <typename TaskHeadT, typename PushT, typename InfoT, typename PipelineT>
using TaskCallback = void(daxa::TaskInterface const &ti, typename PipelineT::PipelineT &pipeline /*, typename TaskHeadT::Uses &uses*/, PushT &push, InfoT const &info);

struct NoTaskInfo {
};

template <typename PushT>
constexpr auto push_constant_size() -> uint32_t {
    return static_cast<uint32_t>(((sizeof(PushT) & ~0x3) + 7) & ~7);
}

template <typename PushT>
void set_push_constant(daxa::TaskInterface const &ti, PushT push) {
    uint32_t offset = 0;
    if constexpr (push_constant_size<PushT>() != 0) {
        ti.recorder.push_constant(push);
        offset = push_constant_size<PushT>();
    }
    // ti.copy_task_head_to(&push.views);
    ti.recorder.push_constant_vptr({ti.attachment_shader_data.data(), ti.attachment_shader_data.size(), offset});
}

template <typename PushT>
void set_push_constant(daxa::TaskInterface const &ti, daxa::RenderCommandRecorder &render_recorder, PushT push) {
    uint32_t offset = 0;
    if constexpr (sizeof(PushT) >= 4) {
        render_recorder.push_constant(push);
        offset = sizeof(PushT);
    }
    render_recorder.push_constant_vptr({ti.attachment_shader_data.data(), ti.attachment_shader_data.size(), offset});
}

template <typename TaskHeadT, typename PushT, typename InfoT, typename PipelineT>
struct Task : TaskHeadT {
    daxa::ShaderSource source;
    std::vector<daxa::ShaderDefine> extra_defines{};
    TaskHeadT::Views views{};
    TaskCallback<TaskHeadT, PushT, InfoT, PipelineT> *callback_{};
    InfoT info{};
    // Not set by user
    // std::string_view name = TaskHeadT::NAME;
    std::shared_ptr<PipelineT> pipeline;
    void callback(daxa::TaskInterface const &ti) {
        auto push = PushT{};
        if (!pipeline->is_valid()) {
            return;
        }
        callback_(ti, pipeline->get(), push, info);
    }
};

template <typename TaskHeadT, typename PushT, typename InfoT>
struct Task<TaskHeadT, PushT, InfoT, AsyncManagedRasterPipeline> : TaskHeadT {
    daxa::ShaderSource vert_source;
    daxa::ShaderSource frag_source;
    std::vector<daxa::RenderAttachment> color_attachments{};
    daxa::Optional<daxa::DepthTestInfo> depth_test{};
    daxa::RasterizerInfo raster{};
    std::vector<daxa::ShaderDefine> extra_defines{};
    TaskHeadT::Views views{};
    TaskCallback<TaskHeadT, PushT, InfoT, AsyncManagedRasterPipeline> *callback_{};
    InfoT info{};
    // Not set by user
    // std::string_view name = TaskHeadT::NAME;
    std::shared_ptr<AsyncManagedRasterPipeline> pipeline;
    void callback(daxa::TaskInterface const &ti) {
        auto push = PushT{};
        // ti.copy_task_head_to(&push.uses);
        if (!pipeline->is_valid()) {
            return;
        }
        callback_(ti, pipeline->get(), push, info);
    }
};

template <typename TaskHeadT, typename PushT, typename InfoT>
using ComputeTask = Task<TaskHeadT, PushT, InfoT, AsyncManagedComputePipeline>;

template <typename TaskHeadT, typename PushT, typename InfoT>
using RasterTask = Task<TaskHeadT, PushT, InfoT, AsyncManagedRasterPipeline>;

struct RecordContext {
    daxa::Device device;
    daxa::TaskGraph task_graph;
    AsyncPipelineManager *pipeline_manager;
    daxa_u32vec2 render_resolution;
    daxa_u32vec2 output_resolution;

    daxa::TaskImageView task_swapchain_image;
    daxa::TaskImageView task_blue_noise_vec2_image;
    daxa::TaskImageView task_debug_texture;
    daxa::TaskBufferView task_input_buffer;
    daxa::TaskBufferView task_globals_buffer;

    std::unordered_map<std::string, std::shared_ptr<AsyncManagedComputePipeline>> *compute_pipelines;
    std::unordered_map<std::string, std::shared_ptr<AsyncManagedRasterPipeline>> *raster_pipelines;

    template <typename TaskHeadT, typename PushT, typename InfoT, typename PipelineT>
    auto find_or_add_pipeline(Task<TaskHeadT, PushT, InfoT, PipelineT> &task, std::string const &shader_id) {
        auto push_constant_size = static_cast<uint32_t>(::push_constant_size<PushT>() + TaskHeadT::attachment_shader_data_size());
        if constexpr (std::is_same_v<PipelineT, AsyncManagedComputePipeline>) {
            auto pipe_iter = compute_pipelines->find(shader_id);
            if (pipe_iter == compute_pipelines->end()) {
                task.extra_defines.push_back({std::string{TaskHeadT::name()} + "Shader", "1"});
                auto emplace_result = compute_pipelines->emplace(
                    shader_id,
                    std::make_shared<AsyncManagedComputePipeline>(pipeline_manager->add_compute_pipeline({
                        .shader_info = {
                            .source = task.source,
                            .compile_options = {.defines = task.extra_defines},
                        },
                        .push_constant_size = push_constant_size,
                        .name = std::string{TaskHeadT::name()},
                    })));
                pipe_iter = emplace_result.first;
            }
            return pipe_iter;
        } else if constexpr (std::is_same_v<PipelineT, AsyncManagedRasterPipeline>) {
            auto pipe_iter = raster_pipelines->find(shader_id);
            // TODO: if we found a pipeline, but it has differing info such as attachments or raster info,
            // we should destroy that old one and create a new one.
            if (pipe_iter == raster_pipelines->end()) {
                task.extra_defines.push_back({std::string{TaskHeadT::name()} + "Shader", "1"});
                auto emplace_result = raster_pipelines->emplace(
                    shader_id,
                    std::make_shared<AsyncManagedRasterPipeline>(pipeline_manager->add_raster_pipeline({
                        .vertex_shader_info = daxa::ShaderCompileInfo{
                            .source = task.vert_source,
                            .compile_options = {.defines = task.extra_defines},
                        },
                        .fragment_shader_info = daxa::ShaderCompileInfo{
                            .source = task.frag_source,
                            .compile_options = {.defines = task.extra_defines},
                        },
                        .color_attachments = task.color_attachments,
                        .depth_test = task.depth_test,
                        .raster = task.raster,
                        .push_constant_size = push_constant_size,
                        .name = std::string{TaskHeadT::name()},
                    })));
                pipe_iter = emplace_result.first;
            }
            return pipe_iter;
        }
    }

    template <typename TaskHeadT, typename PushT, typename InfoT, typename PipelineT>
    void add(Task<TaskHeadT, PushT, InfoT, PipelineT> &&task) {
        auto shader_id = std::string{TaskHeadT::name()};
        for (auto const &define : task.extra_defines) {
            shader_id.append(define.name);
            shader_id.append(define.value);
        }
        auto pipe_iter = find_or_add_pipeline<TaskHeadT, PushT, InfoT, PipelineT>(task, shader_id);
        task.pipeline = pipe_iter->second;
        task_graph.add_task(std::move(task));
    }
};
