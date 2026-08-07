#include "thread_pool.hpp"

#include <thread>
#include <optional>
#include <vector>
#include <deque>
#include <condition_variable>
#include <mutex>

enum struct TaskPriority {
    LOW,
    HIGH
};

struct VirtualTask {
    virtual ~VirtualTask() = default;
    virtual void callback(uint32_t chunk_index, uint32_t thread_index) = 0;

    uint32_t chunk_count = {};
    uint32_t not_finished = {};
    uint32_t started = {};
};

struct TaskChunk {
    std::shared_ptr<VirtualTask> task = {};
    uint32_t chunk_index = {};
};

static constexpr uint32_t EXIT_CHUNK_CODE = std::numeric_limits<uint32_t>::max();
static constexpr uint32_t NO_MORE_CHUNKS_CODE = std::numeric_limits<uint32_t>::max();
static constexpr uint32_t EXTERNAL_THREAD_INDEX = std::numeric_limits<uint32_t>::max();

struct ThreadPool {
  public:
    ThreadPool(std::optional<uint32_t> thread_count = std::nullopt);
    ThreadPool(ThreadPool &&) = default;
    ThreadPool &operator=(ThreadPool &&) = default;
    ThreadPool(ThreadPool const &) = delete;
    ThreadPool &operator=(ThreadPool const &) = delete;
    ~ThreadPool();
    void blocking_dispatch(std::shared_ptr<VirtualTask> task, TaskPriority priority = TaskPriority::LOW);
    void async_dispatch(std::shared_ptr<VirtualTask> task, TaskPriority priority = TaskPriority::LOW);
    void block_on(std::shared_ptr<VirtualTask> task);

  private:
    struct SharedData {
        // Signaled whenever there is work added to one of the work queues
        std::condition_variable work_available = {};
        // Signaled whenever a worker thread detects that it is finishing the last chunk of a task
        std::condition_variable work_done = {};

        std::mutex threadpool_mutex = {};
        std::deque<TaskChunk> high_priority_tasks = {};
        std::deque<TaskChunk> low_priority_tasks = {};
        bool kill = false;
    };
    static void worker(std::shared_ptr<ThreadPool::SharedData> shared_data, uint32_t thread_id);
    std::shared_ptr<SharedData> shared_data = {};
    std::vector<std::thread> worker_threads = {};
};

static ThreadPool s_instance = ThreadPool();

ThreadPool::~ThreadPool() {
    {
        std::unique_lock lock{shared_data->threadpool_mutex};
        shared_data->kill = true;
        shared_data->work_available.notify_all();
    }
    for (auto &worker : worker_threads) {
        worker.join();
    }
}

void ThreadPool::worker(std::shared_ptr<ThreadPool::SharedData> shared_data, uint32_t thread_index) {
    std::unique_lock lock{shared_data->threadpool_mutex};
    while (true) {
        shared_data->work_available.wait(
            lock, [&] { return !shared_data->high_priority_tasks.empty() || !shared_data->low_priority_tasks.empty() || shared_data->kill; });
        if (shared_data->kill) {
            return;
        }

        bool const high_priority_work_available = !shared_data->high_priority_tasks.empty();
        auto &selected_queue = high_priority_work_available ? shared_data->high_priority_tasks : shared_data->low_priority_tasks;
        TaskChunk current_chunk = std::move(selected_queue.front());
        selected_queue.pop_front();
        current_chunk.task->started += 1;
        lock.unlock();

        current_chunk.task->callback(current_chunk.chunk_index, thread_index);

        lock.lock();
        current_chunk.task->not_finished -= 1;
        // Working on last chunk of a task, notify in case there is a thread waiting for this task to be done
        if (current_chunk.task->not_finished == 0) {
            shared_data->work_done.notify_all();
        }
    }
}

ThreadPool::ThreadPool(std::optional<uint32_t> thread_count) {
    uint32_t const real_thread_count = thread_count.value_or(std::thread::hardware_concurrency());
    shared_data = std::make_shared<SharedData>();
    for (uint32_t thread_index = 0; thread_index < real_thread_count; thread_index++) {
        worker_threads.push_back({
            std::thread([=, this]() { ThreadPool::worker(shared_data, thread_index); }),
        });
    }
}

void ThreadPool::blocking_dispatch(std::shared_ptr<VirtualTask> task, TaskPriority priority) {
    // Don't need mutex here as no thread is working on this task yet
    task->not_finished = task->chunk_count;
    auto &selected_queue = priority == TaskPriority::HIGH ? shared_data->high_priority_tasks : shared_data->low_priority_tasks;

    // chunk_index 0 will be worked on by this thread
    std::unique_lock lock{shared_data->threadpool_mutex};

    for (uint32_t chunk_index = 1; chunk_index < task->chunk_count; chunk_index++) {
        selected_queue.push_back({task, chunk_index});
    }

    shared_data->work_available.notify_all();
    // Contribute to finishing this task from this thread
    uint32_t current_chunk_index = 0;
    bool worked_on_last_chunk = false;
    while (current_chunk_index != NO_MORE_CHUNKS_CODE) {
        task->started += 1;

        lock.unlock();
        task->callback(current_chunk_index, EXTERNAL_THREAD_INDEX);
        lock.lock();

        task->not_finished -= 1;
        bool more_chunks_in_queue = (task->started != task->chunk_count);
        if (more_chunks_in_queue) {
            current_chunk_index = selected_queue.front().chunk_index;
            selected_queue.pop_front();
        } else {
            current_chunk_index = NO_MORE_CHUNKS_CODE;
        }

        worked_on_last_chunk = (task->not_finished == 0);
    }

    if (!worked_on_last_chunk) {
        // This thread was not the last one working on this task, therefore we wait here to be notified once
        // the last worker thread processing this task is done
        shared_data->work_done.wait(lock, [&] { return task->not_finished == 0; });
    }
}

void ThreadPool::async_dispatch(std::shared_ptr<VirtualTask> task, TaskPriority priority) {
    task->not_finished = task->chunk_count;
    auto &selected_queue = priority == TaskPriority::HIGH ? shared_data->high_priority_tasks : shared_data->low_priority_tasks;
    {
        std::lock_guard lock(shared_data->threadpool_mutex);
        for (uint32_t chunk_index = 0; chunk_index < task->chunk_count; chunk_index++) {
            selected_queue.push_back({task, chunk_index});
        }
    }
    shared_data->work_available.notify_all();
}

void ThreadPool::block_on(std::shared_ptr<VirtualTask> task) {
    std::unique_lock lock{shared_data->threadpool_mutex};
    shared_data->work_done.wait(lock, [&] { return task->not_finished == 0; });
}

struct SimpleTask : VirtualTask {
    thread_pool::Func *func;
    void *user_ptr;
    SimpleTask(thread_pool::Func *func, void *user_ptr) : func{func}, user_ptr{user_ptr} {
        chunk_count = 1;
    }

    virtual void callback(uint32_t chunk_index, uint32_t thread_index) override {
        func(user_ptr);
    }
};

struct thread_pool::TaskState {
    std::shared_ptr<VirtualTask> task;
};

thread_pool::Task thread_pool::create_task(Func *func, void *user_ptr) {
    auto *result = new TaskState{std::make_shared<SimpleTask>(func, user_ptr)};
    return result;
}
void thread_pool::destroy_task(Task task) {
    delete task;
}

void thread_pool::async_dispatch(Task task) {
    s_instance.async_dispatch(task->task);
}
void thread_pool::wait(Task task) {
    s_instance.block_on(task->task);
}
