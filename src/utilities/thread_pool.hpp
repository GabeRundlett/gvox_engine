#pragma once

#include <functional>

#define ENABLE_THREAD_POOL true

#if ENABLE_THREAD_POOL
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <vector>
#endif

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
