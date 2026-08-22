#pragma once

namespace thread_pool {
    struct TaskState;
    using Task = TaskState *;

    using Func = void(void *);
    Task create_task(Func *func, void *user_ptr);
    void destroy_task(Task task);

    void async_dispatch(Task task);
    void wait(Task task);
    // Non-blocking completion check. Safe to call repeatedly from the
    // dispatching thread to poll a task without stalling on it.
    bool is_done(Task task);

    // Runs func(user_ptr, i) for i in [0, count), spread across the pool, and
    // blocks until all have finished. The calling thread participates.
    using IndexedFunc = void(void *, int);
    void parallel_for(int count, IndexedFunc *func, void *user_ptr);
} // namespace thread_pool
