#pragma once

namespace thread_pool {
    struct TaskState;
    using Task = TaskState *;

    using Func = void(void *);
    Task create_task(Func *func, void *user_ptr);
    void destroy_task(Task task);

    void async_dispatch(Task task);
    void wait(Task task);
} // namespace thread_pool
