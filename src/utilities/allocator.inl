#pragma once

#include <core.inl>

// The "simple" allocator declared here (as well as implemented both here and further
// for the GLSL side in allocator.glsl) is just a simple free-list linear allocator.

#define DECL_SIMPLE_STATIC_ALLOCATOR(AllocatorType_, ElementType_, ElementCount_, IndexType_) \
    struct AllocatorType_ {                                                                   \
        daxa_RWBufferPtr(ElementType_) heap;                                                  \
        daxa_RWBufferPtr(IndexType_) available_element_stack;                                 \
        daxa_RWBufferPtr(IndexType_) released_element_stack;                                  \
        daxa_i32 element_count;                                                               \
        daxa_i32 available_element_stack_size;                                                \
        daxa_i32 released_element_stack_size;                                                 \
    };                                                                                        \
    DAXA_DECL_BUFFER_PTR(AllocatorType_)                                                      \
    CPU_ONLY(DECL_SIMPLE_STATIC_ALLOCATOR_CONSTANTS(AllocatorType_, ElementType_, ElementCount_, IndexType_))

#define DECL_SIMPLE_STATIC_ALLOCATOR_CONSTANTS(AllocatorType_, ElementType_, ElementCount_, IndexType_)                             \
    template <>                                                                                                                     \
    struct StaticAllocatorConstants<AllocatorType_> {                                                                               \
        using AllocatorType = AllocatorType_;                                                                                       \
        using ElementType = ElementType_;                                                                                           \
        using IndexType = IndexType_;                                                                                               \
        static constexpr daxa_u32 MAX_ELEMENTS = ElementCount_;                                                                     \
        static constexpr char const *const allocator_buffer_name = #AllocatorType_ "_allocator_buffer";                             \
        static constexpr char const *const element_buffer_name = #AllocatorType_ "_element_buffer";                                 \
        static constexpr char const *const available_element_stack_buffer_name = #AllocatorType_ "_available_element_stack_buffer"; \
        static constexpr char const *const released_element_stack_buffer_name = #AllocatorType_ "_released_element_stack_buffer";   \
    };

#define SIMPLE_STATIC_ALLOCATOR_BUFFER_USE_N 4
#define SIMPLE_STATIC_ALLOCATOR_USE_BUFFERS(HeapUsage, AllocatorType_)                                                            \
    DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(AllocatorType_), AllocatorType_##_allocator_buffer) \
    DAXA_TH_BUFFER(HeapUsage, AllocatorType_##_heap)                                                                              \
    DAXA_TH_BUFFER(READ_WRITE_CONCURRENT, AllocatorType_##_available_elements)                                     \
    DAXA_TH_BUFFER(READ_WRITE_CONCURRENT, AllocatorType_##_released_elements)

#define SIMPLE_STATIC_ALLOCATOR_BUFFERS_PUSH_USES(AllocatorType_, var_name) \
    daxa_RWBufferPtr(AllocatorType_) var_name = push.uses.AllocatorType_##_allocator_buffer;

#define SIMPLE_STATIC_ALLOCATOR_BUFFER_USES_ASSIGN(TaskHeadName, AllocatorType_, allocator)                       \
    .AllocatorType_##_allocator_buffer = allocator.allocator_buffer.task_resource.view(),                        \
        .AllocatorType_##_heap = allocator.element_buffer.task_resource.view(),                                  \
        .AllocatorType_##_available_elements = allocator.available_element_stack_buffer.task_resource.view(),    \
        .AllocatorType_##_released_elements = allocator.released_element_stack_buffer.task_resource.view()

#if defined(__cplusplus)
template <typename T>
struct StaticAllocatorConstants {
    using AllocatorType = T;
    using ElementType = daxa_u32;
    using IndexType = daxa_u32;
    static constexpr size_t MAX_ELEMENTS = 1;
    static constexpr char const *const allocator_buffer_name = "allocator_buffer";
    static constexpr char const *const element_buffer_name = "element_buffer";
    static constexpr char const *const available_element_stack_buffer_name = "available_element_stack_buffer";
    static constexpr char const *const released_element_stack_buffer_name = "released_element_stack_buffer";
};

template <typename T>
struct StaticAllocatorBufferState {
    TemporalBuffer allocator_buffer;
    TemporalBuffer element_buffer;
    TemporalBuffer available_element_stack_buffer; // todo delete
    TemporalBuffer released_element_stack_buffer; // todo delete

    bool initialized = false;

    void init(GpuContext &gpu_context) {
        allocator_buffer = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(typename StaticAllocatorConstants<T>::AllocatorType),
            .name = StaticAllocatorConstants<T>::allocator_buffer_name,
        });
        element_buffer = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(typename StaticAllocatorConstants<T>::ElementType) * StaticAllocatorConstants<T>::MAX_ELEMENTS,
            .name = StaticAllocatorConstants<T>::element_buffer_name,
        });
        available_element_stack_buffer = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(typename StaticAllocatorConstants<T>::IndexType) * StaticAllocatorConstants<T>::MAX_ELEMENTS,
            .name = StaticAllocatorConstants<T>::available_element_stack_buffer_name,
        });
        released_element_stack_buffer = gpu_context.find_or_add_temporal_buffer({
            .size = sizeof(typename StaticAllocatorConstants<T>::IndexType) * StaticAllocatorConstants<T>::MAX_ELEMENTS,
            .name = StaticAllocatorConstants<T>::released_element_stack_buffer_name,
        });

        gpu_context.frame_task_graph.register_buffer(allocator_buffer.task_resource);
        gpu_context.frame_task_graph.register_buffer(element_buffer.task_resource);
        gpu_context.frame_task_graph.register_buffer(available_element_stack_buffer.task_resource);
        gpu_context.frame_task_graph.register_buffer(released_element_stack_buffer.task_resource);

        // NOTE: gpu_context.startup_task_graph is never submitted/executed by
        // the app right now, so a task added there would silently never run --
        // do this one-time zero-init synchronously instead, matching the
        // pattern GpuContext's own constructor uses for its one-time uploads.
        auto temp_task_graph = daxa::TaskGraph({.device = gpu_context.device, .name = "allocator init"});
        temp_task_graph.register_buffer(allocator_buffer.task_resource);
        temp_task_graph.register_buffer(element_buffer.task_resource);
        temp_task_graph.add_task(
            daxa::InlineTask::Transfer("Allocator State Init")
                .writes(allocator_buffer.task_resource, element_buffer.task_resource)
                .executes([this](daxa::TaskInterface ti) {
                auto staging_buffer = ti.device.create_buffer({
                    .size = sizeof(typename StaticAllocatorConstants<T>::AllocatorType),
                    .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "allocator_staging_buffer",
                });
                ti.recorder.destroy_buffer_deferred(staging_buffer);
                auto *buffer_ptr = ti.device.buffer_host_address_as<typename StaticAllocatorConstants<T>::AllocatorType>(staging_buffer).value();
                *buffer_ptr = typename StaticAllocatorConstants<T>::AllocatorType{
                    .heap = ti.device.device_address(element_buffer.task_resource.id()).value(),
                    .available_element_stack = ti.device.device_address(available_element_stack_buffer.task_resource.id()).value(),
                    .released_element_stack = ti.device.device_address(released_element_stack_buffer.task_resource.id()).value(),
                    .element_count = 0,
                    .available_element_stack_size = 0,
                    .released_element_stack_size = 0,
                };
                ti.recorder.copy_buffer_to_buffer({
                    .src_buffer = staging_buffer,
                    .dst_buffer = allocator_buffer.task_resource.id(),
                    .size = sizeof(typename StaticAllocatorConstants<T>::AllocatorType),
                });
                ti.recorder.clear_buffer({
                    .buffer = element_buffer.task_resource.id(),
                    .size = sizeof(typename StaticAllocatorConstants<T>::ElementType) * StaticAllocatorConstants<T>::MAX_ELEMENTS,
                });
                }));
        temp_task_graph.submit({});
        temp_task_graph.complete({});
        temp_task_graph.execute({});

        initialized = true;
    }
};
#endif
