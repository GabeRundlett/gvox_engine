#pragma once

#include <core.inl>

// The "simple" allocator declared here (as well as implemented both here and further
// for the GLSL side in allocator.glsl) is just a simple free-list linear allocator.

#define DECL_SIMPLE_ALLOCATOR(AllocatorType_, ElementType_, ElementMultiplier_, IndexType_, MaxAllocPerFrame_) \
    struct AllocatorType_ {                                                                                    \
        daxa_RWBufferPtr(ElementType_) heap;                                                                   \
        daxa_RWBufferPtr(IndexType_) available_element_stack;                                                  \
        daxa_RWBufferPtr(IndexType_) released_element_stack;                                                   \
        daxa_i32 element_count;                                                                                \
        daxa_i32 available_element_stack_size;                                                                 \
        daxa_i32 released_element_stack_size;                                                                  \
    };                                                                                                         \
    struct AllocatorType_##GpuOutput {                                                                         \
        daxa_u32 current_element_count;                                                                        \
    };                                                                                                         \
    DAXA_DECL_BUFFER_PTR(AllocatorType_)                                                                       \
    CPU_ONLY(DECL_SIMPLE_ALLOCATOR_CONSTANTS(AllocatorType_, ElementType_, ElementMultiplier_, IndexType_, MaxAllocPerFrame_))

#define DECL_SIMPLE_ALLOCATOR_CONSTANTS(AllocatorType_, ElementType_, ElementMultiplier_, IndexType_, MaxAllocPerFrame_)            \
    template <>                                                                                                                     \
    struct AllocatorConstants<AllocatorType_> {                                                                                     \
        using AllocatorType = AllocatorType_;                                                                                       \
        using ElementType = ElementType_;                                                                                           \
        using IndexType = IndexType_;                                                                                               \
        static constexpr size_t ELEMENT_MULTIPLIER = ElementMultiplier_;                                                            \
        static constexpr daxa_u32 MAX_ELEMENT_ALLOCATIONS_PER_FRAME = MaxAllocPerFrame_;                                            \
        static constexpr char const *const task_allocator_buffer_name = "task_" #AllocatorType_ "_allocator_buffer";                \
        static constexpr char const *const task_element_buffer_name = "task_" #AllocatorType_ "_element_buffer";                    \
        static constexpr char const *const task_old_element_buffer_name = "task" #AllocatorType_ "_old_element_buffer";             \
        static constexpr char const *const allocator_buffer_name = #AllocatorType_ "_allocator_buffer";                             \
        static constexpr char const *const element_buffer_name = #AllocatorType_ "_element_buffer";                                 \
        static constexpr char const *const available_element_stack_buffer_name = #AllocatorType_ "_available_element_stack_buffer"; \
        static constexpr char const *const released_element_stack_buffer_name = #AllocatorType_ "_released_element_stack_buffer";   \
    };

#if defined(__cplusplus)
template <typename T>
struct AllocatorConstants {
    using AllocatorType = T;
    using ElementType = daxa_u32;
    using IndexType = daxa_u32;
    static constexpr size_t ELEMENT_MULTIPLIER = 1;
    static constexpr daxa_u32 MAX_ELEMENT_ALLOCATIONS_PER_FRAME = 1;
    static constexpr char const *const task_allocator_buffer_name = "task_allocator_buffer";
    static constexpr char const *const task_element_buffer_name = "task_element_buffer";
    static constexpr char const *const task_old_element_buffer_name = "task_old_element_buffer";
    static constexpr char const *const allocator_buffer_name = "allocator_buffer";
    static constexpr char const *const element_buffer_name = "element_buffer";
    static constexpr char const *const available_element_stack_buffer_name = "available_element_stack_buffer";
    static constexpr char const *const released_element_stack_buffer_name = "released_element_stack_buffer";
};

template <typename T>
struct AllocatorBufferState {
    daxa::Device device;
    daxa::BufferId allocator_buffer;
    daxa::BufferId element_buffer;
    daxa::BufferId available_element_stack_buffer;
    daxa::BufferId released_element_stack_buffer;
    daxa::TaskBuffer task_allocator_buffer{{.name = AllocatorConstants<T>::task_allocator_buffer_name}};
    daxa::TaskBuffer task_element_buffer{{.name = AllocatorConstants<T>::task_element_buffer_name}};
    daxa::TaskBuffer task_old_element_buffer{{.name = AllocatorConstants<T>::task_old_element_buffer_name}};
    daxa_u32 current_element_count = 0;
    daxa_u32 next_element_count = 0;
    daxa_u32 prev_element_count = 0;
    void create(GpuContext &gpu_context) {
        device = gpu_context.device;
        constexpr auto MAX_ELEMENT_ALLOCATIONS_PER_FRAME = AllocatorConstants<T>::MAX_ELEMENT_ALLOCATIONS_PER_FRAME;
        daxa_u32 element_count = (FRAMES_IN_FLIGHT + 1) * MAX_ELEMENT_ALLOCATIONS_PER_FRAME;
        current_element_count = element_count;
        allocator_buffer = device.create_buffer({
            .size = sizeof(typename AllocatorConstants<T>::AllocatorType),
            .name = AllocatorConstants<T>::allocator_buffer_name,
        });
        element_buffer = device.create_buffer({
            .size = sizeof(typename AllocatorConstants<T>::ElementType) * AllocatorConstants<T>::ELEMENT_MULTIPLIER * current_element_count,
            .name = AllocatorConstants<T>::element_buffer_name,
        });
        available_element_stack_buffer = device.create_buffer({
            .size = sizeof(typename AllocatorConstants<T>::IndexType) * current_element_count,
            .name = AllocatorConstants<T>::available_element_stack_buffer_name,
        });
        released_element_stack_buffer = device.create_buffer({
            .size = sizeof(typename AllocatorConstants<T>::IndexType) * current_element_count,
            .name = AllocatorConstants<T>::released_element_stack_buffer_name,
        });
        task_allocator_buffer.set_buffers({.buffers = std::array{allocator_buffer}});
        task_element_buffer.set_buffers({
            .buffers = std::array{
                element_buffer,
                available_element_stack_buffer,
                released_element_stack_buffer,
            },
        });
        task_old_element_buffer.set_buffers({
            .buffers = std::array{
                element_buffer,
                available_element_stack_buffer,
                released_element_stack_buffer,
            },
        });
    }
    ~AllocatorBufferState() {
        if (!element_buffer.is_empty()) {
            device.destroy_buffer(element_buffer);
        }
        if (!available_element_stack_buffer.is_empty()) {
            device.destroy_buffer(available_element_stack_buffer);
        }
        if (!released_element_stack_buffer.is_empty()) {
            device.destroy_buffer(released_element_stack_buffer);
        }
        device.destroy_buffer(allocator_buffer);
    }
    void init(daxa::Device &device, daxa::CommandRecorder &recorder) {
        auto staging_buffer = device.create_buffer({
            .size = sizeof(typename AllocatorConstants<T>::AllocatorType),
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "staging_buffer",
        });
        recorder.destroy_buffer_deferred(staging_buffer);
        auto *buffer_ptr = device.get_host_address_as<typename AllocatorConstants<T>::AllocatorType>(staging_buffer).value();
        *buffer_ptr = typename AllocatorConstants<T>::AllocatorType{
            .heap = device.get_device_address(element_buffer).value(),
            .available_element_stack = device.get_device_address(available_element_stack_buffer).value(),
            .released_element_stack = device.get_device_address(released_element_stack_buffer).value(),
            .element_count = 0,
            .available_element_stack_size = 0,
            .released_element_stack_size = 0,
        };
        recorder.copy_buffer_to_buffer({
            .src_buffer = staging_buffer,
            .dst_buffer = task_allocator_buffer.get_state().buffers[0],
            .size = sizeof(typename AllocatorConstants<T>::AllocatorType),
        });
    }
    void clear_buffers(daxa::CommandRecorder &recorder) {
        recorder.clear_buffer({
            .buffer = task_element_buffer.get_state().buffers[0],
            .offset = 0,
            .size = sizeof(typename AllocatorConstants<T>::ElementType) * AllocatorConstants<T>::ELEMENT_MULTIPLIER * current_element_count,
            .clear_value = 0,
        });
        recorder.clear_buffer({
            .buffer = task_element_buffer.get_state().buffers[1],
            .offset = 0,
            .size = sizeof(typename AllocatorConstants<T>::IndexType) * current_element_count,
            .clear_value = 0,
        });
        recorder.clear_buffer({
            .buffer = task_element_buffer.get_state().buffers[2],
            .offset = 0,
            .size = sizeof(typename AllocatorConstants<T>::IndexType) * current_element_count,
            .clear_value = 0,
        });
    }
    void realloc(daxa::Device &device, daxa::CommandRecorder &recorder) {
        recorder.copy_buffer_to_buffer({
            .src_buffer = task_old_element_buffer.get_state().buffers[0],
            .dst_buffer = task_element_buffer.get_state().buffers[0],
            .src_offset = 0,
            .dst_offset = 0,
            .size = sizeof(typename AllocatorConstants<T>::ElementType) * AllocatorConstants<T>::ELEMENT_MULTIPLIER * prev_element_count,
        });
        recorder.copy_buffer_to_buffer({
            .src_buffer = task_old_element_buffer.get_state().buffers[1],
            .dst_buffer = task_element_buffer.get_state().buffers[1],
            .src_offset = 0,
            .dst_offset = 0,
            .size = sizeof(typename AllocatorConstants<T>::IndexType) * prev_element_count,
        });
        recorder.copy_buffer_to_buffer({
            .src_buffer = task_old_element_buffer.get_state().buffers[2],
            .dst_buffer = task_element_buffer.get_state().buffers[2],
            .src_offset = 0,
            .dst_offset = 0,
            .size = sizeof(typename AllocatorConstants<T>::IndexType) * prev_element_count,
        });
        recorder.destroy_buffer_deferred(task_old_element_buffer.get_state().buffers[0]);
        recorder.destroy_buffer_deferred(task_old_element_buffer.get_state().buffers[1]);
        recorder.destroy_buffer_deferred(task_old_element_buffer.get_state().buffers[2]);
        task_old_element_buffer.set_buffers({});
        auto staging_buffer = device.create_buffer({
            .size = sizeof(typename AllocatorConstants<T>::AllocatorType),
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "staging_buffer",
        });
        recorder.destroy_buffer_deferred(staging_buffer);
        auto *buffer_ptr = device.get_host_address_as<typename AllocatorConstants<T>::AllocatorType>(staging_buffer).value();
        *buffer_ptr = typename AllocatorConstants<T>::AllocatorType{
            .heap = device.get_device_address(element_buffer).value(),
            .available_element_stack = device.get_device_address(available_element_stack_buffer).value(),
            .released_element_stack = device.get_device_address(released_element_stack_buffer).value(),
        };
        recorder.copy_buffer_to_buffer({
            .src_buffer = staging_buffer,
            .dst_buffer = allocator_buffer,
            .size = offsetof(typename AllocatorConstants<T>::AllocatorType, element_count),
        });
    }
    void for_each_task_buffer(auto const &functor) {
        functor(task_allocator_buffer);
        functor(task_element_buffer);
        functor(task_old_element_buffer);
    }
    void check_for_realloc(daxa::Device &device, size_t current_known_element_count) {
        constexpr auto MAX_ELEMENT_ALLOCATIONS_PER_FRAME = AllocatorConstants<T>::MAX_ELEMENT_ALLOCATIONS_PER_FRAME;
        auto const ELEM_SIZE_BYTES = static_cast<daxa_u32>(sizeof(typename AllocatorConstants<T>::ElementType) * AllocatorConstants<T>::ELEMENT_MULTIPLIER);
        auto const max_count_after_cpu_catch_up = static_cast<daxa_u32>(current_known_element_count + MAX_ELEMENT_ALLOCATIONS_PER_FRAME * (FRAMES_IN_FLIGHT + 1));
        auto const max_size_after_cpu_catch_up = static_cast<size_t>(max_count_after_cpu_catch_up) * ELEM_SIZE_BYTES;
        auto const current_size = static_cast<size_t>(current_element_count) * ELEM_SIZE_BYTES;
        next_element_count = 0;
        if (max_size_after_cpu_catch_up > current_size) {
            next_element_count = current_element_count + static_cast<daxa_u32>(MAX_ELEMENT_ALLOCATIONS_PER_FRAME * (FRAMES_IN_FLIGHT + 1));
            assert(next_element_count > current_element_count);
            prev_element_count = current_element_count;

            // Calculate new buffer size
            current_element_count = std::max(next_element_count * 3 / 2, max_count_after_cpu_catch_up);

            auto new_element_buffer = device.create_buffer({
                .size = ELEM_SIZE_BYTES * current_element_count,
                .name = AllocatorConstants<T>::element_buffer_name,
            });
            auto new_available_element_stack_buffer = device.create_buffer({
                .size = sizeof(typename AllocatorConstants<T>::IndexType) * current_element_count,
                .name = AllocatorConstants<T>::available_element_stack_buffer_name,
            });
            auto new_released_element_stack_buffer = device.create_buffer({
                .size = sizeof(typename AllocatorConstants<T>::IndexType) * current_element_count,
                .name = AllocatorConstants<T>::released_element_stack_buffer_name,
            });
            task_old_element_buffer.swap_buffers(task_element_buffer);
            element_buffer = new_element_buffer;
            available_element_stack_buffer = new_available_element_stack_buffer;
            released_element_stack_buffer = new_released_element_stack_buffer;
            task_element_buffer.set_buffers({
                .buffers = std::array{
                    element_buffer,
                    available_element_stack_buffer,
                    released_element_stack_buffer,
                },
            });
        }
    }
    auto needs_realloc() -> bool {
        return next_element_count != 0;
    }
};
#endif
