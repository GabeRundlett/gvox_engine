#pragma once

#include <shared/core.inl>

// The "simple" allocator declared here (as well as implemented both here and further
// for the GLSL side in allocator.glsl) is just a simple free-list linear allocator.

#define DECL_SIMPLE_ALLOCATOR(AllocatorType_, ElementType_, ElementMultiplier_, IndexType_) \
    struct AllocatorType_ {                                                                 \
        daxa_RWBufferPtr(ElementType_) heap;                                                \
        daxa_RWBufferPtr(IndexType_) available_element_stack;                               \
        daxa_RWBufferPtr(IndexType_) released_element_stack;                                \
        i32 element_count;                                                                  \
        i32 available_element_stack_size;                                                   \
        i32 released_element_stack_size;                                                    \
    };                                                                                      \
    DAXA_DECL_BUFFER_PTR(AllocatorType_)                                                    \
    CPU_ONLY(DECL_SIMPLE_ALLOCATOR_CONSTANTS(AllocatorType_, ElementType_, ElementMultiplier_, IndexType_))

#define DECL_SIMPLE_ALLOCATOR_CONSTANTS(AllocatorType_, ElementType_, ElementMultiplier_, IndexType_)                               \
    template <>                                                                                                                     \
    struct AllocatorConstants<AllocatorType_> {                                                                                     \
        using AllocatorType = AllocatorType_;                                                                                       \
        using ElementType = ElementType_;                                                                                           \
        using IndexType = IndexType_;                                                                                               \
        static constexpr usize ELEMENT_MULTIPLIER = ElementMultiplier_;                                                             \
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
    using ElementType = u32;
    using IndexType = u32;
    static constexpr usize ELEMENT_MULTIPLIER = 1;
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
    daxa::BufferId allocator_buffer;
    daxa::BufferId element_buffer;
    daxa::BufferId available_element_stack_buffer;
    daxa::BufferId released_element_stack_buffer;
    daxa::TaskBuffer task_allocator_buffer{{.name = AllocatorConstants<T>::task_allocator_buffer_name}};
    daxa::TaskBuffer task_element_buffer{{.name = AllocatorConstants<T>::task_element_buffer_name}};
    daxa::TaskBuffer task_old_element_buffer{{.name = AllocatorConstants<T>::task_old_element_buffer_name}};
    u32 current_element_count = 0;
    u32 next_element_count = 0;
    u32 prev_element_count = 0;
    void create(daxa::Device &device, u32 element_count) {
        current_element_count = element_count;
        allocator_buffer = device.create_buffer({
            .size = sizeof(AllocatorConstants<T>::AllocatorType),
            .name = AllocatorConstants<T>::allocator_buffer_name,
        });
        element_buffer = device.create_buffer({
            .size = static_cast<u32>(sizeof(AllocatorConstants<T>::ElementType)) * AllocatorConstants<T>::ELEMENT_MULTIPLIER * current_element_count,
            .name = AllocatorConstants<T>::element_buffer_name,
        });
        available_element_stack_buffer = device.create_buffer({
            .size = static_cast<u32>(sizeof(AllocatorConstants<T>::IndexType)) * current_element_count,
            .name = AllocatorConstants<T>::available_element_stack_buffer_name,
        });
        released_element_stack_buffer = device.create_buffer({
            .size = static_cast<u32>(sizeof(AllocatorConstants<T>::IndexType)) * current_element_count,
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
    void destroy(daxa::Device &device) const {
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
    void init(daxa::Device &device, daxa::CommandList &cmd_list) {
        auto staging_buffer = device.create_buffer({
            .size = sizeof(AllocatorConstants<T>::AllocatorType),
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "staging_buffer",
        });
        cmd_list.destroy_buffer_deferred(staging_buffer);
        auto *buffer_ptr = device.get_host_address_as<AllocatorConstants<T>::AllocatorType>(staging_buffer);
        *buffer_ptr = typename AllocatorConstants<T>::AllocatorType{
            .heap = device.get_device_address(element_buffer),
            .available_element_stack = device.get_device_address(available_element_stack_buffer),
            .released_element_stack = device.get_device_address(released_element_stack_buffer),
            .element_count = 0,
            .available_element_stack_size = 0,
            .released_element_stack_size = 0,
        };
        cmd_list.copy_buffer_to_buffer({
            .src_buffer = staging_buffer,
            .dst_buffer = task_allocator_buffer.get_state().buffers[0],
            .size = sizeof(AllocatorConstants<T>::AllocatorType),
        });
    }
    void clear_buffers(daxa::CommandList &cmd_list) {
        cmd_list.clear_buffer({
            .buffer = task_element_buffer.get_state().buffers[0],
            .offset = 0,
            .size = static_cast<u32>(sizeof(AllocatorConstants<T>::ElementType)) * AllocatorConstants<T>::ELEMENT_MULTIPLIER * current_element_count,
            .clear_value = 0,
        });
        cmd_list.clear_buffer({
            .buffer = task_element_buffer.get_state().buffers[1],
            .offset = 0,
            .size = sizeof(AllocatorConstants<T>::IndexType) * current_element_count,
            .clear_value = 0,
        });
        cmd_list.clear_buffer({
            .buffer = task_element_buffer.get_state().buffers[2],
            .offset = 0,
            .size = sizeof(AllocatorConstants<T>::IndexType) * current_element_count,
            .clear_value = 0,
        });
    }
    void realloc(daxa::Device &device, daxa::CommandList &cmd_list) {
        cmd_list.copy_buffer_to_buffer({
            .src_buffer = task_old_element_buffer.get_state().buffers[0],
            .src_offset = 0,
            .dst_buffer = task_element_buffer.get_state().buffers[0],
            .dst_offset = 0,
            .size = static_cast<u32>(sizeof(AllocatorConstants<T>::ElementType)) * AllocatorConstants<T>::ELEMENT_MULTIPLIER * prev_element_count,
        });
        cmd_list.copy_buffer_to_buffer({
            .src_buffer = task_old_element_buffer.get_state().buffers[1],
            .src_offset = 0,
            .dst_buffer = task_element_buffer.get_state().buffers[1],
            .dst_offset = 0,
            .size = sizeof(AllocatorConstants<T>::IndexType) * prev_element_count,
        });
        cmd_list.copy_buffer_to_buffer({
            .src_buffer = task_old_element_buffer.get_state().buffers[2],
            .src_offset = 0,
            .dst_buffer = task_element_buffer.get_state().buffers[2],
            .dst_offset = 0,
            .size = sizeof(AllocatorConstants<T>::IndexType) * prev_element_count,
        });
        cmd_list.destroy_buffer_deferred(task_old_element_buffer.get_state().buffers[0]);
        cmd_list.destroy_buffer_deferred(task_old_element_buffer.get_state().buffers[1]);
        cmd_list.destroy_buffer_deferred(task_old_element_buffer.get_state().buffers[2]);
        task_old_element_buffer.set_buffers({});
        auto staging_buffer = device.create_buffer({
            .size = sizeof(AllocatorConstants<T>::AllocatorType),
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "staging_buffer",
        });
        cmd_list.destroy_buffer_deferred(staging_buffer);
        auto *buffer_ptr = device.get_host_address_as<AllocatorConstants<T>::AllocatorType>(staging_buffer);
        *buffer_ptr = typename AllocatorConstants<T>::AllocatorType{
            .heap = device.get_device_address(element_buffer),
            .available_element_stack = device.get_device_address(available_element_stack_buffer),
            .released_element_stack = device.get_device_address(released_element_stack_buffer),
        };
        cmd_list.copy_buffer_to_buffer({
            .src_buffer = staging_buffer,
            .dst_buffer = allocator_buffer,
            .size = offsetof(AllocatorConstants<T>::AllocatorType, element_count),
        });
    }
    void for_each_buffer(auto const &functor) {
        functor(allocator_buffer);
        functor(element_buffer);
        functor(available_element_stack_buffer);
        functor(released_element_stack_buffer);
    }
    void for_each_task_buffer(auto const &functor) {
        functor(task_allocator_buffer);
        functor(task_element_buffer);
        functor(task_old_element_buffer);
    }
    void check_for_realloc(daxa::Device &device, usize current_known_size, usize MAX_ELEMENT_ALLOCATIONS_PER_FRAME) {
        auto const ELEM_SIZE_BYTES = static_cast<u32>(sizeof(AllocatorConstants<T>::ElementType)) * AllocatorConstants<T>::ELEMENT_MULTIPLIER;
        auto const max_size_after_cpu_catch_up =
            current_known_size +
            MAX_ELEMENT_ALLOCATIONS_PER_FRAME * (FRAMES_IN_FLIGHT + 1) * ELEM_SIZE_BYTES;
        auto const current_size = static_cast<size_t>(current_element_count) * ELEM_SIZE_BYTES;
        next_element_count = 0;
        if (max_size_after_cpu_catch_up > current_size) {
            next_element_count =
                current_element_count + static_cast<u32>(MAX_ELEMENT_ALLOCATIONS_PER_FRAME * (FRAMES_IN_FLIGHT + 1));
            assert(next_element_count > current_element_count);
            prev_element_count = current_element_count;
            current_element_count = next_element_count * 2;
            auto new_element_buffer = device.create_buffer({
                .size = ELEM_SIZE_BYTES * current_element_count,
                .name = AllocatorConstants<T>::element_buffer_name,
            });
            auto new_available_element_stack_buffer = device.create_buffer({
                .size = static_cast<u32>(sizeof(AllocatorConstants<T>::IndexType)) * current_element_count,
                .name = AllocatorConstants<T>::available_element_stack_buffer_name,
            });
            auto new_released_element_stack_buffer = device.create_buffer({
                .size = static_cast<u32>(sizeof(AllocatorConstants<T>::IndexType)) * current_element_count,
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
