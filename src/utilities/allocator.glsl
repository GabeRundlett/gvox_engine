#if !defined(UserAllocatorType) || !defined(UserIndexType)
#error "You must define all of the above types to include this file!"
#endif

#define FUNC_NAME_HELPER_HELPER(Prefix, Name) Prefix##_##Name
#define FUNC_NAME_HELPER(Prefix, Name) FUNC_NAME_HELPER_HELPER(Prefix, Name)
#define FUNC_NAME(Name) FUNC_NAME_HELPER(UserAllocatorType, Name)

UserIndexType FUNC_NAME(malloc)(daxa_RWBufferPtr(UserAllocatorType) allocator) {
    const int index_in_avail_stack = atomicAdd(deref(allocator).available_element_stack_size, -1) - 1;
    // If we get an index of smaller then zero, the size was 0. This means the stack was empty.
    // In that case we need to create new pages as the available stack is empty.
    if (index_in_avail_stack < 0) {
        // Create new page.
        return UserIndexType(atomicAdd(deref(allocator).element_count, 1));
    } else {
        return UserIndexType(deref(advance(deref(allocator).available_element_stack, index_in_avail_stack)));
    }
}

void FUNC_NAME(free)(daxa_RWBufferPtr(UserAllocatorType) allocator, UserIndexType element_ptr) {
    // Push a new element onto the released stack. This will be moved to the available stack for the next frame.
    const uint index_in_free_stack = atomicAdd(deref(allocator).released_element_stack_size, 1);
    deref(advance(deref(allocator).released_element_stack, index_in_free_stack)) = element_ptr;
}

void FUNC_NAME(perframe)(daxa_RWBufferPtr(UserAllocatorType) allocator) {
    // Clear the available element stack counter if it was emptied below zero.
    deref(allocator).available_element_stack_size = max(deref(allocator).available_element_stack_size, 0);
    // Move all released elements from the released stack to the available element stack.
    while (deref(allocator).released_element_stack_size > 0) {
        --deref(allocator).released_element_stack_size;
        deref(advance(deref(allocator).available_element_stack, deref(allocator).available_element_stack_size)) =
            deref(advance(deref(allocator).released_element_stack, deref(allocator).released_element_stack_size));
        ++deref(allocator).available_element_stack_size;
    }
}

uint FUNC_NAME(get_consumed_element_count)(daxa_BufferPtr(UserAllocatorType) allocator) {
    return deref(allocator).element_count - uint(max(deref(allocator).available_element_stack_size, 0));
}

#undef FUNC_NAME_HELPER_HELPER
#undef FUNC_NAME_HELPER
#undef FUNC_NAME

#undef UserAllocatorType
#undef UserIndexType
