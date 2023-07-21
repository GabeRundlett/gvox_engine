#if !defined(UserAllocatorType) || !defined(UserIndexType)
#error "You must define all of the above types to include this file!"
#endif

#define FUNC_NAME_HELPER_HELPER(Prefix, Name) Prefix##_##Name
#define FUNC_NAME_HELPER(Prefix, Name) FUNC_NAME_HELPER_HELPER(Prefix, Name)
#define FUNC_NAME(Name) FUNC_NAME_HELPER(UserAllocatorType, Name)

UserIndexType FUNC_NAME(malloc)(daxa_RWBufferPtr(UserAllocatorType) allocator) {
    const i32 index_in_avail_stack = atomicAdd(deref(allocator).available_element_stack_size, -1) - 1;
    // If we get an index of smaller then zero, the size was 0. This means the stack was empty.
    // In that case we need to create new pages as the available stack is empty.
    if (index_in_avail_stack < 0) {
        // Create new page.
        return UserIndexType(atomicAdd(deref(allocator).element_count, 1));
    } else {
        return UserIndexType(deref(deref(allocator).available_element_stack[index_in_avail_stack]));
    }
}

void FUNC_NAME(free)(daxa_RWBufferPtr(UserAllocatorType) allocator, UserIndexType element_ptr) {
    const u32 index_in_free_stack = atomicAdd(deref(allocator).released_element_stack_size, 1);
    deref(deref(allocator).released_element_stack[index_in_free_stack]) = element_ptr;
}

void FUNC_NAME(perframe)(daxa_RWBufferPtr(UserAllocatorType) allocator) {
    // Clear the available element stack counter if it was emptied below zero.
    deref(allocator).available_element_stack_size = max(deref(allocator).available_element_stack_size, 0);
    // Move all released elements from the released stack to the available element stack.
    while (deref(allocator).released_element_stack_size > 0) {
        --deref(allocator).released_element_stack_size;
        deref(deref(allocator).available_element_stack[deref(allocator).available_element_stack_size]) = deref(deref(allocator).released_element_stack[deref(allocator).released_element_stack_size]);
        ++deref(allocator).available_element_stack_size;
    }
}

u32 FUNC_NAME(get_consumed_element_count)(daxa_RWBufferPtr(UserAllocatorType) allocator) {
    return deref(allocator).element_count - u32(max(deref(allocator).available_element_stack_size, 0));
}

#undef FUNC_NAME_HELPER_HELPER
#undef FUNC_NAME_HELPER
#undef FUNC_NAME

#undef UserAllocatorType
#undef UserIndexType
