#if !defined(UserAllocatorType) || !defined(UserIndexType)
#error "You must define all of the above types to include this file!"
#endif

#define FUNC_NAME_HELPER_HELPER(Prefix, Name) Prefix##_##Name
#define FUNC_NAME_HELPER(Prefix, Name) FUNC_NAME_HELPER_HELPER(Prefix, Name)
#define FUNC_NAME(Name) FUNC_NAME_HELPER(UserAllocatorType, Name)

UserIndexType FUNC_NAME(malloc)(daxa_RWBufferPtr(UserAllocatorType) allocator) {
    UserIndexType result = UserIndexType(atomicAdd(deref(allocator).element_count, 1));
#if defined(UserMaxElementCount)
    if (result >= UserMaxElementCount) {
        UserIndexType(atomicAdd(deref(allocator).element_count, -1));
    }
#endif
    atomicMax(deref(allocator).element_count_dispatch.x, (result + 1 + 127) / 128);
    return result;
}

uint FUNC_NAME(get_consumed_element_count)(daxa_BufferPtr(UserAllocatorType) allocator) {
    return deref(allocator).element_count;
}

#undef FUNC_NAME_HELPER_HELPER
#undef FUNC_NAME_HELPER
#undef FUNC_NAME

#undef UserAllocatorType
#undef UserIndexType
#undef UserMaxElementCount
