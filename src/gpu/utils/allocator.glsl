#pragma once

#include <shared/shared.inl>

u32 gpu_malloc(GpuAllocator allocator, u32 size) {
    u32 result_address = atomicAdd(deref(allocator.state).offset, size + 1);
    deref(allocator.heap[result_address]) = size + 1;
    return result_address + 1;
}

void gpu_free(GpuAllocator allocator, u32 address) {
    // Doesn't matter for now...

    // if (address != 0) {
    //     i32 size = i32(deref(allocator.heap[address - 1]));
    //     atomicAdd(deref(allocator.state).offset, -size);
    // }
}
