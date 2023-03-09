#pragma once

#include <shared/shared.inl>

u32 gpu_malloc(GpuAllocator allocator, u32 size) {
    u32 result_address = atomicAdd(deref(allocator.state).offset, size);
    return result_address;
}

void gpu_free(GpuAllocator allocator, u32 address, u32 size) {
    // Doesn't matter for now...
}
