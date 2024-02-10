#pragma once

#include <shared/voxels/impl/voxels.inl>

#if !defined(VOXELS_BUFFER_PTRS)
#error "The implementation must define a way for the users to construct the read-only pointers!"
#endif

#if !defined(VOXELS_RW_BUFFER_PTRS)
#error "The implementation must define a way for the users to construct the read-write pointers!"
#endif
