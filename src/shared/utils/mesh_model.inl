#pragma once

#include <shared/core.inl>

struct MeshVertex {
    daxa_f32vec3 pos;
    daxa_f32vec2 tex;
    daxa_u32 rotation;
};
DAXA_DECL_BUFFER_PTR(MeshVertex)

struct MeshGpuInput {
    daxa_u32vec3 size;
    daxa_f32vec3 bound_min;
    daxa_f32vec3 bound_max;
};
DAXA_DECL_BUFFER_PTR(MeshGpuInput)

struct MeshRasterPush {
    daxa_BufferPtr(MeshGpuInput) gpu_input;
    daxa_BufferPtr(MeshVertex) vertex_buffer;
    daxa_BufferPtr(daxa_f32vec3) normal_buffer;
    daxa_RWBufferPtr(daxa_u32) voxel_buffer;
    daxa_ImageViewIndex texture_id;
    daxa_SamplerId texture_sampler;
};

struct MeshPreprocessPush {
    daxa_f32mat4x4 modl_mat;
    daxa_BufferPtr(MeshGpuInput) gpu_input;
    daxa_BufferPtr(MeshVertex) vertex_buffer;
    daxa_BufferPtr(daxa_f32vec3) normal_buffer;
    daxa_u32 triangle_count;
};
