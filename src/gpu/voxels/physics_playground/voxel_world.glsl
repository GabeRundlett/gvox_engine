#pragma once

#include <shared/app.inl>

#include "trace.glsl"

#define DELTA_TIME GAME_PHYS_UPDATE_DT

RigidBody create_rigid_body(
    f32vec3 pos, f32vec3 lin_vel, f32vec3 rot, f32vec3 rot_vel,
    float density, float mass, float restitution, bool is_static, f32vec3 size, u32 shape_type) {
    RigidBody result;
    result.pos = pos;
    result.lin_vel = lin_vel;
    result.rot = rot;
    result.size = size;
    result.rot_vel = rot_vel;
    result.density = density;
    result.mass = mass;
    result.restitution = restitution;
    result.flags = shape_type | u32(is_static);
    return result;
}

RigidBody create_rigid_body_sphere(float radius, f32vec3 pos, float density, float restitution, bool is_static) {
    f32 volume = 4.0 / 3.0 * PI * radius * radius * radius;
    return create_rigid_body(
        pos, f32vec3(0.0, 0.0, 0.0), f32vec3(0.0, 0.0, 0.0), f32vec3(0.0, 0.0, 0.0),
        density, density * volume, restitution, is_static, f32vec3(radius, 0.0, 0.0), RIGID_BODY_SHAPE_TYPE_SPHERE);
}
RigidBody create_rigid_body_box(f32vec3 size, f32vec3 pos, float density, float restitution, bool is_static) {
    f32 volume = size.x * size.y * size.z;
    return create_rigid_body(
        pos, f32vec3(0.0, 0.0, 0.0), f32vec3(0.0, 0.0, 0.0), f32vec3(0.0, 0.0, 0.0),
        density, density * volume, restitution, is_static, size, RIGID_BODY_SHAPE_TYPE_BOX);
}

void voxel_world_startup(daxa_RWBufferPtr(GpuGlobals) globals_ptr, VoxelRWBufferPtrs ptrs) {
    u32 body_n = 0;

    deref(ptrs.globals).rigid_bodies[body_n++] = create_rigid_body_sphere(0.5, f32vec3(1.0, 1.0, 0.0), 1.0, 0.5, false);
    deref(ptrs.globals).rigid_bodies[body_n++] = create_rigid_body_sphere(0.5, f32vec3(2.0, 3.0, 0.0), 1.0, 0.5, false);
    deref(ptrs.globals).rigid_bodies[body_n++] = create_rigid_body_sphere(0.5, f32vec3(4.0, 1.0, 0.0), 1.0, 0.5, false);
    deref(ptrs.globals).rigid_bodies[body_n++] = create_rigid_body_box(f32vec3(1.0), f32vec3(3.0, 2.0, 0.0), 1.0, 0.5, false);
    deref(ptrs.globals).rigid_bodies[body_n++] = create_rigid_body_box(f32vec3(1.3), f32vec3(6.0, 3.0, 0.0), 1.0, 0.5, false);

    deref(ptrs.globals).rigid_body_n = body_n;
}

void voxel_world_perframe(daxa_BufferPtr(GpuInput) gpu_input, daxa_RWBufferPtr(GpuOutput) gpu_output, daxa_RWBufferPtr(GpuGlobals) globals_ptr, VoxelRWBufferPtrs ptrs) {
    deref(ptrs.globals).prev_offset = deref(ptrs.globals).offset;
    deref(ptrs.globals).offset = deref(globals_ptr).player.player_unit_offset;

    bool is_phys_tick = (deref(gpu_input).flags & GAME_FLAG_BITS_NEEDS_PHYS_UPDATE) != 0;
    if (!is_phys_tick) {
        return;
    }

    // f32vec2 frame_dim = deref(gpu_input).frame_dim;
    // f32vec2 inv_frame_dim = f32vec2(1.0) / frame_dim;
    // f32vec4 output_tex_size = f32vec4(frame_dim, inv_frame_dim);
    // f32vec2 uv = get_uv(deref(gpu_input).mouse.pos, output_tex_size);
    // ViewRayContext vrc = vrc_from_uv(globals, uv);
    // f32vec3 ray_dir = ray_dir_ws(vrc);
    // f32vec3 offset = f32vec3(deref(ptrs.globals).offset);
    // f32vec3 ray_pos = ray_origin_ws(vrc) + offset;

    deref(ptrs.globals).rigid_bodies[3].rot.y += DELTA_TIME;
}
