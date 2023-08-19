#pragma once

#include <shared/app.inl>

#include "trace.glsl"

void box_init(in out PhysBox phys_box, vec3 pos) {
    phys_box.pos = pos;
    phys_box.vel = f32vec3(0.0);
    phys_box.size = f32vec3(0.25);
    phys_box.grab_dist = -1.0;
    phys_box.rot = f32vec3(0.0);
}

void box_update(daxa_BufferPtr(GpuInput) gpu_input, in out PhysBox phys_box, in vec3 ray_pos, in vec3 ray_dir) {
    phys_box.prev_pos = phys_box.pos;
    phys_box.prev_rot = phys_box.rot;

    vec3 desired_pos = ray_pos + ray_dir * phys_box.grab_dist;
    if (phys_box.grab_dist != -1.0) {
        vec3 del = (desired_pos - phys_box.pos);
        const float mass = 150.0;
        phys_box.vel += del / max(dot(del, del), 1.0) * mass * deref(gpu_input).delta_time;
        vec3 friction_vec = phys_box.vel;
        apply_friction(gpu_input, phys_box.vel, friction_vec, 15.0);
    } else {
        phys_box.vel += vec3(0, 0, -9.8) * deref(gpu_input).delta_time;
    }

    phys_box.pos += phys_box.vel * deref(gpu_input).delta_time;
    phys_box.rot += f32vec3(0, 0, 1.0) * deref(gpu_input).delta_time;

    if (phys_box.pos.z < -1.0 + phys_box.size.z) {
        phys_box.pos.z = -1.0 + phys_box.size.z;
        phys_box.vel.z *= -0.5;

        vec3 friction_vec = phys_box.vel * vec3(1, 1, 0);
        apply_friction(gpu_input, phys_box.vel, friction_vec, 9.8);
    }
    if (deref(gpu_input).actions[GAME_ACTION_BRUSH_A] != 0) {
        if (phys_box.grab_dist == -1.0) {
            if (hit_box(ray_pos, ray_dir, phys_box.pos, phys_box.size)) {
                phys_box.grab_dist = length(phys_box.pos - ray_pos);
            }
        }
    } else {
        phys_box.grab_dist = -1.0;
    }
}

void voxel_world_startup(daxa_RWBufferPtr(GpuGlobals) globals_ptr, VoxelRWBufferPtrs ptrs) {
    box_init(deref(ptrs.globals).box0, f32vec3(2.0, 1.0, 0.0));
    box_init(deref(ptrs.globals).box1, f32vec3(4.0, 1.0, 0.0));
}

void voxel_world_perframe(daxa_BufferPtr(GpuInput) gpu_input, daxa_RWBufferPtr(GpuOutput) gpu_output, daxa_RWBufferPtr(GpuGlobals) globals_ptr, VoxelRWBufferPtrs ptrs) {
    deref(ptrs.globals).prev_offset = deref(ptrs.globals).offset;
    deref(ptrs.globals).offset = deref(globals_ptr).player.chunk_offset;

    f32vec2 frame_dim = deref(gpu_input).frame_dim;
    f32vec2 inv_frame_dim = f32vec2(1.0) / frame_dim;
    f32vec4 output_tex_size = f32vec4(frame_dim, inv_frame_dim);
    f32vec2 uv = get_uv(deref(gpu_input).mouse.pos, output_tex_size);
    ViewRayContext vrc = vrc_from_uv(globals, uv);
    f32vec3 ray_dir = ray_dir_ws(vrc);
    f32vec3 offset = f32vec3(deref(ptrs.globals).offset);
    f32vec3 ray_pos = ray_origin_ws(vrc) + offset;

    box_update(gpu_input, deref(ptrs.globals).box0, ray_pos, ray_dir);
    box_update(gpu_input, deref(ptrs.globals).box1, ray_pos, ray_dir);
}
