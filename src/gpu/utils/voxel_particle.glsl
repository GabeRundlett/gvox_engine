#pragma once

#include <shared/shared.inl>
#include <utils/math.glsl>
#include <utils/trace.glsl>

void particle_spawn(in out SimulatedVoxelParticle self, u32 index) {
    rand_seed(index);

    self.duration_alive = 0.0 + rand() * 0;
    self.flags = 1;

    self.pos = f32vec3(rand() * 600 + 300, rand() * 400 + 500, 400.0) / 8;
    self.vel = deref(globals).player.forward * 3 + rand_dir() * 2;
    // self.pos = deref(globals).player.cam.pos + deref(globals).player.forward * 1 + vec3(0, 0, -2.5) + deref(globals).player.lateral * 5.5;
    // self.vel = deref(globals).player.forward * 3 + rand_dir() * 2 + deref(globals).player.vel;
}

void particle_update(in out SimulatedVoxelParticle self, daxa_BufferPtr(GpuInput) gpu_input) {
    rand_seed(gl_GlobalInvocationID.x + uint((deref(gpu_input).time) * 100));

    float dt = deref(gpu_input).delta_time;
    self.duration_alive += dt;

#if 0
    self.vel += f32vec3(0.0, 0.0, -9.8) * dt;
    self.pos += self.vel * dt;
#else
    if (self.flags == 2) {
        self.vel += (f32vec3(0.0, 0.0, 1.0) + rand_dir() * 5.1) * dt;
    } else {
        self.vel += f32vec3(0.0, 0.0, -9.8) * dt;
    }

    u32vec3 chunk_n = u32vec3(1u << deref(settings).log2_chunks_per_axis);
    float curr_speed = length(self.vel);
    float curr_dist_in_dt = curr_speed * dt;
    vec3 ray_pos = self.pos;
    float dist = trace_hierarchy_traversal(voxel_malloc_global_allocator, voxel_chunks, chunk_n, ray_pos, self.vel / curr_speed, 512, curr_dist_in_dt, true);
    if (!(dist > curr_dist_in_dt)) {
        self.pos += self.vel / curr_speed * (dist - 0.001);
        vec3 nrm = scene_nrm(voxel_malloc_global_allocator, voxel_chunks, chunk_n, ray_pos);
        const float bounciness = 0.05;
        const float dampening = 0.3;
        self.vel = normalize(reflect(self.vel, nrm) * (0.5 + 0.5 * bounciness) + self.vel * (0.5 - 0.5 * bounciness)) * curr_speed * dampening;
        if (good_rand(gl_GlobalInvocationID.x) < 0.1) {
            self.vel *= 0.2;
            self.flags = 2;
        }
    } else {
        self.pos += self.vel * dt;
    }
#endif
    if (self.duration_alive > 15.0) {
        particle_spawn(self, gl_GlobalInvocationID.x);
    }
}
