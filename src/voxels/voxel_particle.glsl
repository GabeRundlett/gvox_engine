#pragma once

#include <voxels/voxel_particles.inl>
#include <utilities/gpu/math.glsl>
#include <voxels/voxels.glsl>

#define PARTICLE_ALIVE_FLAG (1 << 0)
#define PARTICLE_SMOKE_FLAG (1 << 1)

#define PARTICLE_SLEEP_TIMER_OFFSET (8)
#define PARTICLE_SLEEP_TIMER_MASK (0xff << PARTICLE_SLEEP_TIMER_OFFSET)

vec3 get_particle_worldspace_origin(daxa_RWBufferPtr(GpuGlobals) globals, vec3 pos) {
    // return pos - deref(gpu_input).player.player_unit_offset;
    return floor(pos * VOXEL_SCL) / VOXEL_SCL - deref(gpu_input).player.player_unit_offset;
}

vec3 get_particle_worldspace_pos(daxa_RWBufferPtr(GpuGlobals) globals, vec3 pos) {
    return pos - deref(gpu_input).player.player_unit_offset;
}

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_COMPUTE

void particle_spawn(in out SimulatedVoxelParticle self, uint index) {
    rand_seed(index);

    self.duration_alive = 0.0 + rand() * 1.0;
    self.flags = PARTICLE_ALIVE_FLAG;

    // self.pos = vec3(good_rand(deref(gpu_input).time + 137.41) * 10 + 20, good_rand(deref(gpu_input).time + 41.137) * 10 + 20, 70.0);
    // self.vel = deref(gpu_input).player.forward * 0 + vec3(0, 0, -10);
    self.pos = deref(gpu_input).player.pos + deref(gpu_input).player.player_unit_offset + deref(gpu_input).player.forward * 1 + vec3(0, 0, -2.5) + deref(gpu_input).player.lateral * 1.5;
    self.vel = deref(gpu_input).player.forward * 8 + rand_dir() * 2; // + deref(gpu_input).player.vel;

    vec3 col = vec3(0.3, 0.4, 0.9) * (rand() * 0.5 + 0.5); // vec3(0.5);
    vec3 nrm = vec3(0, 0, 1);
    Voxel particle_voxel = Voxel(1, 0.99, nrm, col);
    self.packed_voxel = pack_voxel(particle_voxel);
}

void particle_update(in out SimulatedVoxelParticle self, VoxelBufferPtrs voxels_buffer_ptrs, daxa_BufferPtr(GpuInput) gpu_input, in out bool should_place) {
    rand_seed(gl_GlobalInvocationID.x + uint((deref(gpu_input).time) * 13741));

    // if (PER_VOXEL_NORMALS != 0) {
    //     Voxel particle_voxel = unpack_voxel(self.packed_voxel);
    //     particle_voxel.normal = normalize(deref(gpu_input).player.pos - get_particle_worldspace_pos(globals, self.pos));
    //     self.packed_voxel = pack_voxel(particle_voxel);
    // }

    const bool PARTICLES_PAUSED = false;

    if (!PARTICLES_PAUSED) {
        if ((self.flags & PARTICLE_ALIVE_FLAG) == 0) {
            particle_spawn(self, gl_GlobalInvocationID.x);
            return;
        }

        float dt = min(deref(gpu_input).delta_time, 0.01);
        self.duration_alive += dt;

        if ((self.flags & PARTICLE_SMOKE_FLAG) != 0) {
            self.vel += (vec3(0.0, 0.0, 1.0) + rand_dir() * 5.1) * dt;
        } else {
            self.vel += vec3(0.0, 0.0, -9.8) * dt;
        }

        float curr_speed = length(self.vel);
        float curr_dist_in_dt = curr_speed * dt;
        vec3 ray_pos = get_particle_worldspace_pos(globals, self.pos);

        VoxelTraceResult trace_result = voxel_trace(VoxelTraceInfo(voxels_buffer_ptrs, self.vel / curr_speed, MAX_STEPS, curr_dist_in_dt, 0.0, true), ray_pos);
        float dist = trace_result.dist;

        if (dist < curr_dist_in_dt) {
            self.pos += self.vel / curr_speed * (dist - 0.001);
            vec3 nrm = trace_result.nrm;
            if (abs(dot(nrm, nrm) - 1.0) > 0.1) {
                nrm = vec3(0, 0, 1);
            }
            nrm = normalize(nrm);

            const float bounciness = 0.5;
            const float dampening = 0.01;
            // self.vel = normalize(reflect(self.vel, nrm) * (0.5 + 0.5 * bounciness) + self.vel * (0.5 - 0.5 * bounciness)) * curr_speed * dampening;
            self.vel = reflect(self.vel, nrm) * bounciness;
            // if (good_rand(gl_GlobalInvocationID.x) < 0.1) {
            //     self.vel *= 0.2;
            //     self.flags |= PARTICLE_SMOKE_FLAG;
            // }
        } else {
            self.pos += self.vel * dt;
        }

        if (min(dist, curr_dist_in_dt) < 0.01 / VOXEL_SCL) {
            self.flags += 1 << PARTICLE_SLEEP_TIMER_OFFSET;
            if (((self.flags & PARTICLE_SLEEP_TIMER_MASK) >> PARTICLE_SLEEP_TIMER_OFFSET) >= 15) {
                should_place = true;
                self.flags &= ~PARTICLE_ALIVE_FLAG & ~PARTICLE_SLEEP_TIMER_MASK;
            }
        } else {
            self.flags &= ~PARTICLE_SLEEP_TIMER_MASK;
        }

        if (self.duration_alive > 10.0) {
            particle_spawn(self, gl_GlobalInvocationID.x);
        }
    } else {
        self.vel = vec3(0);
    }
}

#endif
