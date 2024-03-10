#pragma once

#include <voxels/particles/voxel_particles.inl>
#include <utilities/gpu/math.glsl>
#include <voxels/voxels.glsl>

#define PARTICLE_ALIVE_FLAG (1 << 0)
#define PARTICLE_SMOKE_FLAG (1 << 1)

#define PARTICLE_SLEEP_TIMER_OFFSET (8)
#define PARTICLE_SLEEP_TIMER_MASK (0xff << PARTICLE_SLEEP_TIMER_OFFSET)

vec3 get_particle_worldspace_origin(daxa_BufferPtr(GpuInput) gpu_input, vec3 pos) {
    // return pos - deref(gpu_input).player.player_unit_offset + 0.5 * VOXEL_SIZE;
    return floor(pos * VOXEL_SCL) * VOXEL_SIZE - deref(gpu_input).player.player_unit_offset + 0.5 * VOXEL_SIZE;
}

vec3 get_particle_worldspace_pos(daxa_BufferPtr(GpuInput) gpu_input, vec3 pos) {
    return pos - deref(gpu_input).player.player_unit_offset;
}

void particle_spawn(in out SimulatedVoxelParticle self, uint index, daxa_BufferPtr(GpuInput) gpu_input) {
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

void particle_update(in out SimulatedVoxelParticle self, uint self_index, VoxelBufferPtrs voxels_buffer_ptrs, daxa_BufferPtr(GpuInput) gpu_input, in out bool should_place) {
    rand_seed(self_index + uint((deref(gpu_input).time) * 13741));

    if (deref(gpu_input).frame_index == 0) {
        particle_spawn(self, self_index, gpu_input);
    }

    // if (PER_VOXEL_NORMALS != 0) {
    //     Voxel particle_voxel = unpack_voxel(self.packed_voxel);
    //     particle_voxel.normal = normalize(deref(gpu_input).player.pos - get_particle_worldspace_pos(gpu_input, self.pos));
    //     self.packed_voxel = pack_voxel(particle_voxel);
    // }

    const bool PARTICLES_PAUSED = false;

    if (!PARTICLES_PAUSED) {
        if ((self.flags & PARTICLE_ALIVE_FLAG) == 0) {
            particle_spawn(self, self_index, gpu_input);
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
        vec3 ray_pos = get_particle_worldspace_pos(gpu_input, self.pos);

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
            const float dampening = 0.001;
            // self.vel = normalize(reflect(self.vel, nrm) * (0.5 + 0.5 * bounciness) + self.vel * (0.5 - 0.5 * bounciness)) * curr_speed * dampening;
            self.vel = reflect(self.vel, nrm) * bounciness;
            // if (good_rand(self_index) < 0.1) {
            //     self.vel *= 0.2;
            //     self.flags |= PARTICLE_SMOKE_FLAG;
            // }
        } else {
            self.pos += self.vel * dt;
        }

        if (min(dist, curr_dist_in_dt) < 0.01 * VOXEL_SIZE) {
            self.flags += 1 << PARTICLE_SLEEP_TIMER_OFFSET;
            if (((self.flags & PARTICLE_SLEEP_TIMER_MASK) >> PARTICLE_SLEEP_TIMER_OFFSET) >= 200) {
                should_place = true;
                self.flags &= ~PARTICLE_ALIVE_FLAG & ~PARTICLE_SLEEP_TIMER_MASK;
            }
        } else {
            self.flags &= ~PARTICLE_SLEEP_TIMER_MASK;
        }

        if (self.duration_alive > 10.0) {
            particle_spawn(self, self_index, gpu_input);
        }
    } else {
        self.vel = vec3(0);
        // vec3 col = pow(vec3(105, 126, 78) / 255.0, vec3(2.2));
        // Voxel particle_voxel = Voxel(1, 0.99, vec3(0, 0, 1), col);
        // self.packed_voxel = pack_voxel(particle_voxel);
        // rand_seed(self_index);
        // self.pos = deref(gpu_input).player.pos + deref(gpu_input).player.player_unit_offset + vec3(rand(), rand(), 0) * 5 + vec3(0, 0, -1); // deref(gpu_input).player.forward * 1 + vec3(0, 0, -2.5) + deref(gpu_input).player.lateral * 1.5;
    }
}

void particle_voxelize(daxa_RWBufferPtr(uint) placed_voxel_particles, daxa_RWBufferPtr(VoxelParticlesState) particles_state, in SimulatedVoxelParticle self, uint self_index) {
    uvec3 my_voxel_i = uvec3(self.pos * VOXEL_SCL);
    const uvec3 max_pos = uvec3(2048);
    if (my_voxel_i.x < max_pos.x && my_voxel_i.y < max_pos.y && my_voxel_i.z < max_pos.z) {
        // Commented out, since placing particles in the voxel volume is not well optimized yet.

        // uint my_place_index = atomicAdd(deref(particles_state).place_count, 1);
        // if (my_place_index == 0) {
        //     ChunkWorkItem brush_work_item;
        //     brush_work_item.i = uvec3(0);
        //     brush_work_item.brush_id = BRUSH_FLAGS_PARTICLE_BRUSH;
        //     brush_work_item.brush_input = deref(particles_state).brush_input;
        //     zero_work_item_children(brush_work_item);
        //     queue_root_work_item(particles_state, brush_work_item);
        // }
        // deref(advance(placed_voxel_particles, my_place_index)) = self_index;
        // atomicMin(deref(particles_state).place_bounds_min.x, my_voxel_i.x);
        // atomicMin(deref(particles_state).place_bounds_min.y, my_voxel_i.y);
        // atomicMin(deref(particles_state).place_bounds_min.z, my_voxel_i.z);
        // atomicMax(deref(particles_state).place_bounds_max.x, my_voxel_i.x);
        // atomicMax(deref(particles_state).place_bounds_max.y, my_voxel_i.y);
        // atomicMax(deref(particles_state).place_bounds_max.z, my_voxel_i.z);
    }
}

void particle_point_pos_and_size(in vec3 osPosition, in float voxelSize, in mat4 objectToScreenMatrix, in vec2 halfScreenSize, inout vec4 position, inout float pointSize) {
    const vec4 quadricMat = vec4(1.0, 1.0, 1.0, -1.0);
    float sphereRadius = voxelSize * 1.732051;
    vec4 sphereCenter = vec4(osPosition.xyz, 1.0);
    mat4 modelViewProj = transpose(objectToScreenMatrix);
    mat3x4 matT = mat3x4(mat3(modelViewProj[0].xyz, modelViewProj[1].xyz, modelViewProj[3].xyz) * sphereRadius);
    matT[0].w = dot(sphereCenter, modelViewProj[0]);
    matT[1].w = dot(sphereCenter, modelViewProj[1]);
    matT[2].w = dot(sphereCenter, modelViewProj[3]);
    mat3x4 matD = mat3x4(matT[0] * quadricMat, matT[1] * quadricMat, matT[2] * quadricMat);
    vec4 eqCoefs =
        vec4(dot(matD[0], matT[2]), dot(matD[1], matT[2]), dot(matD[0], matT[0]), dot(matD[1], matT[1])) / dot(matD[2], matT[2]);
    vec4 outPosition = vec4(eqCoefs.x, eqCoefs.y, 0.0, 1.0);
    vec2 AABB = sqrt(eqCoefs.xy * eqCoefs.xy - eqCoefs.zw);
    AABB *= halfScreenSize * 2.0f;
    position.xy = outPosition.xy * position.w * halfScreenSize + halfScreenSize;
    pointSize = max(AABB.x, AABB.y);
}

void particle_render(daxa_RWBufferPtr(ParticleVertex) cube_rendered_particle_verts, daxa_RWBufferPtr(ParticleVertex) splat_rendered_particle_verts,
                     daxa_RWBufferPtr(VoxelParticlesState) particles_state, daxa_BufferPtr(GpuInput) gpu_input, vec3 pos, uint self_index) {
    float voxel_radius = (1023.0 / 1024.0 * VOXEL_SIZE) * 0.5;
    vec3 center_ws = get_particle_worldspace_origin(gpu_input, pos);
    mat4 world_to_sample = deref(gpu_input).player.cam.view_to_sample * deref(gpu_input).player.cam.world_to_view;
    vec2 half_screen_size = vec2(deref(gpu_input).frame_dim) * 0.5;
    float ps_size = 0.0;
    vec4 cs_pos = vec4(0, 0, 0, 1);
    particle_point_pos_and_size(center_ws, voxel_radius, world_to_sample, half_screen_size, cs_pos, ps_size);
    if (any(lessThan(cs_pos.xy + ps_size * 0.5, vec2(0))) ||
        any(greaterThan(cs_pos.xy - ps_size * 0.5, vec2(half_screen_size * 2.0)))) {
        return;
    }

    const float splat_size_threshold = 5.0;
    const bool should_splat = ps_size < splat_size_threshold;

    if (ps_size <= 0.0) {
        return;
    }

    if (should_splat) {
        // TODO: Stochastic pruning?
        uint my_render_index = atomicAdd(deref(particles_state).splat_draw_params.vertex_count, 1);
        deref(advance(splat_rendered_particle_verts, my_render_index)) = ParticleVertex(pos, self_index);
    } else {
        uint my_render_index = atomicAdd(deref(particles_state).cube_draw_params.instance_count, 1);
        deref(advance(cube_rendered_particle_verts, my_render_index)) = ParticleVertex(pos, self_index);
    }
}

void particle_shade(daxa_BufferPtr(GrassStrand) grass_strands, daxa_BufferPtr(SimulatedVoxelParticle) simulated_voxel_particles,
                    daxa_BufferPtr(GpuInput) gpu_input, ParticleVertex particle_vertex, out uint packed_voxel_data, out vec3 nrm, out vec3 vs_velocity) {
    nrm = vec3(0, 0, 1);
    vs_velocity = vec3(0);

    uint particle_index = particle_vertex.id;

    vec3 pos = vec3(0);
    vec3 prev_pos = vec3(0);

    if (particle_index < MAX_SIMULATED_VOXEL_PARTICLES) {
        float dt = min(deref(gpu_input).delta_time, 0.01);
        SimulatedVoxelParticle particle = deref(advance(simulated_voxel_particles, particle_index));
        pos = get_particle_worldspace_origin(gpu_input, particle.pos);
        prev_pos = get_particle_worldspace_origin(gpu_input, particle.pos - particle.vel * dt);
        packed_voxel_data = particle.packed_voxel.data;
    } else {
        particle_index -= MAX_SIMULATED_VOXEL_PARTICLES;
        pos = get_particle_worldspace_origin(gpu_input, particle_vertex.pos);
        prev_pos = pos;
        GrassStrand strand = deref(advance(grass_strands, particle_index));
        Voxel grass_voxel = unpack_voxel(strand.packed_voxel);
        // grass_voxel.color *= vec3(length(pos - strand.origin));
        nrm = grass_voxel.normal;
        packed_voxel_data = pack_voxel(grass_voxel).data;
    }

    vec3 extra_vel = vec3(deref(gpu_input).player.player_unit_offset - deref(gpu_input).player.prev_unit_offset);
    prev_pos += extra_vel;
    vec4 vs_pos = (deref(gpu_input).player.cam.world_to_view * vec4(pos, 1));
    vec4 prev_vs_pos = (deref(gpu_input).player.cam.world_to_view * vec4(prev_pos, 1));
    vs_velocity = (prev_vs_pos.xyz / prev_vs_pos.w) - (vs_pos.xyz / vs_pos.w);

    // TODO: Fix the face-normals for particles. The ray_hit_ws function returns the voxel center.

    // ViewRayContext vrc_particle = vrc_from_uv_and_depth(gpu_input, uv, particles_depth);
    // vec3 ppos = ray_hit_ws(vrc_particle);
    // nrm = ppos - (pos + 0.5 * VOXEL_SIZE);
    // ppos = abs(nrm);
    // if (ppos.x > ppos.y) {
    //     if (ppos.x > ppos.z) {
    //         nrm = vec3(sign(nrm.x), 0, 0);
    //     } else {
    //         nrm = vec3(0, 0, sign(nrm.z));
    //     }
    // } else {
    //     if (ppos.y > ppos.z) {
    //         nrm = vec3(0, sign(nrm.y), 0);
    //     } else {
    //         nrm = vec3(0, 0, sign(nrm.z));
    //     }
    // }

    // nrm = normalize(deref(gpu_input).player.pos - pos);
}
