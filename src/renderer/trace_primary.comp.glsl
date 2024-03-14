#include <renderer/trace_primary.inl>

#include <renderer/kajiya/inc/camera.glsl>
#include <voxels/voxels.glsl>

#if TraceDepthPrepassComputeShader

DAXA_DECL_PUSH_CONSTANT(TraceDepthPrepassComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_ImageViewIndex render_depth_prepass_image = push.uses.render_depth_prepass_image;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_BufferPtr)

#define INPUT deref(gpu_input)
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    vec4 output_tex_size;
    output_tex_size.xy = deref(gpu_input).frame_dim;
    output_tex_size.zw = vec2(1.0, 1.0) / output_tex_size.xy;
    vec2 uv = get_uv(gl_GlobalInvocationID.xy * PREPASS_SCL, output_tex_size);

    ViewRayContext vrc = vrc_from_uv(gpu_input, uv);
    vec3 ray_dir = ray_dir_ws(vrc);
    vec3 cam_pos = ray_origin_ws(vrc);
    vec3 ray_pos = cam_pos;

#if ENABLE_DEPTH_PREPASS
    VoxelTraceResult trace_result = voxel_trace(VoxelTraceInfo(VOXELS_BUFFER_PTRS, ray_dir, MAX_STEPS, MAX_DIST, 16.0 * output_tex_size.w * deref(gpu_input).player.cam.clip_to_view[1][1], true), ray_pos);
#endif

    float depth = length(ray_pos - cam_pos);

    if (any(greaterThanEqual(gl_GlobalInvocationID.xy, (uvec2(output_tex_size.xy)) / PREPASS_SCL))) {
        return;
    }
    imageStore(daxa_image2D(render_depth_prepass_image), ivec2(gl_GlobalInvocationID.xy), vec4(depth, 0, 0, 0));
}
#undef INPUT

#endif

#if TracePrimaryComputeShader

DAXA_DECL_PUSH_CONSTANT(TracePrimaryComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
VOXELS_USE_BUFFERS_PUSH_USES(daxa_BufferPtr)
daxa_ImageViewIndex blue_noise_vec2 = push.uses.blue_noise_vec2;
daxa_ImageViewIndex debug_texture = push.uses.debug_texture;
daxa_ImageViewIndex render_depth_prepass_image = push.uses.render_depth_prepass_image;
daxa_ImageViewIndex g_buffer_image_id = push.uses.g_buffer_image_id;
daxa_ImageViewIndex velocity_image_id = push.uses.velocity_image_id;
daxa_ImageViewIndex vs_normal_image_id = push.uses.vs_normal_image_id;
daxa_ImageViewIndex depth_image_id = push.uses.depth_image_id;

#include <renderer/atmosphere/sky.glsl>

#define PIXEL_I gl_GlobalInvocationID.xy
#define INPUT deref(gpu_input)
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    vec4 output_tex_size = vec4(deref(gpu_input).frame_dim, 0, 0);
    output_tex_size.zw = vec2(1.0, 1.0) / output_tex_size.xy;
    vec2 uv = get_uv(PIXEL_I, output_tex_size);

    ViewRayContext vrc = vrc_from_uv(gpu_input, uv);
    vec3 ray_dir = ray_dir_ws(vrc);
    vec3 cam_pos = ray_origin_ws(vrc);

#if !ENABLE_DEPTH_PREPASS
    float prepass_depth = 0.0;
    float prepass_steps = 0.0;
#else
    float max_depth = MAX_DIST;
    float prepass_depth = max_depth;
    float prepass_steps = 0.0;

    for (int yi = -1; yi <= 1; ++yi) {
        for (int xi = -1; xi <= 1; ++xi) {
            ivec2 pt = ivec2(PIXEL_I / PREPASS_SCL) + ivec2(xi, yi);
            pt = clamp(pt, ivec2(0), ivec2(deref(gpu_input).frame_dim / PREPASS_SCL - 1));
            vec2 prepass_data = texelFetch(daxa_texture2D(render_depth_prepass_image), pt, 0).xy;
            float loaded_depth = prepass_data.x - VOXEL_SIZE;
            prepass_depth = max(min(prepass_depth, loaded_depth), 0);
            if (prepass_depth == loaded_depth || prepass_depth == max_depth) {
                prepass_steps = prepass_data.y / 4.0;
            }
        }
    }
#endif

    vec3 ray_pos = cam_pos + ray_dir * prepass_depth;

    VoxelTraceResult trace_result = voxel_trace(VoxelTraceInfo(VOXELS_BUFFER_PTRS, ray_dir, MAX_STEPS, MAX_DIST, 0.0, true), ray_pos);
    uint step_n = trace_result.step_n;

    uvec4 output_value = uvec4(0);

    // float depth = length(cam_pos - ray_pos);
    // transform depth to ndc space
    vec4 vs_pos = (deref(gpu_input).player.cam.world_to_view * vec4(ray_pos, 1));
    vec4 prev_vs_pos = (deref(gpu_input).player.cam.world_to_view * vec4(ray_pos + trace_result.vel, 1)); // when animated objects exist, this is where they'd put their velocity
    vec4 ss_pos = (deref(gpu_input).player.cam.view_to_sample * vs_pos);
    float depth = ss_pos.z / ss_pos.w;
    vec3 vs_nrm = vec3(0);
    vec3 vs_velocity = vec3(0);

#if !PER_VOXEL_NORMALS && defined(VOXELS_ORIGINAL_IMPL)
    bool is_valid = true;
    if (trace_result.dist != MAX_DIST) {
        uvec3 chunk_n = uvec3(1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);
        PackedVoxel voxel_data = sample_voxel_chunk(VOXELS_BUFFER_PTRS, chunk_n, ray_pos, trace_result.nrm * 0.5);
        Voxel voxel = unpack_voxel(voxel_data);
        is_valid = voxel.material_type == 0;
    }
    vec3 old_nrm = trace_result.nrm;
    if (!is_valid) {
        trace_result.nrm = vec3(0.0);
    }
    vec3 valid_nrm = vec3(0);
    uint thread2x2_root_index = gl_SubgroupInvocationID & (~(8 | 1));
    valid_nrm.x += subgroupBroadcast(trace_result.nrm.x, thread2x2_root_index + 0);
    valid_nrm.y += subgroupBroadcast(trace_result.nrm.y, thread2x2_root_index + 0);
    valid_nrm.z += subgroupBroadcast(trace_result.nrm.z, thread2x2_root_index + 0);
    valid_nrm.x += subgroupBroadcast(trace_result.nrm.x, thread2x2_root_index + 1);
    valid_nrm.y += subgroupBroadcast(trace_result.nrm.y, thread2x2_root_index + 1);
    valid_nrm.z += subgroupBroadcast(trace_result.nrm.z, thread2x2_root_index + 1);
    valid_nrm.x += subgroupBroadcast(trace_result.nrm.x, thread2x2_root_index + 8);
    valid_nrm.y += subgroupBroadcast(trace_result.nrm.y, thread2x2_root_index + 8);
    valid_nrm.z += subgroupBroadcast(trace_result.nrm.z, thread2x2_root_index + 8);
    valid_nrm.x += subgroupBroadcast(trace_result.nrm.x, thread2x2_root_index + 9);
    valid_nrm.y += subgroupBroadcast(trace_result.nrm.y, thread2x2_root_index + 9);
    valid_nrm.z += subgroupBroadcast(trace_result.nrm.z, thread2x2_root_index + 9);
    if (!is_valid) {
        trace_result.nrm = normalize(valid_nrm);
    }
    if (dot(valid_nrm, valid_nrm) == 0.0) {
        trace_result.nrm = old_nrm;
    }
#endif

    if (trace_result.dist == MAX_DIST) {
        output_value.y = nrm_to_u16(vec3(0, 0, 1));
        depth = 0.0;
    } else {
        output_value.x = trace_result.voxel_data.data;
        output_value.y = nrm_to_u16(trace_result.nrm);
        vs_nrm = (deref(gpu_input).player.cam.world_to_view * vec4(trace_result.nrm, 0)).xyz;
        vs_velocity = (prev_vs_pos.xyz / prev_vs_pos.w) - (vs_pos.xyz / vs_pos.w);
    }
    output_value.z = floatBitsToUint(depth);

    if (any(greaterThanEqual(gl_GlobalInvocationID.xy, uvec2(output_tex_size.xy)))) {
        return;
    }

    vs_nrm *= -sign(dot(ray_dir_vs(vrc), vs_nrm));

    imageStore(daxa_uimage2D(g_buffer_image_id), ivec2(gl_GlobalInvocationID.xy), output_value);
    imageStore(daxa_image2D(velocity_image_id), ivec2(gl_GlobalInvocationID.xy), vec4(vs_velocity, 0));
    imageStore(daxa_image2D(vs_normal_image_id), ivec2(gl_GlobalInvocationID.xy), vec4(vs_nrm * 0.5 + 0.5, 0));
    imageStore(daxa_image2D(depth_image_id), ivec2(gl_GlobalInvocationID.xy), vec4(depth, 0, 0, 0));
}
#undef INPUT

#endif

#if CompositeParticlesComputeShader

DAXA_DECL_PUSH_CONSTANT(CompositeParticlesComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(ParticleVertex) cube_rendered_particle_verts = push.uses.cube_rendered_particle_verts;
daxa_BufferPtr(ParticleVertex) splat_rendered_particle_verts = push.uses.splat_rendered_particle_verts;
daxa_ImageViewIndex g_buffer_image_id = push.uses.g_buffer_image_id;
daxa_ImageViewIndex velocity_image_id = push.uses.velocity_image_id;
daxa_ImageViewIndex particles_image_id = push.uses.particles_image_id;
daxa_ImageViewIndex particles_depth_image_id = push.uses.particles_depth_image_id;

#include <voxels/particles/particle.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    vec4 output_tex_size = vec4(deref(gpu_input).frame_dim, 0, 0);
    output_tex_size.zw = vec2(1.0, 1.0) / output_tex_size.xy;

    uvec4 g_buffer_value = texelFetch(daxa_utexture2D(g_buffer_image_id), ivec2(gl_GlobalInvocationID.xy), 0);
    vec3 vs_velocity = texelFetch(daxa_texture2D(velocity_image_id), ivec2(gl_GlobalInvocationID.xy), 0).xyz;
    vec3 nrm = u16_to_nrm(g_buffer_value.y);
    float depth = uintBitsToFloat(g_buffer_value.z);

    vec2 uv = get_uv(gl_GlobalInvocationID.xy, output_tex_size);

#if MAX_RENDERED_VOXEL_PARTICLES > 0
    uint particle_render_id = texelFetch(daxa_utexture2D(particles_image_id), ivec2(gl_GlobalInvocationID.xy), 0).r - 1;
    ParticleVertex particle_vert;
    if (particle_render_id < MAX_RENDERED_VOXEL_PARTICLES) {
        particle_vert = deref(advance(cube_rendered_particle_verts, particle_render_id));
    } else {
        particle_vert = deref(advance(splat_rendered_particle_verts, particle_render_id - MAX_RENDERED_VOXEL_PARTICLES));
    }
    float particles_depth = texelFetch(daxa_texture2D(particles_depth_image_id), ivec2(gl_GlobalInvocationID.xy), 0).r;
    bool is_particle = false;
    if (particles_depth > depth) {
        depth = particles_depth;
        is_particle = true;
    }

    ViewRayContext vrc = vrc_from_uv_and_biased_depth(gpu_input, uv, depth);

    if (is_particle) {
        vec3 hit_ws = ray_hit_ws(vrc);
        particle_shade(grass_strands, simulated_voxel_particles, vrc, gpu_input, particle_vert, g_buffer_value.x, nrm, vs_velocity);
        g_buffer_value.y = nrm_to_u16(nrm);
    }
#endif
    g_buffer_value.z = floatBitsToUint(depth);
    vec3 vs_nrm = (deref(gpu_input).player.cam.world_to_view * vec4(nrm, 0)).xyz;

    vs_nrm *= -sign(dot(ray_dir_vs(vrc), vs_nrm));

    if (any(greaterThanEqual(gl_GlobalInvocationID.xy, uvec2(output_tex_size.xy)))) {
        return;
    }
}

#endif
