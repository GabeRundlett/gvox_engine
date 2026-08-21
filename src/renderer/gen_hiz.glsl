#include "gen_hiz.inl"

DAXA_DECL_PUSH_CONSTANT(GenHizPush, push)

vec2 make_gather_uv(vec2 inv_size, uvec2 top_left_index) {
    return (vec2(top_left_index) + 1.0f) * inv_size;
}

DAXA_DECL_IMAGE_ACCESSOR(image2D, coherent, image2DCoherent)
shared float s_mins[2][GEN_HIZ_Y][GEN_HIZ_X];
void downsample_64x64(
    uvec2 local_index,
    uvec2 grid_index,
    uvec2 min_mip_size,
    int src_mip,
    int mip_count) {
    const vec2 invSize = 1.0f / vec2(min_mip_size);
    vec4 quad_values = vec4(0, 0, 0, 0);
    [[unroll]] for (uint quad_i = 0; quad_i < 4; ++quad_i) {
        ivec2 sub_i = ivec2(quad_i >> 1, quad_i & 1);
        ivec2 src_i = ivec2((grid_index * 16 + local_index) * 2 + sub_i) * 2;
        vec4 fetch;
        if (src_mip == -1) {
            src_i = ivec2(((vec2(src_i) + 1.0) * invSize) * deref(push.uses.gpu_input).frame_dim.xy);

            float t0 = texelFetch(daxa_texture2D(push.uses.src), min(src_i + ivec2(0, 0), ivec2(deref(push.uses.gpu_input).frame_dim.xy) - 1), 0).x;
            float t1 = texelFetch(daxa_texture2D(push.uses.src), min(src_i + ivec2(0, 1), ivec2(deref(push.uses.gpu_input).frame_dim.xy) - 1), 0).x;
            float t2 = texelFetch(daxa_texture2D(push.uses.src), min(src_i + ivec2(1, 0), ivec2(deref(push.uses.gpu_input).frame_dim.xy) - 1), 0).x;
            float t3 = texelFetch(daxa_texture2D(push.uses.src), min(src_i + ivec2(1, 1), ivec2(deref(push.uses.gpu_input).frame_dim.xy) - 1), 0).x;

            fetch = vec4(
                t0,
                t1,
                t2,
                t3);
        } else {
            fetch.x = imageLoad(daxa_access(image2DCoherent, push.uses.mips[src_mip]), min(src_i + ivec2(0, 0), ivec2(min_mip_size) - 1)).x;
            fetch.y = imageLoad(daxa_access(image2DCoherent, push.uses.mips[src_mip]), min(src_i + ivec2(0, 1), ivec2(min_mip_size) - 1)).x;
            fetch.z = imageLoad(daxa_access(image2DCoherent, push.uses.mips[src_mip]), min(src_i + ivec2(1, 0), ivec2(min_mip_size) - 1)).x;
            fetch.w = imageLoad(daxa_access(image2DCoherent, push.uses.mips[src_mip]), min(src_i + ivec2(1, 1), ivec2(min_mip_size) - 1)).x;
        }
        const float min_v = min(min(fetch.x, fetch.y), min(fetch.z, fetch.w));
        ivec2 dst_i = ivec2((grid_index * 16 + local_index) * 2) + sub_i;
        imageStore(daxa_image2D(push.uses.mips[src_mip + 1]), dst_i, vec4(min_v, 0, 0, 0));
        quad_values[quad_i] = min_v;
    }
    {
        const float min_v = min(min(quad_values.x, quad_values.y), min(quad_values.z, quad_values.w));
        ivec2 dst_i = ivec2(grid_index * 16 + local_index);
        imageStore(daxa_image2D(push.uses.mips[src_mip + 2]), dst_i, vec4(min_v, 0, 0, 0));
        s_mins[0][local_index.y][local_index.x] = min_v;
    }
    const uvec2 glob_wg_dst_offset0 = (uvec2(GEN_HIZ_WINDOW_X, GEN_HIZ_WINDOW_Y) * grid_index.xy) / 2;
    [[unroll]] for (uint i = 2; i < mip_count; ++i) {
        const uint ping_pong_src_index = (i & 1u);
        const uint ping_pong_dst_index = ((i + 1) & 1u);
        memoryBarrierShared();
        barrier();
        const bool invoc_active = local_index.x < (GEN_HIZ_WINDOW_X >> (i + 1)) && local_index.y < (GEN_HIZ_WINDOW_Y >> (i + 1));
        if (invoc_active) {
            const uvec2 glob_wg_offset = glob_wg_dst_offset0 >> i;
            const uvec2 src_i = local_index * 2;
            const vec4 fetch = vec4(
                s_mins[ping_pong_src_index][src_i.y + 0][src_i.x + 0],
                s_mins[ping_pong_src_index][src_i.y + 0][src_i.x + 1],
                s_mins[ping_pong_src_index][src_i.y + 1][src_i.x + 0],
                s_mins[ping_pong_src_index][src_i.y + 1][src_i.x + 1]);
            const float min_v = min(min(fetch.x, fetch.y), min(fetch.z, fetch.w));
            const uint dst_mip = src_mip + i + 1;
            if (dst_mip == 6) {
                imageStore(daxa_access(image2DCoherent, push.uses.mips[dst_mip]), ivec2(glob_wg_offset + local_index), vec4(min_v, 0, 0, 0));
            } else {
                imageStore(daxa_image2D(push.uses.mips[dst_mip]), ivec2(glob_wg_offset + local_index), vec4(min_v, 0, 0, 0));
            }
            s_mins[ping_pong_dst_index][local_index.y][local_index.x] = min_v;
        }
    }
}

shared bool s_last_workgroup;
layout(local_size_x = GEN_HIZ_X, local_size_y = GEN_HIZ_Y) in;
void main() {
    downsample_64x64(gl_LocalInvocationID.xy, gl_WorkGroupID.xy, (deref(push.uses.gpu_input).next_lower_po2_render_size * 2), -1, 6);

    if (gl_LocalInvocationID.x == 0 && gl_LocalInvocationID.y == 0) {
        const uint finished_workgroups = atomicAdd(deref(push.counter), 1) + 1;
        s_last_workgroup = finished_workgroups == push.total_workgroup_count;
    }
    memoryBarrierShared();
    barrier();

    if (s_last_workgroup) {
        downsample_64x64(gl_LocalInvocationID.xy, uvec2(0, 0), (deref(push.uses.gpu_input).next_lower_po2_render_size * 2) >> 6, 5, int(push.mip_count - 6));
    }
}
