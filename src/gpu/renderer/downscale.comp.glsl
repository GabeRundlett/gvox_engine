#include <shared/renderer/downscale.inl>
#include <utils/downscale.glsl>
#include <utils/math.glsl>
#include <utils/safety.glsl>

DAXA_DECL_PUSH_CONSTANT(DownscaleComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(GpuGlobals) globals = push.uses.globals;
daxa_ImageViewIndex src_image_id = push.uses.src_image_id;
daxa_ImageViewIndex dst_image_id = push.uses.dst_image_id;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    uvec2 px = gl_GlobalInvocationID.xy;
    const ivec2 src_px = ivec2(px * 2 + HALFRES_SUBSAMPLE_OFFSET);

#if DOWNSCALE_DEPTH || DOWNSCALE_SSAO
    vec4 output_val = safeTexelFetch(src_image_id, src_px, 0);
#elif DOWNSCALE_NRM
    uvec4 g_buffer_value = safeTexelFetchU(src_image_id, src_px, 0);
    vec3 normal_ws = u16_to_nrm_unnormalized(g_buffer_value.y);
    vec3 normal_vs = normalize((deref(globals).player.cam.world_to_view * vec4(normal_ws, 0)).xyz);
    vec4 output_val = vec4(normal_vs, 1);
#endif

    safeImageStore(dst_image_id, ivec2(px), output_val);
}
