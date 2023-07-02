#include <shared/shared.inl>

#include <utils/trace.glsl>
#include <utils/voxels.glsl>

#define SKY_COL (f32vec3(0.02, 0.05, 0.90) * 4)
#define SKY_COL_B (f32vec3(0.11, 0.10, 0.54))

// #define SUN_TIME (deref(gpu_input).time)
#define SUN_TIME 0.9
#define SUN_COL (f32vec3(1, 0.90, 0.4) * 20)
#define SUN_DIR normalize(f32vec3(1.2 * abs(sin(SUN_TIME)), -cos(SUN_TIME), abs(sin(SUN_TIME))))

f32vec3 sample_sky_ambient(f32vec3 nrm) {
    f32 sun_val = dot(nrm, SUN_DIR) * 0.1 + 0.06;
    sun_val = pow(sun_val, 2) * 0.2;
    f32 sky_val = clamp(dot(nrm, f32vec3(0, 0, -1)) * 0.2 + 0.5, 0, 1);
    return mix(SKY_COL + sun_val * SUN_COL, SKY_COL_B, pow(sky_val, 2));
}

f32vec3 sample_sky(f32vec3 nrm) {
    f32vec3 light = sample_sky_ambient(nrm);
    f32 sun_val = dot(nrm, SUN_DIR) * 0.5 + 0.5;
    sun_val = sun_val * 200 - 199;
    sun_val = pow(clamp(sun_val * 1.1, 0, 1), 200);
    light += sun_val * SUN_COL;
    return light;
}

u32vec3 chunk_n;
f32vec3 cam_pos;
u32vec2 pixel_i;

bool is_hit(f32vec3 pos) {
    BoundingBox b;
    b.bound_min = f32vec3(0, 0, 0);
    b.bound_max = f32vec3(chunk_n) * (CHUNK_SIZE / VOXEL_SCL);

    return inside(pos, b);
}

struct HitInfo {
    f32vec3 diff_col;
    f32vec3 emit_col;
    f32vec3 nrm;
    f32 fresnel_fac;
    bool is_hit;
};

HitInfo get_hit_info(in out f32vec3 pos, f32vec3 ray_dir, bool is_primary) {
    HitInfo result;

    f32vec4 raster_color = texelFetch(daxa_texture2D(raster_color_image), i32vec2(pixel_i), 0);

    f32 dist2_a = dot(pos - cam_pos, pos - cam_pos);
    f32 dist2_b = dot(raster_color.xyz - cam_pos, raster_color.xyz - cam_pos);
    if (is_primary && raster_color.a != 0.0 && abs(dist2_b) < abs(dist2_a)) {
        uint id = uint(raster_color.w - 1);
        pos = raster_color.xyz;
        SimulatedVoxelParticle self = deref(simulated_voxel_particles[id]);
        result.is_hit = true;
        if (self.flags == 2) {
            // result.diff_col = f32vec3(0.2);
            // result.emit_col = f32vec3(0.0);
        } else {
            // result.emit_col = f32vec3(0.8, 0.06, 0.01) * dot(self.vel, self.vel) * 0.1 * (0.5 + good_rand(floor(pos * 8) + self.vel * 3));
            // result.diff_col = f32vec3(0.01);
        }
        result.diff_col = f32vec3(0.9);
        result.emit_col = f32vec3(0);
        // result.nrm = f32vec3(0, 0, 1);
        result.nrm = scene_nrm(voxel_malloc_global_allocator, voxel_chunks, chunk_n, pos);
    } else {
        result.is_hit = is_hit(pos);
        if (result.is_hit) {
            u32vec3 chunk_i = u32vec3(floor(pos * (f32(VOXEL_SCL) / CHUNK_SIZE)));
            u32 chunk_index = chunk_i.x + chunk_i.y * chunk_n.x + chunk_i.z * chunk_n.x * chunk_n.y;
            u32vec3 voxel_i = u32vec3(pos * VOXEL_SCL);
            u32vec3 inchunk_voxel_i = voxel_i - chunk_i * CHUNK_SIZE;
            if ((chunk_i.x < chunk_n.x) && (chunk_i.y < chunk_n.y) && (chunk_i.z < chunk_n.z)) {
                u32 voxel_data = sample_voxel_chunk(voxel_malloc_global_allocator, voxel_chunks[chunk_index], inchunk_voxel_i, true);
                f32vec4 sample_col = uint_to_float4(voxel_data);
                if ((voxel_data >> 0x18) == 2) {
                    result.diff_col = f32vec3(0);
                    result.emit_col = sample_col.rgb * 20;
                    result.is_hit = false;
                } else {
                    result.diff_col = max(sample_col.rgb, f32vec3(0.01));
                    // result.diff_col = f32vec3(0.02);
                    result.emit_col = f32vec3(0);
                }
            }
            result.nrm = scene_nrm(voxel_malloc_global_allocator, voxel_chunks, chunk_n, pos);
        } else {
            result.diff_col = f32vec3(0.0);
            result.emit_col = sample_sky(ray_dir);
            result.nrm = -ray_dir;
        }
    }

    result.fresnel_fac = pow(1.0 - dot(-ray_dir, result.nrm), 5.0);
    return result;
}

f32vec3 a_scene_nrm(f32vec3 pos) {
    f32vec3 d = fract(pos * VOXEL_SCL) - .5;
    f32vec3 ad = abs(d);
    f32 m = max(max(ad.x, ad.y), ad.z);
    return -(abs(sign(ad - m)) - 1.) * sign(d);
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    f32vec2 frame_dim = deref(gpu_input).frame_dim;
    f32vec2 inv_frame_dim = f32vec2(1.0, 1.0) / frame_dim;
    f32 aspect = frame_dim.x * inv_frame_dim.y;

    pixel_i = gl_GlobalInvocationID.xy;

    f32vec3 col = f32vec3(0);
    f32 accepted_count = 1;
    f32vec2 uv = f32vec2(pixel_i) * inv_frame_dim;
    uv = (uv - 0.5) * f32vec2(aspect, 1.0) * 2.0;
    chunk_n = u32vec3(1u << deref(settings).log2_chunks_per_axis);
    rand_seed(pixel_i.x + pixel_i.y * deref(gpu_input).frame_dim.x + u32(deref(gpu_input).time * 719393));
    f32vec3 blue_noise = texelFetch(daxa_texture3D(blue_noise_cosine_vec3), ivec3(pixel_i, deref(gpu_input).frame_index) & ivec3(127, 127, 63), 0).xyz * 2 - 1;
    cam_pos = create_view_pos(deref(globals).player);
    f32vec3 cam_dir = create_view_dir(deref(globals).player, uv);
    f32vec4 render_pos = imageLoad(daxa_image2D(render_pos_image_id), i32vec2(pixel_i));
    f32vec3 hit_pos = render_pos.xyz;

    HitInfo hit_info = get_hit_info(hit_pos, cam_dir, true);
    col = hit_info.diff_col + hit_info.emit_col;

    // col = a_scene_nrm(hit_pos);
    // col = fract(hit_pos * 0.125);
    // if (length(uv) < 1.0 / 64) {
    //     col *= 0.1;
    // }

    // imageStore(daxa_image2D(render_col_image_id), i32vec2(pixel_i), f32vec4(hsv2rgb(vec3(0.6 + render_pos.w * 0.008, 1.0, min(1.0, render_pos.w * 0.01))), accepted_count));
    imageStore(daxa_image2D(render_col_image_id), i32vec2(pixel_i), f32vec4(col, accepted_count));
}
