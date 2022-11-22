#include <shared/shared.inl>

DAXA_USE_PUSH_CONSTANT(OpticalDepthCompPush)

#include <utils/rand.glsl>

#define RAYTRACE_NO_VOXELS
#include <utils/raytrace.glsl>

#define SKY_ONLY_OPTICAL_DEPTH
#include <utils/sky.glsl>

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec3 pixel_i = gl_GlobalInvocationID.xyz;
    f32vec2 uv = f32vec2(pixel_i.xy) / 512;
    f32 height = (uv.y) * (ATMOSPHERE_RADIUS - PLANET_RADIUS) + PLANET_RADIUS;
    f32 angle = (uv.x) * PI;

    f32 y = -2 * uv.x + 1;
    f32 x = sin(acos(y));

    Ray ray = Ray(f32vec3(0, 0, height), f32vec3(x, 0, y), f32vec3(0, 0, 0));
    ray.nrm = normalize(ray.nrm);
    ray.inv_nrm = 1.0 / ray.nrm;

    const f32 epsilon = 0.0001;
    ray.o += ray.nrm * epsilon;

    f32 depth = calc_atmosphere_depth(ray);

    f32 result = optical_depth(ray, depth - epsilon * 2);

    imageStore(
        get_image(image2D, daxa_push_constant.image_id),
        i32vec2(pixel_i.xy),
        f32vec4(result, 0, 0, 0));
}
