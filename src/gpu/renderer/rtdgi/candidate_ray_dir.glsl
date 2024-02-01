#pragma once

vec4 blue_noise_for_pixel(uvec2 px, uint n) {
    // const uvec2 tex_dims = uvec2(128, 128);
    // const uvec2 offset = uvec2(r2_sequence(n) * tex_dims);
    vec2 blue_noise = texelFetch(daxa_texture3D(blue_noise_vec2), ivec3(px, n) & ivec3(127, 127, 63), 0).yz;
    return blue_noise.xyxy;
}

vec3 rtdgi_candidate_ray_dir(uvec2 px, daxa_f32mat3x3 tangent_to_world) {
    vec2 urand = blue_noise_for_pixel(px, deref(gpu_input).frame_index).xy;
    vec3 wi = uniform_sample_hemisphere(urand);
    return tangent_to_world * wi;
}
