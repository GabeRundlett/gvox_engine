#pragma once

vec4 blue_noise_for_pixel(daxa_ImageViewIndex blue_noise_tex, uvec2 px, uint n) {
    // const uvec2 tex_dims = uvec2(128, 128);
    // const uvec2 offset = uvec2(r2_sequence(n) * tex_dims);
    vec2 blue_noise = texelFetch(daxa_texture3D(blue_noise_tex), ivec3(px, n) & ivec3(127, 127, 63), 0).yz;
    return blue_noise.xyxy;
}

vec3 rtdgi_candidate_ray_dir(daxa_ImageViewIndex blue_noise_tex, uint frame_index, uvec2 px, mat3 tangent_to_world) {
    vec2 urand = blue_noise_for_pixel(blue_noise_tex, px, frame_index).xy;
    vec3 wi = uniform_sample_hemisphere(urand);
    return tangent_to_world * wi;
}
