#pragma once

#include <utils/math_const.glsl>

// Random functions

float interleaved_gradient_noise(daxa_u32vec2 px) {
    return fract(52.9829189 * fract(0.06711056 * float(px.x) + 0.00583715 * float(px.y)));
}

// Jenkins hash function
daxa_u32 good_rand_hash(daxa_u32 x) {
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}
daxa_u32 good_rand_hash(daxa_u32vec2 v) { return good_rand_hash(v.x ^ good_rand_hash(v.y)); }
daxa_u32 good_rand_hash(daxa_u32vec3 v) {
    return good_rand_hash(v.x ^ good_rand_hash(v.y) ^ good_rand_hash(v.z));
}
daxa_u32 good_rand_hash(daxa_u32vec4 v) {
    return good_rand_hash(v.x ^ good_rand_hash(v.y) ^ good_rand_hash(v.z) ^ good_rand_hash(v.w));
}
daxa_f32 good_rand_float_construct(daxa_u32 m) {
    const daxa_u32 ieee_mantissa = 0x007FFFFFu;
    const daxa_u32 ieee_one = 0x3F800000u;
    m &= ieee_mantissa;
    m |= ieee_one;
    daxa_f32 f = uintBitsToFloat(m);
    return f - 1.0;
}
daxa_f32 good_rand(daxa_f32 x) { return good_rand_float_construct(good_rand_hash(floatBitsToUint(x))); }
daxa_f32 good_rand(daxa_f32vec2 v) { return good_rand_float_construct(good_rand_hash(floatBitsToUint(v))); }
daxa_f32 good_rand(daxa_f32vec3 v) { return good_rand_float_construct(good_rand_hash(floatBitsToUint(v))); }
daxa_f32 good_rand(daxa_f32vec4 v) { return good_rand_float_construct(good_rand_hash(floatBitsToUint(v))); }

// Jenkins hash function. TODO: check if we need a better hash function
uint hash1(uint x) { return good_rand_hash(x); }
uint hash1_mut(inout uint h) {
    uint res = h;
    h = hash1(h);
    return res;
}
uint hash_combine2(uint x, uint y) {
    const uint M = 1664525u, C = 1013904223u;
    uint seed = (x * M + y + C) * M;
    // Tempering (from Matsumoto)
    seed ^= (seed >> 11u);
    seed ^= (seed << 7u) & 0x9d2c5680u;
    seed ^= (seed << 15u) & 0xefc60000u;
    seed ^= (seed >> 18u);
    return seed;
}
uint hash2(daxa_u32vec2 v) { return hash_combine2(v.x, hash1(v.y)); }
uint hash3(daxa_u32vec3 v) { return hash_combine2(v.x, hash2(v.yz)); }
uint hash4(daxa_u32vec4 v) { return hash_combine2(v.x, hash3(v.yzw)); }

daxa_u32 _rand_state;
void rand_seed(daxa_u32 seed) {
    _rand_state = seed;
}

daxa_f32 rand() {
    // https://www.pcg-random.org/
    _rand_state = _rand_state * 747796405u + 2891336453u;
    daxa_u32 result = ((_rand_state >> ((_rand_state >> 28u) + 4u)) ^ _rand_state) * 277803737u;
    result = (result >> 22u) ^ result;
    return result / 4294967295.0;
}

daxa_f32 rand_normal_dist() {
    daxa_f32 theta = 2.0 * M_PI * rand();
    daxa_f32 rho = sqrt(-2.0 * log(rand()));
    return rho * cos(theta);
}

daxa_f32vec3 rand_dir() {
    return normalize(daxa_f32vec3(
        rand_normal_dist(),
        rand_normal_dist(),
        rand_normal_dist()));
}

daxa_f32vec3 rand_hemi_dir(daxa_f32vec3 nrm) {
    daxa_f32vec3 result = rand_dir();
    return result * sign(dot(nrm, result));
}

daxa_f32vec3 rand_lambertian_nrm(daxa_f32vec3 nrm) {
    return normalize(nrm + rand_dir());
}

daxa_f32vec2 rand_circle_pt(daxa_f32vec2 random_input) {
    daxa_f32 theta = 2.0 * M_PI * random_input.x;
    daxa_f32 mag = sqrt(random_input.y);
    return daxa_f32vec2(cos(theta), sin(theta)) * mag;
}

daxa_f32vec2 rand_circle_pt() {
    return rand_circle_pt(daxa_f32vec2(rand(), rand()));
}

// Random noise
daxa_f32vec3 hash33(daxa_f32vec3 p3) {
    p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}
