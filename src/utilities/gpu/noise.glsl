#pragma once

#include <utilities/gpu/math.glsl>

vec4 noise(daxa_ImageViewIndex noise_texture, daxa_SamplerId noise_sampler, vec3 x) {
    const float offset = 1.0 / 512.0;
    vec4 gz0 = textureGather(daxa_sampler2DArray(noise_texture, noise_sampler), vec3(x.xy / 256.0, (int(floor(x.z)) + 0) & 0xff));
    vec4 gz1 = textureGather(daxa_sampler2DArray(noise_texture, noise_sampler), vec3(x.xy / 256.0, (int(floor(x.z)) + 1) & 0xff));
    x.xy = x.xy - 0.5 + offset;

    vec3 w = fract(x);
    vec3 u = w * w * (3.0 - 2.0 * w);
    vec3 du = 6.0 * w * (1.0 - w);

    // float z0 = mix(mix(gz0.w, gz0.z, w.x), mix(gz0.x, gz0.y, w.x), w.y);
    // float z1 = mix(mix(gz1.w, gz1.z, w.x), mix(gz1.x, gz1.y, w.x), w.y);
    // float dist = mix(z0, z1, w.z);
    // vec3 nrm = (z1 - z0) * du;

    float a = gz0.w, b = gz0.z, c = gz0.x, d = gz0.y;
    float e = gz1.w, f = gz1.z, g = gz1.x, h = gz1.y;

    float k0 = a, k1 = b - a, k2 = c - a, k3 = e - a;
    float k4 = a - b - c + d;
    float k5 = a - c - e + g;
    float k6 = a - b - e + f;
    float k7 = -a + b + c - d + e - f - g + h;
    float dist = k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z;
    vec3 nrm = du * (vec3(k1, k2, k3) + u.yzx * vec3(k4, k5, k6) + u.zxy * vec3(k6, k4, k5) + k7 * u.yzx * u.zxy);

    return vec4(dist, nrm);
}

struct FractalNoiseConfig {
    float amplitude;
    float persistance;
    float scale;
    float lacunarity;
    uint octaves;
};

vec4 fractal_noise(daxa_ImageViewIndex noise_texture, daxa_SamplerId noise_sampler, vec3 pos, FractalNoiseConfig config) {
    const float scale = config.scale;
    float a = 0.0;
    float b = 0.5;
    float f = 1.0;
    vec3 d = vec3(0.0);
    for (uint i = 0; i < config.octaves; ++i) {
        vec4 n = noise(noise_texture, noise_sampler, f * pos * scale);
        a += b * n.x;
        d += b * n.yzw * f * scale;
        b *= config.persistance;
        f *= config.lacunarity;
    }
    return vec4(a, d);
}

float analytical_noise_hash(in float n) {
    return fract(sin(n) * 753.5453123);
}

vec4 sd_analytical_noise(in vec3 x) {
    vec3 p = floor(x);
    vec3 w = fract(x);
    vec3 u = w * w * (3.0 - 2.0 * w);
    vec3 du = 6.0 * w * (1.0 - w);
    float n = p.x + p.y * 157.0 + 113.0 * p.z;
    float a = analytical_noise_hash(n + 0.0);
    float b = analytical_noise_hash(n + 1.0);
    float c = analytical_noise_hash(n + 157.0);
    float d = analytical_noise_hash(n + 158.0);
    float e = analytical_noise_hash(n + 113.0);
    float f = analytical_noise_hash(n + 114.0);
    float g = analytical_noise_hash(n + 270.0);
    float h = analytical_noise_hash(n + 271.0);
    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k3 = e - a;
    float k4 = a - b - c + d;
    float k5 = a - c - e + g;
    float k6 = a - b - e + f;
    float k7 = -a + b + c - d + e - f - g + h;
    float dist = k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z;
    vec3 nrm = du * (vec3(k1, k2, k3) + u.yzx * vec3(k4, k5, k6) + u.zxy * vec3(k6, k4, k5) + k7 * u.yzx * u.zxy);
    return vec4(dist, nrm);
}

vec4 sd_analytical_fractal_noise(in vec3 x) {
    const float scale = 0.05;
    float a = 0.0;
    float b = 0.5;
    float f = 1.0;
    vec3 d = vec3(0.0);
    for (uint i = 0; i < 4; ++i) {
        vec4 n = sd_analytical_noise(f * x * scale);
        a += b * n.x;
        d += b * n.yzw * f * scale;
        b *= 0.5;
        f *= 1.8;
    }
    return vec4(a, d);
}

float fractal_noise2(vec3 pos, FractalNoiseConfig config) {
    float value = 0.0;
    float max_value = 0.0;
    float amplitude = config.amplitude;
    mat3 rot_mat = mat3(
        0.2184223, -0.5347182, 0.8163137,
        0.9079879, -0.1951438, -0.3707788,
        0.3575608, 0.8221893, 0.4428939);
    for (uint i = 0; i < config.octaves; ++i) {
        pos = (rot_mat * pos) + vec3(71.444, 25.170, -54.766);
        vec3 p = pos * config.scale;
        value += sd_analytical_noise(p).x * config.amplitude;
        max_value += config.amplitude;
        config.amplitude *= config.persistance;
        config.scale *= config.lacunarity;
    }
    return value / max_value * amplitude;
}

float voronoi_noise(vec3 pos) {
    float value = 1e38;

    for (int zi = 0; zi < 3; ++zi) {
        for (int yi = 0; yi < 3; ++yi) {
            for (int xi = 0; xi < 3; ++xi) {
                vec3 p = pos + vec3(xi - 1, yi - 1, zi - 1);
                p = floor(p) + 0.5;
                p += vec3(rand(), rand(), rand());
                value = min(value, dot(pos - p, pos - p));
            }
        }
    }

    return value;
}

float fractal_voronoi_noise(vec3 pos, FractalNoiseConfig config) {
    float value = 0.0;
    float max_value = 0.0;
    float amplitude = config.amplitude;
    mat3 rot_mat = mat3(
        0.2184223, -0.5347182, 0.8163137,
        0.9079879, -0.1951438, -0.3707788,
        0.3575608, 0.8221893, 0.4428939);
    for (uint i = 0; i < config.octaves; ++i) {
        pos = (rot_mat * pos) + vec3(71.444, 25.170, -54.766);
        vec3 p = pos * config.scale;
        value += voronoi_noise(p) * config.amplitude;
        max_value += config.amplitude;
        config.amplitude *= config.persistance;
        config.scale *= config.lacunarity;
    }
    return value / max_value * amplitude;
}

// Value noise + fbm noise
float fbm_value_noise_hash(in ivec2 p) {
    int n = p.x * 3 + p.y * 113;
    n = (n << 13) ^ n;
    n = n * (n * n * 15731 + 789221) + 1376312589;
    return -1.0 + 2.0 * float(n & 0x0fffffff) / float(0x0fffffff);
}
float value_noise(in vec2 p) {
    ivec2 i = ivec2(floor(p));
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(fbm_value_noise_hash(i + ivec2(0, 0)),
                   fbm_value_noise_hash(i + ivec2(1, 0)), u.x),
               mix(fbm_value_noise_hash(i + ivec2(0, 1)),
                   fbm_value_noise_hash(i + ivec2(1, 1)), u.x),
               u.y);
}
float fbm2(vec2 uv) {
    float f = 0;
    mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
    f = 0.5000 * value_noise(uv);
    uv = m * uv;
    f += 0.2500 * value_noise(uv);
    uv = m * uv;
    return f * .5 + .5;
}
