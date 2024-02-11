#pragma once

#include <utilities/gpu/math.glsl>

daxa_f32vec4 noise(daxa_ImageViewIndex noise_texture, daxa_SamplerId noise_sampler, daxa_f32vec3 x) {
    const daxa_f32 offset = 1.0 / 512.0;
    daxa_f32vec4 gz0 = textureGather(daxa_sampler2DArray(noise_texture, noise_sampler), vec3(x.xy / 256.0, (int(floor(x.z)) + 0) & 0xff));
    daxa_f32vec4 gz1 = textureGather(daxa_sampler2DArray(noise_texture, noise_sampler), vec3(x.xy / 256.0, (int(floor(x.z)) + 1) & 0xff));
    x.xy = x.xy - 0.5 + offset;

    daxa_f32vec3 w = fract(x);
    daxa_f32vec3 u = w * w * (3.0 - 2.0 * w);
    daxa_f32vec3 du = 6.0 * w * (1.0 - w);

    // daxa_f32 z0 = mix(mix(gz0.w, gz0.z, w.x), mix(gz0.x, gz0.y, w.x), w.y);
    // daxa_f32 z1 = mix(mix(gz1.w, gz1.z, w.x), mix(gz1.x, gz1.y, w.x), w.y);
    // daxa_f32 dist = mix(z0, z1, w.z);
    // daxa_f32vec3 nrm = (z1 - z0) * du;

    daxa_f32 a = gz0.w, b = gz0.z, c = gz0.x, d = gz0.y;
    daxa_f32 e = gz1.w, f = gz1.z, g = gz1.x, h = gz1.y;

    daxa_f32 k0 = a, k1 = b - a, k2 = c - a, k3 = e - a;
    daxa_f32 k4 = a - b - c + d;
    daxa_f32 k5 = a - c - e + g;
    daxa_f32 k6 = a - b - e + f;
    daxa_f32 k7 = -a + b + c - d + e - f - g + h;
    daxa_f32 dist = k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z;
    daxa_f32vec3 nrm = du * (daxa_f32vec3(k1, k2, k3) + u.yzx * daxa_f32vec3(k4, k5, k6) + u.zxy * daxa_f32vec3(k6, k4, k5) + k7 * u.yzx * u.zxy);

    return daxa_f32vec4(dist, nrm);
}

struct FractalNoiseConfig {
    daxa_f32 amplitude;
    daxa_f32 persistance;
    daxa_f32 scale;
    daxa_f32 lacunarity;
    daxa_u32 octaves;
};

daxa_f32vec4 fractal_noise(daxa_ImageViewIndex noise_texture, daxa_SamplerId noise_sampler, daxa_f32vec3 pos, FractalNoiseConfig config) {
    const daxa_f32 scale = config.scale;
    daxa_f32 a = 0.0;
    daxa_f32 b = 0.5;
    daxa_f32 f = 1.0;
    daxa_f32vec3 d = daxa_f32vec3(0.0);
    for (daxa_u32 i = 0; i < config.octaves; ++i) {
        daxa_f32vec4 n = noise(noise_texture, noise_sampler, f * pos * scale);
        a += b * n.x;
        d += b * n.yzw * f * scale;
        b *= config.persistance;
        f *= config.lacunarity;
    }
    return daxa_f32vec4(a, d);
}

daxa_f32 analytical_noise_hash(in daxa_f32 n) {
    return fract(sin(n) * 753.5453123);
}

daxa_f32vec4 sd_analytical_noise(in daxa_f32vec3 x) {
    daxa_f32vec3 p = floor(x);
    daxa_f32vec3 w = fract(x);
    daxa_f32vec3 u = w * w * (3.0 - 2.0 * w);
    daxa_f32vec3 du = 6.0 * w * (1.0 - w);
    daxa_f32 n = p.x + p.y * 157.0 + 113.0 * p.z;
    daxa_f32 a = analytical_noise_hash(n + 0.0);
    daxa_f32 b = analytical_noise_hash(n + 1.0);
    daxa_f32 c = analytical_noise_hash(n + 157.0);
    daxa_f32 d = analytical_noise_hash(n + 158.0);
    daxa_f32 e = analytical_noise_hash(n + 113.0);
    daxa_f32 f = analytical_noise_hash(n + 114.0);
    daxa_f32 g = analytical_noise_hash(n + 270.0);
    daxa_f32 h = analytical_noise_hash(n + 271.0);
    daxa_f32 k0 = a;
    daxa_f32 k1 = b - a;
    daxa_f32 k2 = c - a;
    daxa_f32 k3 = e - a;
    daxa_f32 k4 = a - b - c + d;
    daxa_f32 k5 = a - c - e + g;
    daxa_f32 k6 = a - b - e + f;
    daxa_f32 k7 = -a + b + c - d + e - f - g + h;
    daxa_f32 dist = k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z;
    daxa_f32vec3 nrm = du * (daxa_f32vec3(k1, k2, k3) + u.yzx * daxa_f32vec3(k4, k5, k6) + u.zxy * daxa_f32vec3(k6, k4, k5) + k7 * u.yzx * u.zxy);
    return daxa_f32vec4(dist, nrm);
}

daxa_f32vec4 sd_analytical_fractal_noise(in daxa_f32vec3 x) {
    const daxa_f32 scale = 0.05;
    daxa_f32 a = 0.0;
    daxa_f32 b = 0.5;
    daxa_f32 f = 1.0;
    daxa_f32vec3 d = daxa_f32vec3(0.0);
    for (daxa_u32 i = 0; i < 4; ++i) {
        daxa_f32vec4 n = sd_analytical_noise(f * x * scale);
        a += b * n.x;
        d += b * n.yzw * f * scale;
        b *= 0.5;
        f *= 1.8;
    }
    return daxa_f32vec4(a, d);
}

daxa_f32 fractal_noise2(daxa_f32vec3 pos, FractalNoiseConfig config) {
    daxa_f32 value = 0.0;
    daxa_f32 max_value = 0.0;
    daxa_f32 amplitude = config.amplitude;
    daxa_f32mat3x3 rot_mat = daxa_f32mat3x3(
        0.2184223, -0.5347182, 0.8163137,
        0.9079879, -0.1951438, -0.3707788,
        0.3575608, 0.8221893, 0.4428939);
    for (daxa_u32 i = 0; i < config.octaves; ++i) {
        pos = (rot_mat * pos) + daxa_f32vec3(71.444, 25.170, -54.766);
        daxa_f32vec3 p = pos * config.scale;
        value += sd_analytical_noise(p).x * config.amplitude;
        max_value += config.amplitude;
        config.amplitude *= config.persistance;
        config.scale *= config.lacunarity;
    }
    return value / max_value * amplitude;
}

daxa_f32 voronoi_noise(daxa_f32vec3 pos) {
    daxa_f32 value = 1e38;

    for (daxa_i32 zi = 0; zi < 3; ++zi) {
        for (daxa_i32 yi = 0; yi < 3; ++yi) {
            for (daxa_i32 xi = 0; xi < 3; ++xi) {
                daxa_f32vec3 p = pos + daxa_f32vec3(xi - 1, yi - 1, zi - 1);
                p = floor(p) + 0.5;
                p += daxa_f32vec3(rand(), rand(), rand());
                value = min(value, dot(pos - p, pos - p));
            }
        }
    }

    return value;
}

daxa_f32 fractal_voronoi_noise(daxa_f32vec3 pos, FractalNoiseConfig config) {
    daxa_f32 value = 0.0;
    daxa_f32 max_value = 0.0;
    daxa_f32 amplitude = config.amplitude;
    daxa_f32mat3x3 rot_mat = daxa_f32mat3x3(
        0.2184223, -0.5347182, 0.8163137,
        0.9079879, -0.1951438, -0.3707788,
        0.3575608, 0.8221893, 0.4428939);
    for (daxa_u32 i = 0; i < config.octaves; ++i) {
        pos = (rot_mat * pos) + daxa_f32vec3(71.444, 25.170, -54.766);
        daxa_f32vec3 p = pos * config.scale;
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
daxa_f32 fbm2(vec2 uv) {
    daxa_f32 f = 0;
    mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
    f = 0.5000 * value_noise(uv);
    uv = m * uv;
    f += 0.2500 * value_noise(uv);
    uv = m * uv;
    return f * .5 + .5;
}
