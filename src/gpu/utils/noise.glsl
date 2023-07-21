#pragma once

#include <utils/math.glsl>

f32vec4 noise(daxa_ImageViewId noise_texture, daxa_SamplerId noise_sampler, f32vec3 x) {
    const f32 offset = 1.0 / 512.0;
    f32vec4 gz0 = textureGather(daxa_sampler2DArray(noise_texture, noise_sampler), vec3(x.xy / 256.0, (int(floor(x.z)) + 0) & 0xff));
    f32vec4 gz1 = textureGather(daxa_sampler2DArray(noise_texture, noise_sampler), vec3(x.xy / 256.0, (int(floor(x.z)) + 1) & 0xff));
    x.xy = x.xy - 0.5 + offset;

    f32vec3 w = fract(x);
    f32vec3 u = w * w * (3.0 - 2.0 * w);
    f32vec3 du = 6.0 * w * (1.0 - w);

    // f32 z0 = mix(mix(gz0.w, gz0.z, w.x), mix(gz0.x, gz0.y, w.x), w.y);
    // f32 z1 = mix(mix(gz1.w, gz1.z, w.x), mix(gz1.x, gz1.y, w.x), w.y);
    // f32 dist = mix(z0, z1, w.z);
    // f32vec3 nrm = (z1 - z0) * du;

    f32 a = gz0.w, b = gz0.z, c = gz0.x, d = gz0.y;
    f32 e = gz1.w, f = gz1.z, g = gz1.x, h = gz1.y;

    f32 k0 = a, k1 = b - a, k2 = c - a, k3 = e - a;
    f32 k4 = a - b - c + d;
    f32 k5 = a - c - e + g;
    f32 k6 = a - b - e + f;
    f32 k7 = -a + b + c - d + e - f - g + h;
    f32 dist = k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z;
    f32vec3 nrm = du * (f32vec3(k1, k2, k3) + u.yzx * f32vec3(k4, k5, k6) + u.zxy * f32vec3(k6, k4, k5) + k7 * u.yzx * u.zxy);

    return f32vec4(dist, nrm);
}

struct FractalNoiseConfig {
    f32 amplitude;
    f32 persistance;
    f32 scale;
    f32 lacunarity;
    u32 octaves;
};

f32vec4 fractal_noise(daxa_ImageViewId noise_texture, daxa_SamplerId noise_sampler, f32vec3 pos, FractalNoiseConfig config) {
    const f32 scale = config.scale;
    f32 a = 0.0;
    f32 b = 0.5;
    f32 f = 1.0;
    f32vec3 d = f32vec3(0.0);
    for (u32 i = 0; i < config.octaves; ++i) {
        f32vec4 n = noise(noise_texture, noise_sampler, f * pos * scale);
        a += b * n.x;
        d += b * n.yzw * f * scale;
        b *= config.persistance;
        f *= config.lacunarity;
    }
    return f32vec4(a, d);
}

f32 analytical_noise_hash(in f32 n) {
    return fract(sin(n) * 753.5453123);
}

f32vec4 sd_analytical_noise(in f32vec3 x) {
    f32vec3 p = floor(x);
    f32vec3 w = fract(x);
    f32vec3 u = w * w * (3.0 - 2.0 * w);
    f32vec3 du = 6.0 * w * (1.0 - w);
    f32 n = p.x + p.y * 157.0 + 113.0 * p.z;
    f32 a = analytical_noise_hash(n + 0.0);
    f32 b = analytical_noise_hash(n + 1.0);
    f32 c = analytical_noise_hash(n + 157.0);
    f32 d = analytical_noise_hash(n + 158.0);
    f32 e = analytical_noise_hash(n + 113.0);
    f32 f = analytical_noise_hash(n + 114.0);
    f32 g = analytical_noise_hash(n + 270.0);
    f32 h = analytical_noise_hash(n + 271.0);
    f32 k0 = a;
    f32 k1 = b - a;
    f32 k2 = c - a;
    f32 k3 = e - a;
    f32 k4 = a - b - c + d;
    f32 k5 = a - c - e + g;
    f32 k6 = a - b - e + f;
    f32 k7 = -a + b + c - d + e - f - g + h;
    f32 dist = k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z;
    f32vec3 nrm = du * (f32vec3(k1, k2, k3) + u.yzx * f32vec3(k4, k5, k6) + u.zxy * f32vec3(k6, k4, k5) + k7 * u.yzx * u.zxy);
    return f32vec4(dist, nrm);
}

f32vec4 sd_analytical_fractal_noise(in f32vec3 x) {
    const f32 scale = 0.05;
    f32 a = 0.0;
    f32 b = 0.5;
    f32 f = 1.0;
    f32vec3 d = f32vec3(0.0);
    for (u32 i = 0; i < 4; ++i) {
        f32vec4 n = sd_analytical_noise(f * x * scale);
        a += b * n.x;
        d += b * n.yzw * f * scale;
        b *= 0.5;
        f *= 1.8;
    }
    return f32vec4(a, d);
}

f32 fractal_noise2(f32vec3 pos, FractalNoiseConfig config) {
    f32 value = 0.0;
    f32 max_value = 0.0;
    f32 amplitude = config.amplitude;
    f32mat3x3 rot_mat = f32mat3x3(
        0.2184223, -0.5347182, 0.8163137,
        0.9079879, -0.1951438, -0.3707788,
        0.3575608, 0.8221893, 0.4428939);
    for (u32 i = 0; i < config.octaves; ++i) {
        pos = (rot_mat * pos) + f32vec3(71.444, 25.170, -54.766);
        f32vec3 p = pos * config.scale;
        value += sd_analytical_noise(p).x * config.amplitude;
        max_value += config.amplitude;
        config.amplitude *= config.persistance;
        config.scale *= config.lacunarity;
    }
    return value / max_value * amplitude;
}

f32 voronoi_noise(f32vec3 pos) {
    f32 value = 1e38;

    for (i32 zi = 0; zi < 3; ++zi) {
        for (i32 yi = 0; yi < 3; ++yi) {
            for (i32 xi = 0; xi < 3; ++xi) {
                f32vec3 p = pos + f32vec3(xi - 1, yi - 1, zi - 1);
                p = floor(p) + 0.5;
                p += f32vec3(rand(), rand(), rand());
                value = min(value, dot(pos - p, pos - p));
            }
        }
    }

    return value;
}

f32 fractal_voronoi_noise(f32vec3 pos, FractalNoiseConfig config) {
    f32 value = 0.0;
    f32 max_value = 0.0;
    f32 amplitude = config.amplitude;
    f32mat3x3 rot_mat = f32mat3x3(
        0.2184223, -0.5347182, 0.8163137,
        0.9079879, -0.1951438, -0.3707788,
        0.3575608, 0.8221893, 0.4428939);
    for (u32 i = 0; i < config.octaves; ++i) {
        pos = (rot_mat * pos) + f32vec3(71.444, 25.170, -54.766);
        f32vec3 p = pos * config.scale;
        value += voronoi_noise(p) * config.amplitude;
        max_value += config.amplitude;
        config.amplitude *= config.persistance;
        config.scale *= config.lacunarity;
    }
    return value / max_value * amplitude;
}
