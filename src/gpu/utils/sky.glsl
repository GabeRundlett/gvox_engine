#pragma once

#include <shared/core.inl>

#define SKY_COL (f32vec3(0.02, 0.05, 0.90) * 4)
#define SKY_COL_B (f32vec3(0.11, 0.10, 0.54))

// #define SUN_TIME (deref(gpu_input).time)
#define SUN_TIME 0.9
#define SUN_COL (f32vec3(1, 0.90, 0.4) * 5)
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
