#pragma once

vec3 sRGB_to_YCbCr(vec3 col) {
    return mat3(0.2126, -0.1146, 0.5, 0.7152, -0.3854, -0.4542, 0.0722, 0.5, -0.0458) * col;
}

vec3 YCbCr_to_sRGB(vec3 col) {
    return max(vec3(0.0), mat3(1.0, 1.0, 1.0, 0.0, -0.1873, 1.8556, 1.5748, -.4681, 0.0) * col);
}
