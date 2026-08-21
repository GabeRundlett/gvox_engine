#pragma once

#include <renderer/kajiya/inc/math_const.glsl>
#include <glm/glm.hpp>
#include <daxa/daxa.hpp>

constexpr auto ceil_log2(uint32_t x) -> uint32_t {
    constexpr uint32_t t[5] = {
        0xFFFF0000u,
        0x0000FF00u,
        0x000000F0u,
        0x0000000Cu,
        0x00000002u};

    uint32_t y = (((x & (x - 1)) == 0) ? 0 : 1);
    int j = 16;

    for (uint32_t const i : t) {
        int const k = (((x & i) == 0) ? 0 : j);
        y += static_cast<uint32_t>(k);
        x >>= k;
        j >>= 1;
    }

    return y;
}

float dot(daxa_f32vec3 a, daxa_f32vec3 b);
float length(daxa_f32vec3 v);
daxa_f32vec3 normalize(daxa_f32vec3 v);
daxa_f32vec3 sign(daxa_f32vec3 v);

daxa_f32vec3 operator+(daxa_f32vec3 a, daxa_f32vec3 b);
daxa_i32vec3 operator+(daxa_i32vec3 a, daxa_i32vec3 b);
daxa_f32vec3 operator-(daxa_f32vec3 a, daxa_f32vec3 b);
daxa_f32vec3 operator*(daxa_f32vec3 a, daxa_f32vec3 b);
daxa_f32vec3 operator*(daxa_f32vec3 a, float b);

glm::mat4 rotation_matrix(float yaw, float pitch, float roll);
glm::mat4 inv_rotation_matrix(float yaw, float pitch, float roll);
glm::mat4 translation_matrix(daxa_f32vec3 pos);
daxa_f32vec3 apply_inv_rotation(daxa_f32vec3 pt, daxa_f32vec3 ypr);

inline constexpr auto round_up_div(auto x, auto y) {
    return (x + y - 1) / y;
}

inline constexpr auto find_msb(uint32_t v) -> uint32_t {
    uint32_t index = 0;
    while (v != 0) {
        v = v >> 1;
        index = index + 1;
    }
    return index;
}
inline constexpr auto find_next_lower_po2(uint32_t v) -> uint32_t {
    auto const msb = find_msb(v);
    return 1u << ((msb == 0 ? 1 : msb) - 1);
}

inline auto get_aligned(daxa_u64 operand, daxa_u64 granularity) -> daxa_u64 {
    return ((operand + (granularity - 1)) & ~(granularity - 1));
}

