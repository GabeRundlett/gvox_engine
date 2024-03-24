#pragma once

#include <array>
#include <renderer/kajiya/inc/math_const.glsl>
#include <glm/glm.hpp>
#include <daxa/daxa.hpp>

constexpr auto ceil_log2(uint32_t x) -> uint32_t {
    constexpr auto const t = std::array<uint32_t, 5>{
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
