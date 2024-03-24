#include "math.hpp"

float dot(daxa_f32vec3 a, daxa_f32vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float length(daxa_f32vec3 v) {
    return sqrt(dot(v, v));
}

daxa_f32vec3 normalize(daxa_f32vec3 v) {
    float len = length(v);
    v.x /= len;
    v.y /= len;
    v.z /= len;
    return v;
}

daxa_f32vec3 operator+(daxa_f32vec3 a, daxa_f32vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
daxa_i32vec3 operator+(daxa_i32vec3 a, daxa_i32vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
daxa_f32vec3 operator-(daxa_f32vec3 a, daxa_f32vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
daxa_f32vec3 operator*(daxa_f32vec3 a, daxa_f32vec3 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}
daxa_f32vec3 operator*(daxa_f32vec3 a, float b) {
    return {a.x * b, a.y * b, a.z * b};
}

glm::mat4 rotation_matrix(float yaw, float pitch, float roll) {
    float sin_rot_x = sin(pitch), cos_rot_x = cos(pitch);
    float sin_rot_y = sin(roll), cos_rot_y = cos(roll);
    float sin_rot_z = sin(yaw), cos_rot_z = cos(yaw);
    return glm::mat4(
               cos_rot_z, -sin_rot_z, 0, 0,
               sin_rot_z, cos_rot_z, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1) *
           glm::mat4(
               1, 0, 0, 0,
               0, cos_rot_x, sin_rot_x, 0,
               0, -sin_rot_x, cos_rot_x, 0,
               0, 0, 0, 1) *
           glm::mat4(
               cos_rot_y, -sin_rot_y, 0, 0,
               sin_rot_y, cos_rot_y, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1);
}
glm::mat4 inv_rotation_matrix(float yaw, float pitch, float roll) {
    float sin_rot_x = sin(-pitch), cos_rot_x = cos(-pitch);
    float sin_rot_y = sin(-roll), cos_rot_y = cos(-roll);
    float sin_rot_z = sin(-yaw), cos_rot_z = cos(-yaw);
    return glm::mat4(
               cos_rot_y, -sin_rot_y, 0, 0,
               sin_rot_y, cos_rot_y, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1) *
           glm::mat4(
               1, 0, 0, 0,
               0, cos_rot_x, sin_rot_x, 0,
               0, -sin_rot_x, cos_rot_x, 0,
               0, 0, 0, 1) *
           glm::mat4(
               cos_rot_z, -sin_rot_z, 0, 0,
               sin_rot_z, cos_rot_z, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1);
}
glm::mat4 translation_matrix(daxa_f32vec3 pos) {
    return glm::mat4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        pos.x, pos.y, pos.z, 1);
}

daxa_f32vec3 apply_inv_rotation(daxa_f32vec3 pt, daxa_f32vec3 ypr) {
    float yaw = ypr.x;
    float pitch = ypr.y;
    float roll = ypr.z;
    auto res = inv_rotation_matrix(yaw, pitch, roll) * glm::vec4(pt.x, pt.y, pt.z, 0.0);
    return {res.x, res.y, res.z};
}
