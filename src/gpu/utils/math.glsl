#pragma once

#include <utils/defs.glsl>
#include <utils/math_const.glsl>

// Definitions

#define MAX_STEPS 512
const float MAX_DIST = 1.0e9;

// Objects

struct Sphere {
    daxa_f32vec3 o;
    daxa_f32 r;
};
struct BoundingBox {
    daxa_f32vec3 bound_min, bound_max;
};
struct CapsulePoints {
    daxa_f32vec3 p0, p1;
    daxa_f32 r;
};
struct Aabb {
    vec3 pmin;
    vec3 pmax;
};

// Common Functions

#define select(cond, a, b) ((cond) ? (a) : (b))
#define rcp(x) (1.0 / x)

float square(float x) { return x * x; }
vec2 square(vec2 x) { return x * x; }
vec3 square(vec3 x) { return x * x; }
vec4 square(vec4 x) { return x * x; }

float length_squared(vec2 v) { return dot(v, v); }
float length_squared(vec3 v) { return dot(v, v); }
float length_squared(vec4 v) { return dot(v, v); }

daxa_f32 saturate(daxa_f32 x) { return clamp(x, 0.0, 1.0); }
daxa_f32vec2 saturate(daxa_f32vec2 x) { return clamp(x, daxa_f32vec2(0.0), daxa_f32vec2(1.0)); }
daxa_f32vec3 saturate(daxa_f32vec3 x) { return clamp(x, daxa_f32vec3(0.0), daxa_f32vec3(1.0)); }
daxa_f32vec4 saturate(daxa_f32vec4 x) { return clamp(x, daxa_f32vec4(0.0), daxa_f32vec4(1.0)); }
daxa_f32 nonzero_sign(daxa_f32 x) {
    if (x < 0.0)
        return -1.0;
    return 1.0;
}
daxa_f32vec2 nonzero_sign(daxa_f32vec2 x) {
    return daxa_f32vec2(nonzero_sign(x.x), nonzero_sign(x.y));
}
daxa_f32vec3 nonzero_sign(daxa_f32vec3 x) {
    return daxa_f32vec3(nonzero_sign(x.x), nonzero_sign(x.y), nonzero_sign(x.z));
}
daxa_f32vec4 nonzero_sign(daxa_f32vec4 x) {
    return daxa_f32vec4(nonzero_sign(x.x), nonzero_sign(x.y), nonzero_sign(x.z), nonzero_sign(x.w));
}
daxa_f32 deg2rad(daxa_f32 d) {
    return d * M_PI / 180.0;
}
daxa_f32 rad2deg(daxa_f32 r) {
    return r * 180.0 / M_PI;
}
daxa_f32vec3 rotate_x(daxa_f32vec3 v, daxa_f32 angle) {
    float sin_rot_x = sin(angle), cos_rot_x = cos(angle);
    daxa_f32mat3x3 rot_mat = daxa_f32mat3x3(
        1, 0, 0,
        0, cos_rot_x, sin_rot_x,
        0, -sin_rot_x, cos_rot_x);
    return rot_mat * v;
}
daxa_f32vec3 rotate_y(daxa_f32vec3 v, daxa_f32 angle) {
    float sin_rot_y = sin(angle), cos_rot_y = cos(angle);
    daxa_f32mat3x3 rot_mat = daxa_f32mat3x3(
        cos_rot_y, 0, sin_rot_y,
        0, 1, 0,
        -sin_rot_y, 0, cos_rot_y);
    return rot_mat * v;
}
daxa_f32vec3 rotate_z(daxa_f32vec3 v, daxa_f32 angle) {
    float sin_rot_z = sin(angle), cos_rot_z = cos(angle);
    daxa_f32mat3x3 rot_mat = daxa_f32mat3x3(
        cos_rot_z, -sin_rot_z, 0,
        sin_rot_z, cos_rot_z, 0,
        0, 0, 1);
    return rot_mat * v;
}

// Color functions
daxa_f32vec3 rgb2hsv(daxa_f32vec3 c) {
    daxa_f32vec4 K = daxa_f32vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    daxa_f32vec4 p = mix(daxa_f32vec4(c.bg, K.wz), daxa_f32vec4(c.gb, K.xy), step(c.b, c.g));
    daxa_f32vec4 q = mix(daxa_f32vec4(p.xyw, c.r), daxa_f32vec4(c.r, p.yzx), step(p.x, c.r));
    daxa_f32 d = q.x - min(q.w, q.y);
    daxa_f32 e = 1.0e-10;
    return daxa_f32vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
daxa_f32vec3 hsv2rgb(daxa_f32vec3 c) {
    daxa_f32vec4 k = daxa_f32vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    daxa_f32vec3 p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * mix(k.xxx, clamp(p - k.xxx, 0.0, 1.0), c.y);
}
daxa_f32vec4 uint_rgba8_to_f32vec4(daxa_u32 u) {
    daxa_f32vec4 result;
    result.r = daxa_f32((u >> 0x00) & 0xff) / 255.0;
    result.g = daxa_f32((u >> 0x08) & 0xff) / 255.0;
    result.b = daxa_f32((u >> 0x10) & 0xff) / 255.0;
    result.a = daxa_f32((u >> 0x18) & 0xff) / 255.0;

    result = pow(result, daxa_f32vec4(2.2));
    return result;
}
daxa_u32 daxa_f32vec4_to_uint_rgba8(daxa_f32vec4 f) {
    f = clamp(f, daxa_f32vec4(0), daxa_f32vec4(1));
    f = pow(f, daxa_f32vec4(1.0 / 2.2));

    daxa_u32 result = 0;
    result |= daxa_u32(clamp(f.r, 0, 1) * 255) << 0x00;
    result |= daxa_u32(clamp(f.g, 0, 1) * 255) << 0x08;
    result |= daxa_u32(clamp(f.b, 0, 1) * 255) << 0x10;
    result |= daxa_u32(clamp(f.a, 0, 1) * 255) << 0x18;
    return result;
}

float uint_to_u01_float(uint h) {
    const uint mantissaMask = 0x007FFFFFu;
    const uint one = 0x3F800000u;

    h &= mantissaMask;
    h |= one;

    float r2 = uintBitsToFloat(h);
    return r2 - 1.0;
}

// [Drobot2014a] Low Level Optimizations for GCN
float fast_sqrt(float x) {
    return uintBitsToFloat(0x1fbd1df5 + (floatBitsToUint(x) >> 1u));
}

// [Eberly2014] GPGPU Programming for Games and Science
float fast_acos(float inX) {
    float x = abs(inX);
    float res = -0.156583f * x + M_PI * 0.5;
    res *= fast_sqrt(1.0f - x);
    return (inX >= 0) ? res : (M_PI - res);
}

float inverse_depth_relative_diff(float primary_depth, float secondary_depth) {
    return abs(max(1e-20, primary_depth) / max(1e-20, secondary_depth) - 1.0);
}

float exponential_squish(float len, float squish_scale) {
    return exp2(-clamp(squish_scale * len, 0, 100));
}

float exponential_unsquish(float len, float squish_scale) {
    return max(0.0, -1.0 / squish_scale * log2(1e-30 + len));
}

#define URGB9E5_CONCENTRATION 4.0
#define URGB9E5_MIN_EXPONENT -8.0
daxa_f32 urgb9e5_scale_exp_inv(daxa_f32 x) { return (exp((x + URGB9E5_MIN_EXPONENT) / URGB9E5_CONCENTRATION)); }
daxa_f32vec3 uint_urgb9e5_to_f32vec3(daxa_u32 u) {
    daxa_f32vec3 result;
    result.r = daxa_f32((u >> 0x00) & 0x1ff);
    result.g = daxa_f32((u >> 0x09) & 0x1ff);
    result.b = daxa_f32((u >> 0x12) & 0x1ff);
    daxa_f32 scale = urgb9e5_scale_exp_inv((u >> 0x1b) & 0x1f) / 511.0;
    return result * scale;
}
daxa_f32 urgb9e5_scale_exp(daxa_f32 x) { return URGB9E5_CONCENTRATION * log(x) - URGB9E5_MIN_EXPONENT; }
daxa_u32 daxa_f32vec3_to_uint_urgb9e5(daxa_f32vec3 f) {
    daxa_f32 scale = max(max(f.x, 0.0), max(f.y, f.z));
    daxa_f32 exponent = ceil(clamp(urgb9e5_scale_exp(scale), 0, 31));
    daxa_f32 fac = 511.0 / urgb9e5_scale_exp_inv(exponent);
    daxa_u32 result = 0;
    result |= daxa_u32(clamp(f.r * fac, 0.0, 511.0)) << 0x00;
    result |= daxa_u32(clamp(f.g * fac, 0.0, 511.0)) << 0x09;
    result |= daxa_u32(clamp(f.b * fac, 0.0, 511.0)) << 0x12;
    result |= daxa_u32(exponent) << 0x1b;
    return result;
}

#include <utils/normal.glsl>

daxa_u32 ceil_log2(daxa_u32 x) {
    return findMSB(x) + daxa_u32(bitCount(x) > 1);
}

// Bit Functions

void flag_set(in out daxa_u32 bitfield, daxa_u32 index, bool value) {
    if (value) {
        bitfield |= 1u << index;
    } else {
        bitfield &= ~(1u << index);
    }
}
bool flag_get(daxa_u32 bitfield, daxa_u32 index) {
    return ((bitfield >> index) & 1u) == 1u;
}

// Shape Functions

daxa_b32 inside(daxa_f32vec3 p, Sphere s) {
    return dot(p - s.o, p - s.o) < s.r * s.r;
}
daxa_b32 inside(daxa_f32vec3 p, BoundingBox b) {
    return (p.x >= b.bound_min.x && p.x < b.bound_max.x &&
            p.y >= b.bound_min.y && p.y < b.bound_max.y &&
            p.z >= b.bound_min.z && p.z < b.bound_max.z);
}
daxa_b32 overlaps(BoundingBox a, BoundingBox b) {
    daxa_b32 x_overlap = a.bound_max.x >= b.bound_min.x && b.bound_max.x >= a.bound_min.x;
    daxa_b32 y_overlap = a.bound_max.y >= b.bound_min.y && b.bound_max.y >= a.bound_min.y;
    daxa_b32 z_overlap = a.bound_max.z >= b.bound_min.z && b.bound_max.z >= a.bound_min.z;
    return x_overlap && y_overlap && z_overlap;
}
void intersect(in out daxa_f32vec3 ray_pos, daxa_f32vec3 ray_dir, daxa_f32vec3 inv_dir, BoundingBox b) {
    if (inside(ray_pos, b)) {
        return;
    }
    daxa_f32 tx1 = (b.bound_min.x - ray_pos.x) * inv_dir.x;
    daxa_f32 tx2 = (b.bound_max.x - ray_pos.x) * inv_dir.x;
    daxa_f32 tmin = min(tx1, tx2);
    daxa_f32 tmax = max(tx1, tx2);
    daxa_f32 ty1 = (b.bound_min.y - ray_pos.y) * inv_dir.y;
    daxa_f32 ty2 = (b.bound_max.y - ray_pos.y) * inv_dir.y;
    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));
    daxa_f32 tz1 = (b.bound_min.z - ray_pos.z) * inv_dir.z;
    daxa_f32 tz2 = (b.bound_max.z - ray_pos.z) * inv_dir.z;
    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));

    // daxa_f32 dist = max(min(tmax, tmin), 0);
    daxa_f32 dist = MAX_DIST;
    if (tmax >= tmin) {
        if (tmin > 0) {
            dist = tmin;
        }
    }

    ray_pos = ray_pos + ray_dir * dist;
}

#include <utils/signed_distance.glsl>
#include <utils/random.glsl>

daxa_f32mat3x3 tbn_from_normal(daxa_f32vec3 nrm) {
    daxa_f32vec3 tangent = normalize(cross(nrm, -nrm.zxy));
    daxa_f32vec3 bi_tangent = cross(nrm, tangent);
    return daxa_f32mat3x3(tangent, bi_tangent, nrm);
}

// Building an Orthonormal Basis, Revisited
// http://jcgt.org/published/0006/01/01/
daxa_f32mat3x3 build_orthonormal_basis(vec3 n) {
    vec3 b1;
    vec3 b2;

    if (n.z < 0.0) {
        const float a = 1.0 / (1.0 - n.z);
        const float b = n.x * n.y * a;
        b1 = vec3(1.0 - n.x * n.x * a, -b, n.x);
        b2 = vec3(b, n.y * n.y * a - 1.0, -n.y);
    } else {
        const float a = 1.0 / (1.0 + n.z);
        const float b = -n.x * n.y * a;
        b1 = vec3(1.0 - n.x * n.x * a, b, -n.x);
        b2 = vec3(b, 1.0 - n.y * n.y * a, -n.y);
    }

    return daxa_f32mat3x3(b1, b2, n);
}

vec3 uniform_sample_cone(vec2 urand, float cos_theta_max) {
    float cos_theta = (1.0 - urand.x) + urand.x * cos_theta_max;
    float sin_theta = sqrt(clamp(1.0 - cos_theta * cos_theta, 0.0, 1.0));
    float phi = urand.y * (M_PI * 2.0);
    return vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

daxa_f32vec3 uniform_sample_hemisphere(daxa_f32vec2 urand) {
    float phi = urand.y * 2.0 * M_PI;
    float cos_theta = 1.0 - urand.x;
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    return daxa_f32vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

float rtr_encode_cos_theta_for_fp16(float x) {
    return 1 - x;
}

float rtr_decode_cos_theta_from_fp16(float x) {
    return 1 - x;
}

vec2 rsi(vec3 r0, vec3 rd, float sr) {
    // ray-sphere intersection that assumes
    // the sphere is centered at the origin.
    // No intersection when result.x > result.y
    float a = dot(rd, rd);
    float b = 2.0 * dot(rd, r0);
    float c = dot(r0, r0) - (sr * sr);
    float d = (b * b) - 4.0 * a * c;
    if (d < 0.0)
        return vec2(1e5, -1e5);
    return vec2(
        (-b - sqrt(d)) / (2.0 * a),
        (-b + sqrt(d)) / (2.0 * a));
}

vec3 atmosphere(vec3 r, vec3 r0, vec3 pSun, float iSun, float rPlanet, float rAtmos, vec3 kRlh, float kMie, float shRlh, float shMie, float g) {
    // return daxa_f32vec3(0, 0, 0);
    const int iSteps = 16;
    const int jSteps = 2;
    // Normalize the sun and view directions.
    pSun = normalize(pSun);
    r = normalize(r);

    // Calculate the step size of the primary ray.
    vec2 p = rsi(r0, r, rAtmos);
    if (p.x > p.y)
        return vec3(0, 0, 0);
    p.y = min(p.y, rsi(r0, r, rPlanet).x);
    float iStepSize = (p.y - p.x) / float(iSteps);

    // Initialize the primary ray time.
    float iTime = 0.0;

    // Initialize accumulators for Rayleigh and Mie scattering.
    vec3 totalRlh = vec3(0, 0, 0);
    vec3 totalMie = vec3(0, 0, 0);

    // Initialize optical depth accumulators for the primary ray.
    float iOdRlh = 0.0;
    float iOdMie = 0.0;

    // Calculate the Rayleigh and Mie phases.
    float mu = dot(r, pSun);
    float mumu = mu * mu;
    float gg = g * g;
    float pRlh = 3.0 / (16.0 * M_PI) * (1.0 + mumu);
    float pMie = 3.0 / (8.0 * M_PI) * ((1.0 - gg) * (mumu + 1.0)) / (pow(1.0 + gg - 2.0 * mu * g, 1.5) * (2.0 + gg));

    // Sample the primary ray.
    for (int i = 0; i < iSteps; i++) {

        // Calculate the primary ray sample position.
        vec3 iPos = r0 + r * (iTime + iStepSize * 0.5);

        // Calculate the height of the sample.
        float iHeight = length(iPos) - rPlanet;

        // Calculate the optical depth of the Rayleigh and Mie scattering for this step.
        float odStepRlh = exp(-iHeight / shRlh) * iStepSize;
        float odStepMie = exp(-iHeight / shMie) * iStepSize;

        // Accumulate optical depth.
        iOdRlh += odStepRlh;
        iOdMie += odStepMie;

        // Calculate the step size of the secondary ray.
        float jStepSize = rsi(iPos, pSun, rAtmos).y / float(jSteps);

        // Initialize the secondary ray time.
        float jTime = 0.0;

        // Initialize optical depth accumulators for the secondary ray.
        float jOdRlh = 0.0;
        float jOdMie = 0.0;

        // Sample the secondary ray.
        for (int j = 0; j < jSteps; j++) {

            // Calculate the secondary ray sample position.
            vec3 jPos = iPos + pSun * (jTime + jStepSize * 0.5);

            // Calculate the height of the sample.
            float jHeight = length(jPos) - rPlanet;

            // Accumulate the optical depth.
            jOdRlh += exp(-jHeight / shRlh) * jStepSize;
            jOdMie += exp(-jHeight / shMie) * jStepSize;

            // Increment the secondary ray time.
            jTime += jStepSize;
        }

        // Calculate attenuation.
        vec3 attn = exp(-(kMie * (iOdMie + jOdMie) + kRlh * (iOdRlh + jOdRlh)));

        // Accumulate scattering.
        totalRlh += odStepRlh * attn;
        totalMie += odStepMie * attn;

        // Increment the primary ray time.
        iTime += iStepSize;
    }

    // Calculate and return the final color.
    return iSun * (pRlh * kRlh * totalRlh + pMie * kMie * totalMie);
}

bool rectangles_overlap(vec3 a_min, vec3 a_max, vec3 b_min, vec3 b_max) {
    bool x_disjoint = (a_max.x < b_min.x) || (b_max.x < a_min.x);
    bool y_disjoint = (a_max.y < b_min.y) || (b_max.y < a_min.y);
    bool z_disjoint = (a_max.z < b_min.z) || (b_max.z < a_min.z);
    return !x_disjoint && !y_disjoint && !z_disjoint;
}

// https://www.shadertoy.com/view/cdSBRG
daxa_i32 imod(daxa_i32 x, daxa_i32 m) {
    return x >= 0 ? x % m : m - 1 - (-x - 1) % m;
}
daxa_i32vec3 imod3(daxa_i32vec3 p, daxa_i32 m) {
    return daxa_i32vec3(imod(p.x, m), imod(p.y, m), imod(p.z, m));
}
daxa_i32vec3 imod3(daxa_i32vec3 p, daxa_i32vec3 m) {
    return daxa_i32vec3(imod(p.x, m.x), imod(p.y, m.y), imod(p.z, m.z));
}

daxa_f32mat4x4 rotation_matrix(daxa_f32 yaw, daxa_f32 pitch, daxa_f32 roll) {
    float sin_rot_x = sin(pitch), cos_rot_x = cos(pitch);
    float sin_rot_y = sin(roll), cos_rot_y = cos(roll);
    float sin_rot_z = sin(yaw), cos_rot_z = cos(yaw);
    return daxa_f32mat4x4(
               cos_rot_z, -sin_rot_z, 0, 0,
               sin_rot_z, cos_rot_z, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1) *
           daxa_f32mat4x4(
               1, 0, 0, 0,
               0, cos_rot_x, sin_rot_x, 0,
               0, -sin_rot_x, cos_rot_x, 0,
               0, 0, 0, 1) *
           daxa_f32mat4x4(
               cos_rot_y, -sin_rot_y, 0, 0,
               sin_rot_y, cos_rot_y, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1);
}
daxa_f32mat4x4 inv_rotation_matrix(daxa_f32 yaw, daxa_f32 pitch, daxa_f32 roll) {
    float sin_rot_x = sin(-pitch), cos_rot_x = cos(-pitch);
    float sin_rot_y = sin(-roll), cos_rot_y = cos(-roll);
    float sin_rot_z = sin(-yaw), cos_rot_z = cos(-yaw);
    return daxa_f32mat4x4(
               cos_rot_y, -sin_rot_y, 0, 0,
               sin_rot_y, cos_rot_y, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1) *
           daxa_f32mat4x4(
               1, 0, 0, 0,
               0, cos_rot_x, sin_rot_x, 0,
               0, -sin_rot_x, cos_rot_x, 0,
               0, 0, 0, 1) *
           daxa_f32mat4x4(
               cos_rot_z, -sin_rot_z, 0, 0,
               sin_rot_z, cos_rot_z, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1);
}
daxa_f32mat4x4 translation_matrix(daxa_f32vec3 pos) {
    return daxa_f32mat4x4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        pos, 1);
}

vec3 apply_inv_rotation(vec3 pt, vec3 ypr) {
    float yaw = ypr[0];
    float pitch = ypr[1];
    float roll = ypr[2];
    return (inv_rotation_matrix(ypr[0], ypr[1], ypr[2]) * vec4(pt, 0.0)).xyz;
}

daxa_f32vec3 position_world_to_clip(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec3 v) {
    daxa_f32vec4 p = (deref(globals).player.cam.world_to_view * daxa_f32vec4(v, 1));
    p = (deref(globals).player.cam.view_to_clip * p);
    return p.xyz / p.w;
}

daxa_f32vec2 get_uv(daxa_i32vec2 pix, daxa_f32vec4 tex_size) { return (daxa_f32vec2(pix) + 0.5) * tex_size.zw; }
daxa_f32vec2 get_uv(daxa_f32vec2 pix, daxa_f32vec4 tex_size) { return (pix + 0.5) * tex_size.zw; }
daxa_f32vec2 cs_to_uv(daxa_f32vec2 cs) { return cs * daxa_f32vec2(0.5, -0.5) + daxa_f32vec2(0.5, 0.5); }
daxa_f32vec2 uv_to_cs(daxa_f32vec2 uv) { return (uv - 0.5) * daxa_f32vec2(2, -2); }
daxa_f32vec2 uv_to_ss(daxa_BufferPtr(GpuInput) gpu_input, daxa_f32vec2 uv, daxa_f32vec4 tex_size) { return uv - deref(gpu_input).halton_jitter.xy * tex_size.zw * 1.0; }
daxa_f32vec2 ss_to_uv(daxa_BufferPtr(GpuInput) gpu_input, daxa_f32vec2 ss, daxa_f32vec4 tex_size) { return ss + deref(gpu_input).halton_jitter.xy * tex_size.zw * 1.0; }

struct ViewRayContext {
    daxa_f32vec4 ray_dir_cs;
    daxa_f32vec4 ray_dir_vs_h;
    daxa_f32vec4 ray_dir_ws_h;
    daxa_f32vec4 ray_origin_cs;
    daxa_f32vec4 ray_origin_vs_h;
    daxa_f32vec4 ray_origin_ws_h;
    daxa_f32vec4 ray_hit_cs;
    daxa_f32vec4 ray_hit_vs_h;
    daxa_f32vec4 ray_hit_ws_h;
};

daxa_f32vec3 ray_dir_vs(in ViewRayContext vrc) { return normalize(vrc.ray_dir_vs_h.xyz); }
daxa_f32vec3 ray_dir_ws(in ViewRayContext vrc) { return normalize(vrc.ray_dir_ws_h.xyz); }
daxa_f32vec3 ray_origin_vs(in ViewRayContext vrc) { return vrc.ray_origin_vs_h.xyz / vrc.ray_origin_vs_h.w; }
daxa_f32vec3 ray_origin_ws(in ViewRayContext vrc) { return vrc.ray_origin_ws_h.xyz / vrc.ray_origin_ws_h.w; }
daxa_f32vec3 ray_hit_vs(in ViewRayContext vrc) { return vrc.ray_hit_vs_h.xyz / vrc.ray_hit_vs_h.w; }
daxa_f32vec3 ray_hit_ws(in ViewRayContext vrc) { return vrc.ray_hit_ws_h.xyz / vrc.ray_hit_ws_h.w; }
daxa_f32vec3 biased_secondary_ray_origin_ws(in ViewRayContext vrc) {
    return ray_hit_ws(vrc) - ray_dir_ws(vrc) * (length(ray_hit_vs(vrc)) + length(ray_hit_ws(vrc))) * 1e-4;
}
daxa_f32vec3 biased_secondary_ray_origin_ws_with_normal(in ViewRayContext vrc, daxa_f32vec3 normal) {
    daxa_f32vec3 ws_abs = abs(ray_hit_ws(vrc));
    float max_comp = max(max(ws_abs.x, ws_abs.y), max(ws_abs.z, -ray_hit_vs(vrc).z));
    return ray_hit_ws(vrc) + (normal - ray_dir_ws(vrc)) * max(1e-4, max_comp * 1e-6);
}
ViewRayContext vrc_from_uv(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec2 uv) {
    ViewRayContext res;
    res.ray_dir_cs = daxa_f32vec4(uv_to_cs(uv), 0.0, 1.0);
    res.ray_dir_vs_h = deref(globals).player.cam.sample_to_view * res.ray_dir_cs;
    res.ray_dir_ws_h = deref(globals).player.cam.view_to_world * res.ray_dir_vs_h;
    res.ray_origin_cs = daxa_f32vec4(uv_to_cs(uv), 1.0, 1.0);
    res.ray_origin_vs_h = deref(globals).player.cam.sample_to_view * res.ray_origin_cs;
    res.ray_origin_ws_h = deref(globals).player.cam.view_to_world * res.ray_origin_vs_h;
    return res;
}
ViewRayContext unjittered_vrc_from_uv(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec2 uv) {
    ViewRayContext res;
    res.ray_dir_cs = daxa_f32vec4(uv_to_cs(uv), 0.0, 1.0);
    res.ray_dir_vs_h = deref(globals).player.cam.clip_to_view * res.ray_dir_cs;
    res.ray_dir_ws_h = deref(globals).player.cam.view_to_world * res.ray_dir_vs_h;
    res.ray_origin_cs = daxa_f32vec4(uv_to_cs(uv), 1.0, 1.0);
    res.ray_origin_vs_h = deref(globals).player.cam.clip_to_view * res.ray_origin_cs;
    res.ray_origin_ws_h = deref(globals).player.cam.view_to_world * res.ray_origin_vs_h;
    return res;
}
ViewRayContext vrc_from_uv_and_depth(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec2 uv, float depth) {
    ViewRayContext res;
    res.ray_dir_cs = daxa_f32vec4(uv_to_cs(uv), 0.0, 1.0);
    res.ray_dir_vs_h = deref(globals).player.cam.sample_to_view * res.ray_dir_cs;
    res.ray_dir_ws_h = deref(globals).player.cam.view_to_world * res.ray_dir_vs_h;
    res.ray_origin_cs = daxa_f32vec4(uv_to_cs(uv), 1.0, 1.0);
    res.ray_origin_vs_h = deref(globals).player.cam.sample_to_view * res.ray_origin_cs;
    res.ray_origin_ws_h = deref(globals).player.cam.view_to_world * res.ray_origin_vs_h;
    res.ray_hit_cs = daxa_f32vec4(uv_to_cs(uv), depth, 1.0);
    res.ray_hit_vs_h = deref(globals).player.cam.sample_to_view * res.ray_hit_cs;
    res.ray_hit_ws_h = deref(globals).player.cam.view_to_world * res.ray_hit_vs_h;
    return res;
}
#define BIAS uintBitsToFloat(0x3f800040) // uintBitsToFloat(0x3f800040) == 1.00000762939453125
ViewRayContext vrc_from_uv_and_biased_depth(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec2 uv, float depth) {
    return vrc_from_uv_and_depth(globals, uv, min(1.0, depth * BIAS));
}

daxa_f32vec3 direction_view_to_world(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec3 v) {
    return (deref(globals).player.cam.view_to_world * daxa_f32vec4(v, 0)).xyz;
}
daxa_f32vec3 direction_world_to_view(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec3 v) {
    return (deref(globals).player.cam.world_to_view * daxa_f32vec4(v, 0)).xyz;
}
daxa_f32vec3 position_world_to_view(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec3 v) {
    return (deref(globals).player.cam.world_to_view, daxa_f32vec4(v, 1)).xyz;
}
daxa_f32vec3 position_view_to_world(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec3 v) {
    return (deref(globals).player.cam.view_to_world, daxa_f32vec4(v, 1)).xyz;
}

daxa_f32vec3 position_world_to_sample(daxa_RWBufferPtr(GpuGlobals) globals, daxa_f32vec3 v) {
    daxa_f32vec4 p = deref(globals).player.cam.world_to_view * daxa_f32vec4(v, 1);
    p = deref(globals).player.cam.view_to_sample * p;
    return p.xyz / p.w;
}

struct Bilinear {
    daxa_f32vec2 origin;
    daxa_f32vec2 weights;
};

daxa_i32vec2 px0(in Bilinear b) { return daxa_i32vec2(b.origin); }
daxa_i32vec2 px1(in Bilinear b) { return daxa_i32vec2(b.origin) + daxa_i32vec2(1, 0); }
daxa_i32vec2 px2(in Bilinear b) { return daxa_i32vec2(b.origin) + daxa_i32vec2(0, 1); }
daxa_i32vec2 px3(in Bilinear b) { return daxa_i32vec2(b.origin) + daxa_i32vec2(1, 1); }

Bilinear get_bilinear_filter(daxa_f32vec2 uv, daxa_f32vec2 tex_size) {
    Bilinear result;
    result.origin = trunc(uv * tex_size - 0.5);
    result.weights = fract(uv * tex_size - 0.5);
    return result;
}

daxa_f32vec4 get_bilinear_custom_weights(Bilinear f, daxa_f32vec4 custom_weights) {
    daxa_f32vec4 weights;
    weights.x = (1.0 - f.weights.x) * (1.0 - f.weights.y);
    weights.y = f.weights.x * (1.0 - f.weights.y);
    weights.z = (1.0 - f.weights.x) * f.weights.y;
    weights.w = f.weights.x * f.weights.y;
    return weights * custom_weights;
}

daxa_f32vec4 apply_bilinear_custom_weights(daxa_f32vec4 s00, daxa_f32vec4 s10, daxa_f32vec4 s01, daxa_f32vec4 s11, daxa_f32vec4 w, bool should_normalize) {
    daxa_f32vec4 r = s00 * w.x + s10 * w.y + s01 * w.z + s11 * w.w;
    return r * (should_normalize ? (1.0 / dot(w, daxa_f32vec4(1.0))) : 1.0);
}

void apply_friction(daxa_BufferPtr(GpuInput) gpu_input, in out vec3 vel, vec3 friction_vec, float friction_coeff) {
    float fac = deref(gpu_input).delta_time * friction_coeff;
    vec3 new_vel = vel - normalize(friction_vec) * fac;

    if (dot(vel, new_vel) > 0.000) {
        vel = new_vel;
    } else {
        vel -= friction_vec;
    }
}

mat3 CUBE_MAP_FACE_ROTATION(uint face) {
    switch (face) {
    case 0: return mat3(+0, +0, -1, +0, -1, +0, -1, +0, +0);
    case 1: return mat3(+0, +0, +1, +0, -1, +0, +1, +0, +0);
    case 2: return mat3(+1, +0, +0, +0, +0, +1, +0, -1, +0);
    case 3: return mat3(+1, +0, +0, +0, +0, -1, +0, +1, +0);
    case 4: return mat3(+1, +0, +0, +0, -1, +0, +0, +0, -1);
    default: return mat3(-1, +0, +0, +0, -1, +0, +0, +0, +1);
    }
}
