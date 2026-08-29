#pragma once

#include "defs.inl"

#if defined(__cplusplus)
#define uniform
#include <glm/glm.hpp>
using namespace glm;
#else
typedef float<2> vec2;
typedef float<3> vec3;
typedef int<3> ivec3;
typedef uint<3> uvec3;
typedef uint<2> uvec2;

struct mat3 {
    float data[3 * 3];
};

static inline uniform float get(uniform mat3 &m, uniform int i, uniform int j) {
    return m.data[i * 3 + j];
}

static inline float get(mat3 &m, uniform int i, uniform int j) {
    return m.data[i * 3 + j];
}

static inline void set(uniform mat3 &m, uniform int i, uniform int j, uniform float x) {
    m.data[i * 3 + j] = x;
}

static inline uniform mat3 operator*(uniform mat3 m1, uniform mat3 m2) {
    uniform mat3 result;
    const uniform float SrcA00 = get(m1, 0, 0);
    const uniform float SrcA01 = get(m1, 0, 1);
    const uniform float SrcA02 = get(m1, 0, 2);
    const uniform float SrcA10 = get(m1, 1, 0);
    const uniform float SrcA11 = get(m1, 1, 1);
    const uniform float SrcA12 = get(m1, 1, 2);
    const uniform float SrcA20 = get(m1, 2, 0);
    const uniform float SrcA21 = get(m1, 2, 1);
    const uniform float SrcA22 = get(m1, 2, 2);

    const uniform float SrcB00 = get(m2, 0, 0);
    const uniform float SrcB01 = get(m2, 0, 1);
    const uniform float SrcB02 = get(m2, 0, 2);
    const uniform float SrcB10 = get(m2, 1, 0);
    const uniform float SrcB11 = get(m2, 1, 1);
    const uniform float SrcB12 = get(m2, 1, 2);
    const uniform float SrcB20 = get(m2, 2, 0);
    const uniform float SrcB21 = get(m2, 2, 1);
    const uniform float SrcB22 = get(m2, 2, 2);

    set(result, 0, 0, SrcA00 * SrcB00 + SrcA10 * SrcB01 + SrcA20 * SrcB02);
    set(result, 0, 1, SrcA01 * SrcB00 + SrcA11 * SrcB01 + SrcA21 * SrcB02);
    set(result, 0, 2, SrcA02 * SrcB00 + SrcA12 * SrcB01 + SrcA22 * SrcB02);
    set(result, 1, 0, SrcA00 * SrcB10 + SrcA10 * SrcB11 + SrcA20 * SrcB12);
    set(result, 1, 1, SrcA01 * SrcB10 + SrcA11 * SrcB11 + SrcA21 * SrcB12);
    set(result, 1, 2, SrcA02 * SrcB10 + SrcA12 * SrcB11 + SrcA22 * SrcB12);
    set(result, 2, 0, SrcA00 * SrcB20 + SrcA10 * SrcB21 + SrcA20 * SrcB22);
    set(result, 2, 1, SrcA01 * SrcB20 + SrcA11 * SrcB21 + SrcA21 * SrcB22);
    set(result, 2, 2, SrcA02 * SrcB20 + SrcA12 * SrcB21 + SrcA22 * SrcB22);

    return result;
}

static inline vec3 operator*(uniform mat3 m, vec3 v) {
    vec3 result;
    result.x = get(m, 0, 0) * v.x + get(m, 1, 0) * v.y + get(m, 2, 0) * v.z;
    result.y = get(m, 0, 1) * v.x + get(m, 1, 1) * v.y + get(m, 2, 1) * v.z;
    result.z = get(m, 0, 2) * v.x + get(m, 1, 2) * v.y + get(m, 2, 2) * v.z;
    return result;
}
static inline vec3 operator*(mat3 m, vec3 v) {
    vec3 result;
    result.x = get(m, 0, 0) * v.x + get(m, 1, 0) * v.y + get(m, 2, 0) * v.z;
    result.y = get(m, 0, 1) * v.x + get(m, 1, 1) * v.y + get(m, 2, 1) * v.z;
    result.z = get(m, 0, 2) * v.x + get(m, 1, 2) * v.y + get(m, 2, 2) * v.z;
    return result;
}
static inline vec3 floor(vec3 v) {
    vec3 result;
    result.x = floor(v.x);
    result.y = floor(v.y);
    result.z = floor(v.z);
    return result;
}

static inline vec2 round(vec2 v) {
    vec2 result;
    result.x = round(v.x);
    result.y = round(v.y);
    return result;
}

static inline vec3 fract(vec3 v) {
    vec3 result;
    result.x = v.x - floor(v.x);
    result.y = v.y - floor(v.y);
    result.z = v.z - floor(v.z);
    return result;
}

static inline vec3 abs(vec3 v) {
    vec3 result;
    result.x = abs(v.x);
    result.y = abs(v.y);
    result.z = abs(v.z);
    return result;
}

static inline float length(vec3 v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}
static inline vec3 normalize(vec3 v) {
    return v / length(v);
}
static inline float dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
#endif

struct DensityNrm {
    float val;
    vec3 nrm;
};

#define ARITHMETIC_OPERATOR_MINMAX(name, type, elem_type, op) \
    static inline type name(type a, type b) {                 \
        type result;                                          \
        result.min = a.min op b.min;                          \
        result.max = a.max op b.max;                          \
        return result;                                        \
    }                                                         \
    static inline type name(type v, elem_type s) {            \
        type result;                                          \
        result.min = v.min op s;                              \
        result.max = v.max op s;                              \
        return result;                                        \
    }                                                         \
    static inline type name(elem_type s, type v) {            \
        type result;                                          \
        result.min = s op v.min;                              \
        result.max = s op v.max;                              \
        return result;                                        \
    }

#define ARITHMETIC_OPERATOR_DENSITY_NRM(name, type, op) \
    static inline type name(type a, type b) {           \
        type result;                                    \
        result.val = a.val op b.val;                    \
        result.nrm = a.nrm op b.nrm;                    \
        return result;                                  \
    }

ARITHMETIC_OPERATOR_MINMAX(operator+, MinMax, int, +)
ARITHMETIC_OPERATOR_MINMAX(operator-, MinMax, int, -)
ARITHMETIC_OPERATOR_MINMAX(operator*, MinMax, int, *)
ARITHMETIC_OPERATOR_MINMAX(operator/, MinMax, int, /)

ARITHMETIC_OPERATOR_DENSITY_NRM(operator+, DensityNrm, +)
ARITHMETIC_OPERATOR_DENSITY_NRM(operator-, DensityNrm, -)

static inline float fast_random(RandomCtx random_ctx, ivec3 p) {
    p = ((p & (RANDOM_BUFFER_SIZE - 1)) + RANDOM_BUFFER_SIZE) & (RANDOM_BUFFER_SIZE - 1);
    float result = random_ctx[p.x + p.y * RANDOM_BUFFER_SIZE + p.z * RANDOM_BUFFER_SIZE * RANDOM_BUFFER_SIZE];
    return result / 255.0f;
    // RNGState rngstate;
    // seed_rng(&rngstate, p.x + p.y * RANDOM_BUFFER_SIZE + p.z * RANDOM_BUFFER_SIZE * RANDOM_BUFFER_SIZE);
    // float result = frandom(&rngstate);
    // return result - floor(result);
}

static float msign(float v) {
    return (v >= 0.0f) ? 1.0f : -1.0f;
}

static vec2 map_octahedral(vec3 nor) {
    const float fac = 1.0f / (abs(nor.x) + abs(nor.y) + abs(nor.z));
    nor.x *= fac;
    nor.y *= fac;
    if (nor.z < 0.0f) {
        const vec2 temp = {nor.x, nor.y};
        nor.x = (1.0f - abs(temp.y)) * msign(temp.x);
        nor.y = (1.0f - abs(temp.x)) * msign(temp.y);
    }
    vec2 result = {nor.x, nor.y};
    return result;
}
static vec3 unmap_octahedral(vec2 v) {
    vec3 nor = {v.x, v.y, 1.0f - abs(v.x) - abs(v.y)}; // Rune Stubbe's version,
    float t = max(-nor.z, 0.0f);                       // much faster than original
    nor.x += (nor.x > 0.0f) ? -t : t;                  // implementation of this
    nor.y += (nor.y > 0.0f) ? -t : t;                  // technique
    return normalize(nor);
}

static float SNORM_SCALE(uint N) {
    float result = 1 << (N - 1u);
    return result - 0.5f;
}
static float UNORM_SCALE(uint N) {
    float result = 1 << N;
    return result - 1.0f;
}
static uint PACK_UNORM(float x, uint N) {
    return round(x * UNORM_SCALE(N));
}
static float UNPACK_UNORM(uint x, uint N) {
    float result = (x) & ((1u << (N)) - 1u);
    return result / UNORM_SCALE(N);
}

#define PACK_SNORM_X2(v, N)                               \
    uvec2 d = round(SNORM_SCALE(N) + v * SNORM_SCALE(N)); \
    return d.x | (d.y << N)
#define UNPACK_SNORM_X2(d, N)            \
    uvec2 tmp0 = {(d), (d) >> N};        \
    vec2 tmp1 = tmp0 & ((1u << N) - 1u); \
    return tmp1 / SNORM_SCALE(N) - 1.0f

// #define UNORM_SCALE(N) ((1 << (N)) - 1.0f)
// #define PACK_UNORM(x, N) (round((x) * UNORM_SCALE((N))))
// #define UNPACK_UNORM(x, N) (((x) & ((1u << (N)) - 1u)) / UNORM_SCALE((N)))

static float sRGB_OETF(float a) {
    if (.0031308f >= a)
        return 12.92f * a;
    return 1.055f * pow(a, .4166666666666667f) - .055f;
}
static vec3 sRGB_OETF(vec3 a) {
    vec3 result;
    result.r = sRGB_OETF(a.r);
    result.g = sRGB_OETF(a.g);
    result.b = sRGB_OETF(a.b);
    return result;
}
static float sRGB_EOTF(float a) {
    if (.04045f < a)
        return pow((a + .055f) / 1.055f, 2.4f);
    return a / 12.92f;
}
static vec3 sRGB_EOTF(vec3 a) {
    vec3 result;
    result.r = sRGB_EOTF(a.r);
    result.g = sRGB_EOTF(a.g);
    result.b = sRGB_EOTF(a.b);
    return result;
}

static uint pack_snorm_2x04(vec2 v) { PACK_SNORM_X2(v, 4); }
static vec2 unpack_snorm_2x04(uint d) { UNPACK_SNORM_X2(d, 4); }

// static uint pack_snorm_2x08(vec2 v) { PACK_SNORM_X2(v, 8); }
// static vec2 unpack_snorm_2x08(uint d) { UNPACK_SNORM_X2(d, 8); }

static uint pack_octahedral_08(vec3 nor) { return pack_snorm_2x04(map_octahedral(nor)); }
static vec3 unpack_octahedral_08(uint data) { return unmap_octahedral(unpack_snorm_2x04(data)); }
// static uint pack_octahedral_16(vec3 nor) { return pack_snorm_2x08(map_octahedral(nor)); }
// static vec3 unpack_octahedral_16(uint data) { return unmap_octahedral(unpack_snorm_2x08(data)); }

static uint pack_rgb565(vec3 col) { return (PACK_UNORM(col.r, 5) << 0) | (PACK_UNORM(col.g, 6) << 5) | (PACK_UNORM(col.b, 5) << 11); }
static vec3 unpack_rgb565(uint data) {
    vec3 result = {UNPACK_UNORM(data >> 0, 5), UNPACK_UNORM(data >> 5, 6), UNPACK_UNORM(data >> 11, 5)};
    return result;
}

struct Voxel {
    vec3 albedo;
    vec3 normal;
    float roughness;
    uint material_type;
};
struct PackedVoxel {
    uint data;
};

static PackedVoxel pack_voxel(Voxel v) {
    vec3 corrected = sRGB_OETF(v.albedo);
    PackedVoxel result;
    result.data =
        (pack_rgb565(corrected)) |
        (pack_octahedral_08(v.normal) << 16) |
        (PACK_UNORM(sqrt(v.roughness), 4) << 24) |
        (v.material_type & 0xf) << 28;
    return result;
}
static Voxel unpack_voxel(PackedVoxel v) {
    Voxel result;
    result.albedo = unpack_rgb565(v.data >> 0);
    result.albedo = sRGB_EOTF(result.albedo);
    result.normal = unpack_octahedral_08(v.data >> 16);
    result.roughness = UNPACK_UNORM((v.data >> 24) & 0xf, 4);
    result.roughness = result.roughness * result.roughness;
    result.material_type = (v.data >> 28) & 0xf;
    return result;
}

// static PackedVoxel pack_voxel(Voxel v) {
//     PackedVoxel result;
//     result.data = pack_rgb565(v.col) | (pack_octahedral_16(v.nrm) << 16);
//     return result;
// }
// static Voxel unpack_voxel(PackedVoxel v) {
//     Voxel result;
//     result.col = unpack_rgb565(v.data >> 0);
//     result.nrm = unpack_octahedral_16(v.data >> 16);
//     return result;
// }

static inline DensityNrm noise_interpolate(float a, float b, float c, float d, float e, float f, float g, float h, vec3 w, float scale, float amp) {
    vec3 u = w * w * (3.0f - 2.0f * w);
    vec3 du = 6.0f * w * (1.0f - w);

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k3 = e - a;
    float k4 = a - b - c + d;
    float k5 = a - c - e + g;
    float k6 = a - b - e + f;
    float k7 = -a + b + c - d + e - f - g + h;

    vec3 d0 = {k1, k2, k3};

    vec3 d1 = {u.y, u.z, u.x};
    vec3 d2 = {k4, k5, k6};

    vec3 d3 = {u.z, u.x, u.y};
    vec3 d4 = {k6, k4, k5};

    vec3 d5 = {u.y, u.z, u.x};
    vec3 d6 = {u.z, u.x, u.y};

    float result_val = k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z;
    vec3 result_nrm = du * (d0 + d1 * d2 + d3 * d4 + k7 * d5 * d6);
    DensityNrm result;
    result.val = (result_val * 2.0f - 1.0f) * amp;
    result.nrm = result_nrm * amp * scale * 2.0f;
    return result;
}

static inline DensityNrm noise(RandomCtx random_ctx, vec3 x, float scale, float amp) {
    x = x * scale;

    ivec3 p = floor(x);
    vec3 w = fract(x);

    const ivec3 offset_a = {0, 0, 0};
    const ivec3 offset_b = {1, 0, 0};
    const ivec3 offset_c = {0, 1, 0};
    const ivec3 offset_d = {1, 1, 0};
    const ivec3 offset_e = {0, 0, 1};
    const ivec3 offset_f = {1, 0, 1};
    const ivec3 offset_g = {0, 1, 1};
    const ivec3 offset_h = {1, 1, 1};

    float a = fast_random(random_ctx, p + offset_a);
    float b = fast_random(random_ctx, p + offset_b);
    float c = fast_random(random_ctx, p + offset_c);
    float d = fast_random(random_ctx, p + offset_d);
    float e = fast_random(random_ctx, p + offset_e);
    float f = fast_random(random_ctx, p + offset_f);
    float g = fast_random(random_ctx, p + offset_g);
    float h = fast_random(random_ctx, p + offset_h);

    return noise_interpolate(a, b, c, d, e, f, g, h, w, scale, amp);
}

float sd_noise_conservative_max_slope(RandomCtx random_ctx, vec3 region_center, vec3 region_size, float scale, float amp) {
#if 1
    vec3 p0 = {(region_center.x - region_size.x) * scale, (region_center.y - region_size.y) * scale, (region_center.z - region_size.z) * scale};
    vec3 p7 = {(region_center.x + region_size.x) * scale, (region_center.y + region_size.y) * scale, (region_center.z + region_size.z) * scale};

    vec3 fp0 = floor(p0 + 0.5f);
    vec3 fp7 = floor(p7 + 0.5f);

    if (fp0.x == fp7.x && fp0.y == fp7.y && fp0.z == fp7.z) {
        vec3 p1 = {(region_center.x + region_size.x) * scale, (region_center.y - region_size.y) * scale, (region_center.z - region_size.z) * scale};
        vec3 p2 = {(region_center.x - region_size.x) * scale, (region_center.y + region_size.y) * scale, (region_center.z - region_size.z) * scale};
        vec3 p3 = {(region_center.x + region_size.x) * scale, (region_center.y + region_size.y) * scale, (region_center.z - region_size.z) * scale};
        vec3 p4 = {(region_center.x - region_size.x) * scale, (region_center.y - region_size.y) * scale, (region_center.z + region_size.z) * scale};
        vec3 p5 = {(region_center.x + region_size.x) * scale, (region_center.y - region_size.y) * scale, (region_center.z + region_size.z) * scale};
        vec3 p6 = {(region_center.x - region_size.x) * scale, (region_center.y + region_size.y) * scale, (region_center.z + region_size.z) * scale};

        float max_slope = 0.0;

        DensityNrm dn;

        dn = noise(random_ctx, p0, 1.0, amp);
        max_slope = max(max_slope, abs(dn.nrm.x) + abs(dn.nrm.y) + abs(dn.nrm.z));
        dn = noise(random_ctx, p1, 1.0, amp);
        max_slope = max(max_slope, abs(dn.nrm.x) + abs(dn.nrm.y) + abs(dn.nrm.z));
        dn = noise(random_ctx, p2, 1.0, amp);
        max_slope = max(max_slope, abs(dn.nrm.x) + abs(dn.nrm.y) + abs(dn.nrm.z));
        dn = noise(random_ctx, p3, 1.0, amp);
        max_slope = max(max_slope, abs(dn.nrm.x) + abs(dn.nrm.y) + abs(dn.nrm.z));
        dn = noise(random_ctx, p4, 1.0, amp);
        max_slope = max(max_slope, abs(dn.nrm.x) + abs(dn.nrm.y) + abs(dn.nrm.z));
        dn = noise(random_ctx, p5, 1.0, amp);
        max_slope = max(max_slope, abs(dn.nrm.x) + abs(dn.nrm.y) + abs(dn.nrm.z));
        dn = noise(random_ctx, p6, 1.0, amp);
        max_slope = max(max_slope, abs(dn.nrm.x) + abs(dn.nrm.y) + abs(dn.nrm.z));
        dn = noise(random_ctx, p7, 1.0, amp);
        max_slope = max(max_slope, abs(dn.nrm.x) + abs(dn.nrm.y) + abs(dn.nrm.z));
        return min(2.0f * 1.5f * scale * amp, max_slope * scale);
    } else
#endif
    {
        // Use the lipschitz constant to compute min/max est. For the cubic interpolation
        // function of this noise function, said constant is 2.0 * 1.5 * scale * amp.
        //  - 2.0 * amp comes from the re-scaling of the range between -amp and amp (line 112-113)
        //  - scale comes from the re-scaling of the domain on the same line (p * scale)
        //  - 1.5 comes from the maximum abs value of the first derivative of the noise
        //    interpolation function (between 0 and 1): d/dx of 3x^2-2x^3 = 6x-6x^2.
        return 2.0 * 1.5 * scale * amp;
    }
}

static inline MinMax minmax_noise_in_region(RandomCtx random_ctx, vec3 region_center, vec3 region_size, float scale, float amp) {
#if 1
    vec3 p0 = {(region_center.x - region_size.x) * scale, (region_center.y - region_size.y) * scale, (region_center.z - region_size.z) * scale};
    vec3 p7 = {(region_center.x + region_size.x) * scale, (region_center.y + region_size.y) * scale, (region_center.z + region_size.z) * scale};

    vec3 fp0 = floor(p0 + 0.5f);
    vec3 fp7 = floor(p7 + 0.5f);

    if (fp0.x == fp7.x && fp0.y == fp7.y && fp0.z == fp7.z) {
        vec3 p1 = {(region_center.x + region_size.x) * scale, (region_center.y - region_size.y) * scale, (region_center.z - region_size.z) * scale};
        vec3 p2 = {(region_center.x - region_size.x) * scale, (region_center.y + region_size.y) * scale, (region_center.z - region_size.z) * scale};
        vec3 p3 = {(region_center.x + region_size.x) * scale, (region_center.y + region_size.y) * scale, (region_center.z - region_size.z) * scale};
        vec3 p4 = {(region_center.x - region_size.x) * scale, (region_center.y - region_size.y) * scale, (region_center.z + region_size.z) * scale};
        vec3 p5 = {(region_center.x + region_size.x) * scale, (region_center.y - region_size.y) * scale, (region_center.z + region_size.z) * scale};
        vec3 p6 = {(region_center.x - region_size.x) * scale, (region_center.y + region_size.y) * scale, (region_center.z + region_size.z) * scale};

        DensityNrm dn;
        MinMax result;

        ivec3 p = floor(p0);
        vec3 w0 = fract(p0);
        vec3 w1 = fract(p1);
        vec3 w2 = fract(p2);
        vec3 w3 = fract(p3);
        vec3 w4 = fract(p4);
        vec3 w5 = fract(p5);
        vec3 w6 = fract(p6);
        vec3 w7 = fract(p7);

        const ivec3 offset_a = {0, 0, 0};
        const ivec3 offset_b = {1, 0, 0};
        const ivec3 offset_c = {0, 1, 0};
        const ivec3 offset_d = {1, 1, 0};
        const ivec3 offset_e = {0, 0, 1};
        const ivec3 offset_f = {1, 0, 1};
        const ivec3 offset_g = {0, 1, 1};
        const ivec3 offset_h = {1, 1, 1};

        float a = fast_random(random_ctx, p + offset_a);
        float b = fast_random(random_ctx, p + offset_b);
        float c = fast_random(random_ctx, p + offset_c);
        float d = fast_random(random_ctx, p + offset_d);
        float e = fast_random(random_ctx, p + offset_e);
        float f = fast_random(random_ctx, p + offset_f);
        float g = fast_random(random_ctx, p + offset_g);
        float h = fast_random(random_ctx, p + offset_h);

        dn = noise_interpolate(a, b, c, d, e, f, g, h, w0, 1.0, amp);
        result.min = dn.val;
        result.max = dn.val;
        dn = noise_interpolate(a, b, c, d, e, f, g, h, w1, 1.0, amp);
        result.min = min(result.min, dn.val);
        result.max = max(result.max, dn.val);
        dn = noise_interpolate(a, b, c, d, e, f, g, h, w2, 1.0, amp);
        result.min = min(result.min, dn.val);
        result.max = max(result.max, dn.val);
        dn = noise_interpolate(a, b, c, d, e, f, g, h, w3, 1.0, amp);
        result.min = min(result.min, dn.val);
        result.max = max(result.max, dn.val);
        dn = noise_interpolate(a, b, c, d, e, f, g, h, w4, 1.0, amp);
        result.min = min(result.min, dn.val);
        result.max = max(result.max, dn.val);
        dn = noise_interpolate(a, b, c, d, e, f, g, h, w5, 1.0, amp);
        result.min = min(result.min, dn.val);
        result.max = max(result.max, dn.val);
        dn = noise_interpolate(a, b, c, d, e, f, g, h, w6, 1.0, amp);
        result.min = min(result.min, dn.val);
        result.max = max(result.max, dn.val);
        dn = noise_interpolate(a, b, c, d, e, f, g, h, w7, 1.0, amp);
        result.min = min(result.min, dn.val);
        result.max = max(result.max, dn.val);
        return result;
    } else
#endif
    {
        // Use the lipschitz constant to compute min/max est. For the cubic interpolation
        // function of this noise function, said constant is 2.0 * 1.5 * scale * amp.
        //  - 2.0 * amp comes from the re-scaling of the range between -amp and amp (line 112-113)
        //  - scale comes from the re-scaling of the domain on the same line (p * scale)
        //  - 1.5 comes from the maximum abs value of the first derivative of the noise
        //    interpolation function (between 0 and 1): d/dx of 3x^2-2x^3 = 6x-6x^2.
        // float lipschitz = 2.0f * 1.5f * scale * amp;
        float lipschitz = sd_noise_conservative_max_slope(random_ctx, region_center, region_size, scale, amp);
        DensityNrm dn = noise(random_ctx, region_center, scale, amp);
        float noise_val = dn.val;
        float max_dist = length(region_size);
        MinMax result;
        result.min = max(noise_val - max_dist * lipschitz, -amp);
        result.max = min(noise_val + max_dist * lipschitz, amp);
        return result;
    }
}

static inline DensityNrm gradient_z(vec3 p, float slope, float offset) {
    DensityNrm result;
    result.val = (p.z - offset) * slope;
    vec3 nrm = {0, 0, slope};
    result.nrm = nrm;
    return result;
}
static inline MinMax minmax_gradient_z(vec3 p0, vec3 p1, float slope, float offset) {
    MinMax temp;
    temp.min = (p0.z - offset) * slope;
    temp.max = (p1.z - offset) * slope;
    MinMax result;
    result.min = min(temp.min, temp.max);
    result.max = max(temp.min, temp.max);
    return result;
}

#if defined(__cplusplus)
#define MAT3_INIT(a, b, c, d, e, f, g, h, i) \
    {a, b, c, d, e, f, g, h, i}
#else
#define MAT3_INIT(a, b, c, d, e, f, g, h, i) \
    {                                        \
        {                                    \
            a, b, c, d, e, f, g, h, i        \
        }                                    \
    }
#endif

const uniform mat3 m = MAT3_INIT(+0.0f, +0.80f, +0.60f,
                                 -0.8f, +0.36f, -0.48f,
                                 -0.6f, -0.48f, +0.64f);
const uniform mat3 mi = MAT3_INIT(+0.0f, -0.80f, -0.60f,
                                  +0.8f, +0.36f, -0.48f,
                                  +0.6f, -0.48f, +0.64f);

static inline DensityNrm voxel_value(RandomCtx random_ctx, uniform NoiseSettings const *uniform noise_settings, vec3 pos) {
    DensityNrm result;
    // pos = fract(pos / 4.0f) * 4.0f - 2.0f;
    // result.val = length(pos) - 2.0f;
    // result.nrm = normalize(pos);
    uniform mat3 inv = MAT3_INIT(1, 0, 0, 0, 1, 0, 0, 0, 1);
    result = gradient_z(pos, 1, 30.0f);
    {
        uniform float noise_persistence = noise_settings->persistence;
        uniform float noise_lacunarity = noise_settings->lacunarity;
        uniform float noise_scale = noise_settings->scale;
        uniform float noise_amplitude = noise_settings->amplitude;
        for (uniform int i = 0; i < noise_settings->octaves; ++i) {
            pos = m * pos;
            inv = mi * inv;
            DensityNrm dn = noise(random_ctx, pos, noise_scale, noise_amplitude);
            result.val += dn.val;
            result.nrm += inv * dn.nrm;
            noise_scale *= noise_lacunarity;
            noise_amplitude *= noise_persistence;
        }
    }
    result.nrm = normalize(result.nrm);
    return result;
}
static inline MinMax voxel_minmax_value(RandomCtx random_ctx, uniform NoiseSettings const *uniform noise_settings, vec3 p0, vec3 p1) {
    MinMax result = {0, 0};
    // MinMax sphere_minmax = {-1, 1};
    // result = result + sphere_minmax;
    result = result + minmax_gradient_z(p0, p1, 1, 30.0f);
    {
        uniform float noise_persistence = noise_settings->persistence;
        uniform float noise_lacunarity = noise_settings->lacunarity;
        uniform float noise_scale = noise_settings->scale;
        uniform float noise_amplitude = noise_settings->amplitude;
        for (uniform int i = 0; i < noise_settings->octaves; ++i) {
            p0 = m * p0;
            p1 = m * p1;
            result = result + minmax_noise_in_region(random_ctx, (p0 + p1) * 0.5f, abs(p1 - p0), noise_scale, noise_amplitude);
            noise_scale *= noise_lacunarity;
            noise_amplitude *= noise_persistence;
        }
    }
    return result;
}

// Building an Orthonormal Basis, Revisited
// http://jcgt.org/published/0006/01/01/
static inline mat3 build_orthonormal_basis(vec3 n) {
    vec3 b1;
    vec3 b2;

    if (n.z < 0.0) {
        const float a = 1.0 / (1.0 - n.z);
        const float b = n.x * n.y * a;
        b1.x = 1.0 - n.x * n.x * a;
        b1.y = -b;
        b1.z = n.x;
        b2.x = b;
        b2.y = n.y * n.y * a - 1.0;
        b2.z = -n.y;
    } else {
        const float a = 1.0 / (1.0 + n.z);
        const float b = -n.x * n.y * a;
        b1.x = 1.0 - n.x * n.x * a;
        b1.y = b;
        b1.z = -n.x;
        b2.x = b;
        b2.y = 1.0 - n.y * n.y * a;
        b2.z = -n.y;
    }

    mat3 result = MAT3_INIT(b1.x, b1.y, b1.z, b2.x, b2.y, b2.z, n.x, n.y, n.z);
    return result;
}

#if !defined M_PI
#define M_PI 3.1415926535f
#endif

static inline vec3 uniform_sample_cone(vec2 urand, float cos_theta_max) {
    float cos_theta = (1.0 - urand.x) + urand.x * cos_theta_max;
    float sin_theta = sqrt(clamp(1.0 - cos_theta * cos_theta, 0.0, 1.0));
    float phi = urand.y * (M_PI * 2.0);
    vec3 result = {sin_theta * cos(phi), sin_theta * sin(phi), cos_theta};
    return result;
}

static inline vec3 dither_nrm(RandomCtx random_ctx, vec3 nrm, ivec3 pos) {
    ivec3 o0 = {nrm.x * 100, nrm.y * 100, nrm.z * 100};
    ivec3 o1 = {nrm.z * 100, nrm.x * 100, nrm.y * 100};

    const mat3 basis = build_orthonormal_basis(normalize(nrm));
    const vec2 urand = {fast_random(random_ctx, pos + o0), fast_random(random_ctx, pos + o1)};
    vec3 p = uniform_sample_cone(urand, cos(0.0123));
    return basis * p;
}

const uniform vec3 UP = {0, 0, 1};

const uniform vec3 GRASS_COL = {0.1174, 0.3131, 0.0157};
const uniform vec3 DIRT_COL = {0.2391, 0.1345, 0.0415};
const uniform vec3 STONE_COL = {0.3602, 0.2977, 0.1459};
const uniform vec3 GRAVEL_COL = {0.2559, 0.1992, 0.0434};
