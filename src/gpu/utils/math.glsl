#pragma once

// Definitions

#define PI 3.14159265
#define MAX_SD 10000.0

// Objects

struct Sphere {
    f32vec3 o;
    f32 r;
};
struct BoundingBox {
    f32vec3 bound_min, bound_max;
};
struct CapsulePoints {
    f32vec3 p0, p1;
    f32 r;
};

// Common Functions

f32 deg2rad(f32 d) {
    return d * PI / 180.0;
}
f32 rad2deg(f32 r) {
    return r * 180.0 / PI;
}
f32vec3 rotate_x(f32vec3 v, f32 angle) {
    float sin_rot_x = sin(angle), cos_rot_x = cos(angle);
    f32mat3x3 rot_mat = f32mat3x3(
        1, 0, 0,
        0, cos_rot_x, sin_rot_x,
        0, -sin_rot_x, cos_rot_x);
    return rot_mat * v;
}
f32vec3 rotate_y(f32vec3 v, f32 angle) {
    float sin_rot_y = sin(angle), cos_rot_y = cos(angle);
    f32mat3x3 rot_mat = f32mat3x3(
        cos_rot_y, 0, sin_rot_y,
        0, 1, 0,
        -sin_rot_y, 0, cos_rot_y);
    return rot_mat * v;
}
f32vec3 rotate_z(f32vec3 v, f32 angle) {
    float sin_rot_z = sin(angle), cos_rot_z = cos(angle);
    f32mat3x3 rot_mat = f32mat3x3(
        cos_rot_z, -sin_rot_z, 0,
        sin_rot_z, cos_rot_z, 0,
        0, 0, 1);
    return rot_mat * v;
}
f32vec3 rgb2hsv(f32vec3 c) {
    f32vec4 K = f32vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    f32vec4 p = mix(f32vec4(c.bg, K.wz), f32vec4(c.gb, K.xy), step(c.b, c.g));
    f32vec4 q = mix(f32vec4(p.xyw, c.r), f32vec4(c.r, p.yzx), step(p.x, c.r));
    f32 d = q.x - min(q.w, q.y);
    f32 e = 1.0e-10;
    return f32vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
f32vec3 hsv2rgb(f32vec3 c) {
    f32vec4 k = f32vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    f32vec3 p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * mix(k.xxx, clamp(p - k.xxx, 0.0, 1.0), c.y);
}
f32vec4 uint_to_float4(u32 u) {
    f32vec4 result;
    result.r = f32((u >> 0x00) & 0xff) / 255.0;
    result.g = f32((u >> 0x08) & 0xff) / 255.0;
    result.b = f32((u >> 0x10) & 0xff) / 255.0;
    result.a = f32((u >> 0x18) & 0xff) / 255.0;

    result = pow(result, f32vec4(2.2));

    // result = result * f32vec4(16, 32, 8, 32);
    // result.gba = pow(result.gba, f32vec3(2.2));
    // result.g = 1 - result.g;
    // result.rgb = hsv2rgb(result.rgb);

    // f32 c = result.r - ( 16.0 / 256.0);
    // f32 d = result.g - (128.0 / 256.0);
    // f32 e = result.b - (128.0 / 256.0);
    // result.r = (298.0 / 256.0) * c + (409.0 / 256.0) * e + (128.0 / 256.0);
    // result.g = (298.0 / 256.0) * c + (100.0 / 256.0) * d + (208.0 / 256.0) * e + 128.0 / 256.0;
    // result.b = (298.0 / 256.0) * c + (516.0 / 256.0) * d + (128.0 / 256.0);

    return result;
}
u32 float4_to_uint(f32vec4 f) {
    f = clamp(f, f32vec4(0), f32vec4(1));

    // f32vec3 yuv = f32vec3(
    //     dot(f32vec3(0.299, 0.587, 0.114), f.rgb),
    //     dot(f32vec3(-0.174, -0.289, 0.436), f.rgb),
    //     dot(f32vec3(0.615, -0.515, -0.100), f.rgb));
    // f.rgb = yuv;

    // f.rgb = rgb2hsv(f.rgb);
    // f.g = 1 - f.g;
    // f.gba = pow(f.gba, f32vec3(1.0 / 2.2));
    // f = f / f32vec4(16, 32, 8, 32);

    f = pow(f, f32vec4(1.0 / 2.2));

    u32 result = 0;
    result |= u32(clamp(f.r, 0, 1) * 255) << 0x00;
    result |= u32(clamp(f.g, 0, 1) * 255) << 0x08;
    result |= u32(clamp(f.b, 0, 1) * 255) << 0x10;
    result |= u32(clamp(f.a, 0, 1) * 255) << 0x18;
    return result;
}
u32 ceil_log2(u32 x) {
    return findMSB(x) + u32(bitCount(x) > 1);
}

// Bit Functions

void flag_set(in out u32 bitfield, u32 index, bool value) {
    if (value) {
        bitfield |= 1u << index;
    } else {
        bitfield &= ~(1u << index);
    }
}
bool flag_get(u32 bitfield, u32 index) {
    return ((bitfield >> index) & 1u) == 1u;
}

// Shape Functions

b32 inside(f32vec3 p, Sphere s) {
    return dot(p - s.o, p - s.o) < s.r * s.r;
}
b32 inside(f32vec3 p, BoundingBox b) {
    return (p.x >= b.bound_min.x && p.x < b.bound_max.x &&
            p.y >= b.bound_min.y && p.y < b.bound_max.y &&
            p.z >= b.bound_min.z && p.z < b.bound_max.z);
}
b32 overlaps(BoundingBox a, BoundingBox b) {
    b32 x_overlap = a.bound_max.x >= b.bound_min.x && b.bound_max.x >= a.bound_min.x;
    b32 y_overlap = a.bound_max.y >= b.bound_min.y && b.bound_max.y >= a.bound_min.y;
    b32 z_overlap = a.bound_max.z >= b.bound_min.z && b.bound_max.z >= a.bound_min.z;
    return x_overlap && y_overlap && z_overlap;
}
void intersect(in out f32vec3 ray_pos, f32vec3 ray_dir, f32vec3 inv_dir, BoundingBox b) {
    if (inside(ray_pos, b)) {
        return;
    }
    f32 tx1 = (b.bound_min.x - ray_pos.x) * inv_dir.x;
    f32 tx2 = (b.bound_max.x - ray_pos.x) * inv_dir.x;
    f32 tmin = min(tx1, tx2);
    f32 tmax = max(tx1, tx2);
    f32 ty1 = (b.bound_min.y - ray_pos.y) * inv_dir.y;
    f32 ty2 = (b.bound_max.y - ray_pos.y) * inv_dir.y;
    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));
    f32 tz1 = (b.bound_min.z - ray_pos.z) * inv_dir.z;
    f32 tz2 = (b.bound_max.z - ray_pos.z) * inv_dir.z;
    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));

    // f32 dist = max(min(tmax, tmin), 0);
    f32 dist = MAX_SD;
    if (tmax >= tmin) {
        if (tmin > 0) {
            dist = tmin;
        }
    }

    ray_pos = ray_pos + ray_dir * dist;
}

/// SIGNED DISTANCE FUNCTIONS

f32 sd_shapes_dot2(in f32vec2 v) { return dot(v, v); }
f32 sd_shapes_dot2(in f32vec3 v) { return dot(v, v); }
f32 sd_shapes_ndot(in f32vec2 a, in f32vec2 b) { return a.x * b.x - a.y * b.y; }

// Operators

f32 sd_union(in f32 a, in f32 b) {
    return min(a, b);
}
f32 sd_smooth_union(in f32 a, in f32 b, in f32 k) {
    f32 h = clamp(0.5 + 0.5 * (a - b) / k, 0.0, 1.0);
    return mix(a, b, h) - k * h * (1.0 - h);
}
f32 sd_intersection(in f32 a, in f32 b) {
    return max(a, b);
}
f32 sd_smooth_intersection(in f32 a, in f32 b, in f32 k) {
    return sd_smooth_union(a, b, -k);
}
f32 sd_difference(in f32 a, in f32 b) {
    return sd_intersection(a, -b);
}
f32 sd_smooth_difference(in f32 a, in f32 b, in f32 k) {
    return sd_smooth_intersection(a, -b, k);
}

// Shapes

f32 sd_plane_x(in f32vec3 p) {
    return p.x;
}
f32 sd_plane_y(in f32vec3 p) {
    return p.y;
}
f32 sd_plane_z(in f32vec3 p) {
    return p.z;
}
f32 sd_sphere(in f32vec3 p, in f32 r) {
    return length(p) - r;
}
f32 sd_ellipsoid(in f32vec3 p, in f32vec3 r) {
    f32 k0 = length(p / r);
    f32 k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}
f32 sd_box(in f32vec3 p, in f32vec3 size) {
    f32vec3 d = abs(p) - size;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}
f32 sd_box(in f32vec3 p, in BoundingBox box) {
    return sd_box(p - (box.bound_max + box.bound_min) * 0.5, (box.bound_max - box.bound_min) * 0.5);
}
f32 sd_box_frame(in f32vec3 p, in f32vec3 b, in f32 e) {
    p = abs(p) - b;
    f32vec3 q = abs(p + e) - e;
    return min(
        min(length(max(f32vec3(p.x, q.y, q.z), 0.0)) + min(max(p.x, max(q.y, q.z)), 0.0),
            length(max(f32vec3(q.x, p.y, q.z), 0.0)) + min(max(q.x, max(p.y, q.z)), 0.0)),
        length(max(f32vec3(q.x, q.y, p.z), 0.0)) + min(max(q.x, max(q.y, p.z)), 0.0));
}
f32 sd_box_frame(in f32vec3 p, in BoundingBox box, in f32 e) {
    return sd_box_frame(p - (box.bound_max + box.bound_min) * 0.5, (box.bound_max - box.bound_min) * 0.5, e);
}
f32 sd_cylinder(in f32vec3 p, in f32 r, in f32 h) {
    f32vec2 d = abs(f32vec2(length(p.xy), p.z)) - f32vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}
f32 sd_cylinder(f32vec3 p, f32vec3 a, f32vec3 b, f32 r) {
    f32vec3 ba = b - a;
    f32vec3 pa = p - a;
    f32 baba = dot(ba, ba);
    f32 paba = dot(pa, ba);
    f32 x = length(pa * baba - ba * paba) - r * baba;
    f32 y = abs(paba - baba * 0.5) - baba * 0.5;
    f32 x2 = x * x;
    f32 y2 = y * y * baba;
    f32 d = (max(x, y) < 0.0) ? -min(x2, y2) : (((x > 0.0) ? x2 : 0.0) + ((y > 0.0) ? y2 : 0.0));
    return sign(d) * sqrt(abs(d)) / baba;
}
f32 sd_triangular_prism(in f32vec3 p, in f32 r, in f32 h) {
    const f32 k = sqrt(3.0);
    h *= 0.5 * k;
    p.xy /= h;
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0 / k;
    if (p.x + k * p.y > 0.0)
        p.xy = f32vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
    p.x -= clamp(p.x, -2.0, 0.0);
    f32 d1 = length(p.xy) * sign(-p.y) * h;
    f32 d2 = abs(p.z) - r;
    return length(max(f32vec2(d1, d2), 0.0)) + min(max(d1, d2), 0.);
}
f32 sd_hexagonal_prism(in f32vec3 p, in f32 r, in f32 h) {
    f32vec3 q = abs(p);
    const f32vec3 k = f32vec3(-0.8660254, 0.5, 0.57735);
    p = abs(p);
    p.xy -= 2.0 * min(dot(k.xy, p.xy), 0.0) * k.xy;
    f32vec2 d = f32vec2(
        length(p.xy - f32vec2(clamp(p.x, -k.z * h, k.z * h), h)) * sign(p.y - h),
        p.z - r);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}
f32 sd_octagonal_prism(in f32vec3 p, in f32 r, in f32 h) {
    const f32vec3 k = f32vec3(-0.9238795325, 0.3826834323, 0.4142135623);
    p = abs(p);
    p.xy -= 2.0 * min(dot(f32vec2(k.x, k.y), p.xy), 0.0) * f32vec2(k.x, k.y);
    p.xy -= 2.0 * min(dot(f32vec2(-k.x, k.y), p.xy), 0.0) * f32vec2(-k.x, k.y);
    p.xy -= f32vec2(clamp(p.x, -k.z * r, k.z * r), r);
    f32vec2 d = f32vec2(length(p.xy) * sign(p.y), p.z - h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}
f32 sd_capsule(in f32vec3 p, in f32vec3 a, in f32vec3 b, in f32 r) {
    f32vec3 pa = p - a, ba = b - a;
    f32 h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}
f32 sd_cone(in f32vec3 p, in f32 c, in f32 h) {
    f32vec2 q = h * f32vec2(c, -1);
    f32vec2 w = f32vec2(length(p.xy), p.z);
    f32vec2 a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
    f32vec2 b = w - q * f32vec2(clamp(w.x / q.x, 0.0, 1.0), 1.0);
    f32 k = sign(q.y);
    f32 d = min(dot(a, a), dot(b, b));
    f32 s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
    return sqrt(d) * sign(s);
}
f32 sd_round_cone(in f32vec3 p, in f32 r1, in f32 r2, in f32 h) {
    f32vec2 q = f32vec2(length(p.xy), p.z);
    f32 b = (r1 - r2) / h;
    f32 a = sqrt(1.0 - b * b);
    f32 k = dot(q, f32vec2(-b, a));
    if (k < 0.0)
        return length(q) - r1;
    if (k > a * h)
        return length(q - f32vec2(0.0, h)) - r2;
    return dot(q, f32vec2(a, b)) - r1;
}
f32 sd_round_cone(in f32vec3 p, in f32vec3 a, in f32vec3 b, in f32 r1, in f32 r2) {
    f32vec3 ba = b - a;
    f32 l2 = dot(ba, ba);
    f32 rr = r1 - r2;
    f32 a2 = l2 - rr * rr;
    f32 il2 = 1.0 / l2;
    f32vec3 pa = p - a;
    f32 y = dot(pa, ba);
    f32 z = y - l2;
    f32vec3 xp = pa * l2 - ba * y;
    f32 x2 = dot(xp, xp);
    f32 y2 = y * y * l2;
    f32 z2 = z * z * l2;
    f32 k = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k)
        return sqrt(x2 + z2) * il2 - r2;
    if (sign(y) * a2 * y2 < k)
        return sqrt(x2 + y2) * il2 - r1;
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}
f32 sd_capped_cone(in f32vec3 p, in f32 r1, in f32 r2, in f32 h) {
    f32vec2 q = f32vec2(length(p.xy), p.z);
    f32vec2 k1 = f32vec2(r2, h);
    f32vec2 k2 = f32vec2(r2 - r1, 2.0 * h);
    f32vec2 ca = f32vec2(q.x - min(q.x, (q.y < 0.0) ? r1 : r2), abs(q.y) - h);
    f32vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / sd_shapes_dot2(k2), 0.0, 1.0);
    f32 s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return s * sqrt(min(sd_shapes_dot2(ca), sd_shapes_dot2(cb)));
}
f32 sd_capped_cone(in f32vec3 p, in f32vec3 a, in f32vec3 b, in f32 ra, in f32 rb) {
    f32 rba = rb - ra;
    f32 baba = dot(b - a, b - a);
    f32 papa = dot(p - a, p - a);
    f32 paba = dot(p - a, b - a) / baba;
    f32 x = sqrt(papa - paba * paba * baba);
    f32 cax = max(0.0, x - ((paba < 0.5) ? ra : rb));
    f32 cay = abs(paba - 0.5) - 0.5;
    f32 k = rba * rba + baba;
    f32 f = clamp((rba * (x - ra) + paba * baba) / k, 0.0, 1.0);
    f32 cbx = x - ra - f * rba;
    f32 cby = paba - f;
    f32 s = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;
    return s * sqrt(min(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba));
}
f32 sd_torus(in f32vec3 p, in f32vec2 t) {
    return length(f32vec2(length(p.xy) - t.x, p.z)) - t.y;
}
f32 sd_octahedron(in f32vec3 p, in f32 s) {
    p = abs(p);
    f32 m = p.x + p.y + p.z - s;
    f32vec3 q;
    if (3.0 * p.x < m)
        q = p.xyz;
    else if (3.0 * p.y < m)
        q = p.yzx;
    else if (3.0 * p.z < m)
        q = p.zxy;
    else
        return m * 0.57735027;
    f32 k = clamp(0.5 * (q.z - q.y + s), 0.0, s);
    return length(f32vec3(q.x, q.y - s + k, q.z - k));
}
f32 sd_pyramid(in f32vec3 p, in f32 r, in f32 h) {
    h = h / r;
    p = p / r;
    f32 m2 = h * h + 0.25;
    p.xy = abs(p.xy);
    p.xy = (p.y > p.x) ? p.yx : p.xy;
    p.xy -= 0.5;
    f32vec3 q = f32vec3(p.y, h * p.z - 0.5 * p.x, h * p.x + 0.5 * p.z);
    f32 s = max(-q.x, 0.0);
    f32 t = clamp((q.y - 0.5 * p.y) / (m2 + 0.25), 0.0, 1.0);
    f32 a = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
    f32 b = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);
    f32 d2 = min(q.y, -q.x * m2 - q.y * 0.5) > 0.0 ? 0.0 : min(a, b);
    return sqrt((d2 + q.z * q.z) / m2) * sign(max(q.z, -p.z)) * r;
}

// Random functions

u32 good_rand_hash(u32 x) {
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}
u32 good_rand_hash(u32vec2 v) { return good_rand_hash(v.x ^ good_rand_hash(v.y)); }
u32 good_rand_hash(u32vec3 v) {
    return good_rand_hash(v.x ^ good_rand_hash(v.y) ^ good_rand_hash(v.z));
}
u32 good_rand_hash(u32vec4 v) {
    return good_rand_hash(v.x ^ good_rand_hash(v.y) ^ good_rand_hash(v.z) ^ good_rand_hash(v.w));
}
f32 good_rand_float_construct(u32 m) {
    const u32 ieee_mantissa = 0x007FFFFFu;
    const u32 ieee_one = 0x3F800000u;
    m &= ieee_mantissa;
    m |= ieee_one;
    f32 f = uintBitsToFloat(m);
    return f - 1.0;
}
f32 good_rand(f32 x) { return good_rand_float_construct(good_rand_hash(floatBitsToUint(x))); }
f32 good_rand(f32vec2 v) { return good_rand_float_construct(good_rand_hash(floatBitsToUint(v))); }
f32 good_rand(f32vec3 v) { return good_rand_float_construct(good_rand_hash(floatBitsToUint(v))); }
f32 good_rand(f32vec4 v) { return good_rand_float_construct(good_rand_hash(floatBitsToUint(v))); }

/// Hash without Sine
// MIT License...
/* Copyright (c)2014 David Hoskins.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

// Gabe Rundlett
// Modified mostly for my own naming

f32 rand(f32 p) {
    p = fract(p * .1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p) * 2 - 1;
}
f32 rand(f32vec2 p) {
    f32vec3 p3 = fract(f32vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z) * 2 - 1;
}
f32 rand(f32vec3 p3) {
    p3 = fract(p3 * .1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z) * 2 - 1;
}
f32 rand(f32vec4 p4) {
    p4 = fract(p4 * f32vec4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.x + p4.y) * (p4.z + p4.w)) * 2 - 1;
}

f32vec2 rand2(f32 p) {
    f32vec3 p3 = fract(f32vec3(p) * f32vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}
f32vec2 rand2(f32vec2 p) {
    f32vec3 p3 = fract(f32vec3(p.xyx) * f32vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}
f32vec2 rand2(f32vec3 p3) {
    p3 = fract(p3 * f32vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}
f32vec2 rand2(f32vec4 p4) {
    p4 = fract(p4 * f32vec4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xy + p4.yz) * p4.zy);
}

f32vec3 rand3(f32 p) {
    f32vec3 p3 = fract(f32vec3(p) * f32vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xxy + p3.yzz) * p3.zyx);
}
f32vec3 rand3(f32vec2 p) {
    f32vec3 p3 = fract(f32vec3(p.xyx) * f32vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yzz) * p3.zyx);
}
f32vec3 rand3(f32vec3 p3) {
    p3 = fract(p3 * f32vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}
f32vec3 rand3(f32vec4 p4) {
    p4 = fract(p4 * f32vec4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xyz + p4.yzw) * p4.zyw);
}

f32vec4 rand4(f32 p) {
    f32vec4 p4 = fract(f32vec4(p) * f32vec4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xxyz + p4.yzzw) * p4.zywx);
}
f32vec4 rand4(f32vec2 p) {
    f32vec4 p4 = fract(f32vec4(p.xyxy) * f32vec4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xxyz + p4.yzzw) * p4.zywx);
}
f32vec4 rand4(f32vec3 p) {
    f32vec4 p4 = fract(f32vec4(p.xyzx) * f32vec4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xxyz + p4.yzzw) * p4.zywx);
}
f32vec4 rand4(f32vec4 p4) {
    p4 = fract(p4 * f32vec4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xxyz + p4.yzzw) * p4.zywx);
}

/// end Hash without Sine

f32vec3 ortho(f32vec3 v) {
    return mix(f32vec3(-v.y, v.x, 0.0), f32vec3(0.0, -v.z, v.y), step(abs(v.x), abs(v.z)));
}

f32vec3 around(f32vec3 v, f32vec3 z) {
    f32vec3 t = ortho(z), b = cross(z, t);
    return t * f32vec3(v.x, v.x, v.x) + (b * f32vec3(v.y, v.y, v.y) + (z * v.z));
}

f32vec3 isotropic(f32 rp, f32 c) {
    f32 p = 2 * 3.14159 * rp, s = sqrt(1.0 - c * c);
    return f32vec3(cos(p) * s, sin(p) * s, c);
}

f32vec3 rand_pt(f32vec3 n, f32vec2 rnd) {
    f32 c = sqrt(rnd.y);
    return around(isotropic(rnd.x, c), n);
}

f32vec3 rand_pt_in_sphere(f32vec2 rnd) {
    f32 l = acos(2 * rnd.x - 1) - PI / 2;
    f32 p = 2 * PI * rnd.y;
    return f32vec3(cos(l) * cos(p), cos(l) * sin(p), sin(l));
}

f32vec3 rand_lambertian_nrm(f32vec3 n, f32vec2 rnd) {
    f32vec3 pt = rand_pt_in_sphere(rnd);
    return normalize(pt + n);
}

f32vec3 rand_lambertian_reflect(f32vec3 i, f32vec3 n, f32vec2 rnd, f32 roughness) {
    f32vec3 pt = rand_pt_in_sphere(rnd) * clamp(roughness, 0, 1);
    return normalize(pt + reflect(i, n));
}
