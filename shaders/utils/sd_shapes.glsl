#pragma once

#include <shared/shapes.inl>
#include <utils/math.glsl>

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

f32 sd_plane(in f32vec3 p) {
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
    f32vec2 d = abs(f32vec2(length(p.xy), p.z)) - f32vec2(h, r);
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
