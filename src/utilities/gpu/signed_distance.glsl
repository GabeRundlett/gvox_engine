#pragma once

daxa_f32 sd_shapes_dot2(in daxa_f32vec2 v) { return dot(v, v); }
daxa_f32 sd_shapes_dot2(in daxa_f32vec3 v) { return dot(v, v); }
daxa_f32 sd_shapes_ndot(in daxa_f32vec2 a, in daxa_f32vec2 b) { return a.x * b.x - a.y * b.y; }

// Operators

// These are safe for min/max operations!
daxa_f32 sd_set(daxa_f32 a, daxa_f32 b) {
    return b;
}
daxa_f32vec4 sd_set(daxa_f32vec4 a, daxa_f32vec4 b) {
    return b;
}
daxa_f32 sd_add(daxa_f32 a, daxa_f32 b) {
    return (a + b);
}
daxa_f32 sd_union(in daxa_f32 a, in daxa_f32 b) {
    return min(a, b);
}

// These are either unsafe or unknown for min/max operations
daxa_f32 sd_smooth_union(in daxa_f32 a, in daxa_f32 b, in daxa_f32 k) {
    daxa_f32 h = clamp(0.5 + 0.5 * (a - b) / k, 0.0, 1.0);
    return mix(a, b, h) - k * h * (1.0 - h);
}
daxa_f32 sd_intersection(in daxa_f32 a, in daxa_f32 b) {
    return max(a, b);
}
daxa_f32 sd_smooth_intersection(in daxa_f32 a, in daxa_f32 b, in daxa_f32 k) {
    return sd_smooth_union(a, b, -k);
}
daxa_f32 sd_difference(in daxa_f32 a, in daxa_f32 b) {
    return sd_intersection(a, -b);
}
daxa_f32 sd_smooth_difference(in daxa_f32 a, in daxa_f32 b, in daxa_f32 k) {
    return sd_smooth_intersection(a, -b, k);
}
daxa_f32 sd_mul(daxa_f32 a, daxa_f32 b) {
    return (a * b);
}

// Shapes

// assumed sphere is at (0, 0, 0)
float sd_sphere_nearest(vec3 p, float r) {
    return length(p) - r;
}
float sd_sphere_furthest(vec3 p, float r) {
    return length(p) + r * 2.0;
}

// assumed box is centered at (0, 0, 0)
float sd_box_nearest(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}
float sd_box_furthest(vec3 p, vec3 b) {
    return length(b * nonzero_sign(p) + p);
}

// assumed sphere is at (0, 0, 0)
vec2 minmax_sd_sphere_in_region(vec3 region_center, vec3 region_size, float r) {
    float min_d = sd_box_nearest(-region_center, region_size);
    float max_d = sd_box_furthest(-region_center, region_size);
    return vec2(min_d, max_d) - r;
}

daxa_f32 sd_plane_x(in daxa_f32vec3 p) {
    return p.x;
}
daxa_f32 sd_plane_y(in daxa_f32vec3 p) {
    return p.y;
}
daxa_f32 sd_plane_z(in daxa_f32vec3 p) {
    return p.z;
}
daxa_f32 sd_sphere(in daxa_f32vec3 p, in daxa_f32 r) {
    return length(p) - r;
}
daxa_f32 sd_ellipsoid(in daxa_f32vec3 p, in daxa_f32vec3 r) {
    daxa_f32 k0 = length(p / r);
    daxa_f32 k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}
daxa_f32 sd_box(in daxa_f32vec3 p, in daxa_f32vec3 size) {
    daxa_f32vec3 d = abs(p) - size;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}
daxa_f32 sd_box(in daxa_f32vec3 p, in BoundingBox box) {
    return sd_box(p - (box.bound_max + box.bound_min) * 0.5, (box.bound_max - box.bound_min) * 0.5);
}
daxa_f32 sd_box_frame(in daxa_f32vec3 p, in daxa_f32vec3 b, in daxa_f32 e) {
    p = abs(p) - b;
    daxa_f32vec3 q = abs(p + e) - e;
    return min(
        min(length(max(daxa_f32vec3(p.x, q.y, q.z), 0.0)) + min(max(p.x, max(q.y, q.z)), 0.0),
            length(max(daxa_f32vec3(q.x, p.y, q.z), 0.0)) + min(max(q.x, max(p.y, q.z)), 0.0)),
        length(max(daxa_f32vec3(q.x, q.y, p.z), 0.0)) + min(max(q.x, max(q.y, p.z)), 0.0));
}
daxa_f32 sd_box_frame(in daxa_f32vec3 p, in BoundingBox box, in daxa_f32 e) {
    return sd_box_frame(p - (box.bound_max + box.bound_min) * 0.5, (box.bound_max - box.bound_min) * 0.5, e);
}
daxa_f32 sd_cylinder(in daxa_f32vec3 p, in daxa_f32 r, in daxa_f32 h) {
    daxa_f32vec2 d = abs(daxa_f32vec2(length(p.xy), p.z)) - daxa_f32vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}
daxa_f32 sd_cylinder(daxa_f32vec3 p, daxa_f32vec3 a, daxa_f32vec3 b, daxa_f32 r) {
    daxa_f32vec3 ba = b - a;
    daxa_f32vec3 pa = p - a;
    daxa_f32 baba = dot(ba, ba);
    daxa_f32 paba = dot(pa, ba);
    daxa_f32 x = length(pa * baba - ba * paba) - r * baba;
    daxa_f32 y = abs(paba - baba * 0.5) - baba * 0.5;
    daxa_f32 x2 = x * x;
    daxa_f32 y2 = y * y * baba;
    daxa_f32 d = (max(x, y) < 0.0) ? -min(x2, y2) : (((x > 0.0) ? x2 : 0.0) + ((y > 0.0) ? y2 : 0.0));
    return sign(d) * sqrt(abs(d)) / baba;
}
daxa_f32 sd_triangular_prism(in daxa_f32vec3 p, in daxa_f32 r, in daxa_f32 h) {
    const daxa_f32 k = sqrt(3.0);
    h *= 0.5 * k;
    p.xy /= h;
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0 / k;
    if (p.x + k * p.y > 0.0)
        p.xy = daxa_f32vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
    p.x -= clamp(p.x, -2.0, 0.0);
    daxa_f32 d1 = length(p.xy) * sign(-p.y) * h;
    daxa_f32 d2 = abs(p.z) - r;
    return length(max(daxa_f32vec2(d1, d2), 0.0)) + min(max(d1, d2), 0.);
}
daxa_f32 sd_hexagonal_prism(in daxa_f32vec3 p, in daxa_f32 r, in daxa_f32 h) {
    daxa_f32vec3 q = abs(p);
    const daxa_f32vec3 k = daxa_f32vec3(-0.8660254, 0.5, 0.57735);
    p = abs(p);
    p.xy -= 2.0 * min(dot(k.xy, p.xy), 0.0) * k.xy;
    daxa_f32vec2 d = daxa_f32vec2(
        length(p.xy - daxa_f32vec2(clamp(p.x, -k.z * h, k.z * h), h)) * sign(p.y - h),
        p.z - r);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}
daxa_f32 sd_octagonal_prism(in daxa_f32vec3 p, in daxa_f32 r, in daxa_f32 h) {
    const daxa_f32vec3 k = daxa_f32vec3(-0.9238795325, 0.3826834323, 0.4142135623);
    p = abs(p);
    p.xy -= 2.0 * min(dot(daxa_f32vec2(k.x, k.y), p.xy), 0.0) * daxa_f32vec2(k.x, k.y);
    p.xy -= 2.0 * min(dot(daxa_f32vec2(-k.x, k.y), p.xy), 0.0) * daxa_f32vec2(-k.x, k.y);
    p.xy -= daxa_f32vec2(clamp(p.x, -k.z * r, k.z * r), r);
    daxa_f32vec2 d = daxa_f32vec2(length(p.xy) * sign(p.y), p.z - h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}
daxa_f32 sd_capsule(in daxa_f32vec3 p, in daxa_f32vec3 a, in daxa_f32vec3 b, in daxa_f32 r) {
    daxa_f32vec3 pa = p - a, ba = b - a;
    daxa_f32 h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}
daxa_f32 sd_cone(in daxa_f32vec3 p, in daxa_f32 c, in daxa_f32 h) {
    daxa_f32vec2 q = h * daxa_f32vec2(c, -1);
    daxa_f32vec2 w = daxa_f32vec2(length(p.xy), p.z);
    daxa_f32vec2 a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
    daxa_f32vec2 b = w - q * daxa_f32vec2(clamp(w.x / q.x, 0.0, 1.0), 1.0);
    daxa_f32 k = sign(q.y);
    daxa_f32 d = min(dot(a, a), dot(b, b));
    daxa_f32 s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
    return sqrt(d) * sign(s);
}
daxa_f32 sd_round_cone(in daxa_f32vec3 p, in daxa_f32 r1, in daxa_f32 r2, in daxa_f32 h) {
    daxa_f32vec2 q = daxa_f32vec2(length(p.xy), p.z);
    daxa_f32 b = (r1 - r2) / h;
    daxa_f32 a = sqrt(1.0 - b * b);
    daxa_f32 k = dot(q, daxa_f32vec2(-b, a));
    if (k < 0.0)
        return length(q) - r1;
    if (k > a * h)
        return length(q - daxa_f32vec2(0.0, h)) - r2;
    return dot(q, daxa_f32vec2(a, b)) - r1;
}
daxa_f32 sd_round_cone(in daxa_f32vec3 p, in daxa_f32vec3 a, in daxa_f32vec3 b, in daxa_f32 r1, in daxa_f32 r2) {
    daxa_f32vec3 ba = b - a;
    daxa_f32 l2 = dot(ba, ba);
    daxa_f32 rr = r1 - r2;
    daxa_f32 a2 = l2 - rr * rr;
    daxa_f32 il2 = 1.0 / l2;
    daxa_f32vec3 pa = p - a;
    daxa_f32 y = dot(pa, ba);
    daxa_f32 z = y - l2;
    daxa_f32vec3 xp = pa * l2 - ba * y;
    daxa_f32 x2 = dot(xp, xp);
    daxa_f32 y2 = y * y * l2;
    daxa_f32 z2 = z * z * l2;
    daxa_f32 k = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k)
        return sqrt(x2 + z2) * il2 - r2;
    if (sign(y) * a2 * y2 < k)
        return sqrt(x2 + y2) * il2 - r1;
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}
daxa_f32 sd_capped_cone(in daxa_f32vec3 p, in daxa_f32 r1, in daxa_f32 r2, in daxa_f32 h) {
    daxa_f32vec2 q = daxa_f32vec2(length(p.xy), p.z);
    daxa_f32vec2 k1 = daxa_f32vec2(r2, h);
    daxa_f32vec2 k2 = daxa_f32vec2(r2 - r1, 2.0 * h);
    daxa_f32vec2 ca = daxa_f32vec2(q.x - min(q.x, (q.y < 0.0) ? r1 : r2), abs(q.y) - h);
    daxa_f32vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / sd_shapes_dot2(k2), 0.0, 1.0);
    daxa_f32 s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return s * sqrt(min(sd_shapes_dot2(ca), sd_shapes_dot2(cb)));
}
daxa_f32 sd_capped_cone(in daxa_f32vec3 p, in daxa_f32vec3 a, in daxa_f32vec3 b, in daxa_f32 ra, in daxa_f32 rb) {
    daxa_f32 rba = rb - ra;
    daxa_f32 baba = dot(b - a, b - a);
    daxa_f32 papa = dot(p - a, p - a);
    daxa_f32 paba = dot(p - a, b - a) / baba;
    daxa_f32 x = sqrt(papa - paba * paba * baba);
    daxa_f32 cax = max(0.0, x - ((paba < 0.5) ? ra : rb));
    daxa_f32 cay = abs(paba - 0.5) - 0.5;
    daxa_f32 k = rba * rba + baba;
    daxa_f32 f = clamp((rba * (x - ra) + paba * baba) / k, 0.0, 1.0);
    daxa_f32 cbx = x - ra - f * rba;
    daxa_f32 cby = paba - f;
    daxa_f32 s = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;
    return s * sqrt(min(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba));
}
daxa_f32 sd_torus(in daxa_f32vec3 p, in daxa_f32vec2 t) {
    return length(daxa_f32vec2(length(p.xy) - t.x, p.z)) - t.y;
}
daxa_f32 sd_octahedron(in daxa_f32vec3 p, in daxa_f32 s) {
    p = abs(p);
    daxa_f32 m = p.x + p.y + p.z - s;
    daxa_f32vec3 q;
    if (3.0 * p.x < m)
        q = p.xyz;
    else if (3.0 * p.y < m)
        q = p.yzx;
    else if (3.0 * p.z < m)
        q = p.zxy;
    else
        return m * 0.57735027;
    daxa_f32 k = clamp(0.5 * (q.z - q.y + s), 0.0, s);
    return length(daxa_f32vec3(q.x, q.y - s + k, q.z - k));
}
daxa_f32 sd_pyramid(in daxa_f32vec3 p, in daxa_f32 r, in daxa_f32 h) {
    h = h / r;
    p = p / r;
    daxa_f32 m2 = h * h + 0.25;
    p.xy = abs(p.xy);
    p.xy = (p.y > p.x) ? p.yx : p.xy;
    p.xy -= 0.5;
    daxa_f32vec3 q = daxa_f32vec3(p.y, h * p.z - 0.5 * p.x, h * p.x + 0.5 * p.z);
    daxa_f32 s = max(-q.x, 0.0);
    daxa_f32 t = clamp((q.y - 0.5 * p.y) / (m2 + 0.25), 0.0, 1.0);
    daxa_f32 a = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
    daxa_f32 b = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);
    daxa_f32 d2 = min(q.y, -q.x * m2 - q.y * 0.5) > 0.0 ? 0.0 : min(a, b);
    return sqrt((d2 + q.z * q.z) / m2) * sign(max(q.z, -p.z)) * r;
}