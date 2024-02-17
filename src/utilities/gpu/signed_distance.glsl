#pragma once

float sd_shapes_dot2(in vec2 v) { return dot(v, v); }
float sd_shapes_dot2(in vec3 v) { return dot(v, v); }
float sd_shapes_ndot(in vec2 a, in vec2 b) { return a.x * b.x - a.y * b.y; }

// Operators

// These are safe for min/max operations!
float sd_set(float a, float b) {
    return b;
}
vec4 sd_set(vec4 a, vec4 b) {
    return b;
}
float sd_add(float a, float b) {
    return (a + b);
}
float sd_union(in float a, in float b) {
    return min(a, b);
}

// These are either unsafe or unknown for min/max operations
float sd_smooth_union(in float a, in float b, in float k) {
    float h = clamp(0.5 + 0.5 * (a - b) / k, 0.0, 1.0);
    return mix(a, b, h) - k * h * (1.0 - h);
}
float sd_intersection(in float a, in float b) {
    return max(a, b);
}
float sd_smooth_intersection(in float a, in float b, in float k) {
    return sd_smooth_union(a, b, -k);
}
float sd_difference(in float a, in float b) {
    return sd_intersection(a, -b);
}
float sd_smooth_difference(in float a, in float b, in float k) {
    return sd_smooth_intersection(a, -b, k);
}
float sd_mul(float a, float b) {
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

float sd_plane_x(in vec3 p) {
    return p.x;
}
float sd_plane_y(in vec3 p) {
    return p.y;
}
float sd_plane_z(in vec3 p) {
    return p.z;
}
float sd_sphere(in vec3 p, in float r) {
    return length(p) - r;
}
float sd_ellipsoid(in vec3 p, in vec3 r) {
    float k0 = length(p / r);
    float k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}
float sd_box(in vec3 p, in vec3 size) {
    vec3 d = abs(p) - size;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}
float sd_box(in vec3 p, in BoundingBox box) {
    return sd_box(p - (box.bound_max + box.bound_min) * 0.5, (box.bound_max - box.bound_min) * 0.5);
}
float sd_box_frame(in vec3 p, in vec3 b, in float e) {
    p = abs(p) - b;
    vec3 q = abs(p + e) - e;
    return min(
        min(length(max(vec3(p.x, q.y, q.z), 0.0)) + min(max(p.x, max(q.y, q.z)), 0.0),
            length(max(vec3(q.x, p.y, q.z), 0.0)) + min(max(q.x, max(p.y, q.z)), 0.0)),
        length(max(vec3(q.x, q.y, p.z), 0.0)) + min(max(q.x, max(q.y, p.z)), 0.0));
}
float sd_box_frame(in vec3 p, in BoundingBox box, in float e) {
    return sd_box_frame(p - (box.bound_max + box.bound_min) * 0.5, (box.bound_max - box.bound_min) * 0.5, e);
}
float sd_cylinder(in vec3 p, in float r, in float h) {
    vec2 d = abs(vec2(length(p.xy), p.z)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}
float sd_cylinder(vec3 p, vec3 a, vec3 b, float r) {
    vec3 ba = b - a;
    vec3 pa = p - a;
    float baba = dot(ba, ba);
    float paba = dot(pa, ba);
    float x = length(pa * baba - ba * paba) - r * baba;
    float y = abs(paba - baba * 0.5) - baba * 0.5;
    float x2 = x * x;
    float y2 = y * y * baba;
    float d = (max(x, y) < 0.0) ? -min(x2, y2) : (((x > 0.0) ? x2 : 0.0) + ((y > 0.0) ? y2 : 0.0));
    return sign(d) * sqrt(abs(d)) / baba;
}
float sd_triangular_prism(in vec3 p, in float r, in float h) {
    const float k = sqrt(3.0);
    h *= 0.5 * k;
    p.xy /= h;
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0 / k;
    if (p.x + k * p.y > 0.0)
        p.xy = vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
    p.x -= clamp(p.x, -2.0, 0.0);
    float d1 = length(p.xy) * sign(-p.y) * h;
    float d2 = abs(p.z) - r;
    return length(max(vec2(d1, d2), 0.0)) + min(max(d1, d2), 0.);
}
float sd_hexagonal_prism(in vec3 p, in float r, in float h) {
    vec3 q = abs(p);
    const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
    p = abs(p);
    p.xy -= 2.0 * min(dot(k.xy, p.xy), 0.0) * k.xy;
    vec2 d = vec2(
        length(p.xy - vec2(clamp(p.x, -k.z * h, k.z * h), h)) * sign(p.y - h),
        p.z - r);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}
float sd_octagonal_prism(in vec3 p, in float r, in float h) {
    const vec3 k = vec3(-0.9238795325, 0.3826834323, 0.4142135623);
    p = abs(p);
    p.xy -= 2.0 * min(dot(vec2(k.x, k.y), p.xy), 0.0) * vec2(k.x, k.y);
    p.xy -= 2.0 * min(dot(vec2(-k.x, k.y), p.xy), 0.0) * vec2(-k.x, k.y);
    p.xy -= vec2(clamp(p.x, -k.z * r, k.z * r), r);
    vec2 d = vec2(length(p.xy) * sign(p.y), p.z - h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}
float sd_capsule(in vec3 p, in vec3 a, in vec3 b, in float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}
float sd_cone(in vec3 p, in float c, in float h) {
    vec2 q = h * vec2(c, -1);
    vec2 w = vec2(length(p.xy), p.z);
    vec2 a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
    vec2 b = w - q * vec2(clamp(w.x / q.x, 0.0, 1.0), 1.0);
    float k = sign(q.y);
    float d = min(dot(a, a), dot(b, b));
    float s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
    return sqrt(d) * sign(s);
}
float sd_round_cone(in vec3 p, in float r1, in float r2, in float h) {
    vec2 q = vec2(length(p.xy), p.z);
    float b = (r1 - r2) / h;
    float a = sqrt(1.0 - b * b);
    float k = dot(q, vec2(-b, a));
    if (k < 0.0)
        return length(q) - r1;
    if (k > a * h)
        return length(q - vec2(0.0, h)) - r2;
    return dot(q, vec2(a, b)) - r1;
}
float sd_round_cone(in vec3 p, in vec3 a, in vec3 b, in float r1, in float r2) {
    vec3 ba = b - a;
    float l2 = dot(ba, ba);
    float rr = r1 - r2;
    float a2 = l2 - rr * rr;
    float il2 = 1.0 / l2;
    vec3 pa = p - a;
    float y = dot(pa, ba);
    float z = y - l2;
    vec3 xp = pa * l2 - ba * y;
    float x2 = dot(xp, xp);
    float y2 = y * y * l2;
    float z2 = z * z * l2;
    float k = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k)
        return sqrt(x2 + z2) * il2 - r2;
    if (sign(y) * a2 * y2 < k)
        return sqrt(x2 + y2) * il2 - r1;
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}
float sd_capped_cone(in vec3 p, in float r1, in float r2, in float h) {
    vec2 q = vec2(length(p.xy), p.z);
    vec2 k1 = vec2(r2, h);
    vec2 k2 = vec2(r2 - r1, 2.0 * h);
    vec2 ca = vec2(q.x - min(q.x, (q.y < 0.0) ? r1 : r2), abs(q.y) - h);
    vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / sd_shapes_dot2(k2), 0.0, 1.0);
    float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return s * sqrt(min(sd_shapes_dot2(ca), sd_shapes_dot2(cb)));
}
float sd_capped_cone(in vec3 p, in vec3 a, in vec3 b, in float ra, in float rb) {
    float rba = rb - ra;
    float baba = dot(b - a, b - a);
    float papa = dot(p - a, p - a);
    float paba = dot(p - a, b - a) / baba;
    float x = sqrt(papa - paba * paba * baba);
    float cax = max(0.0, x - ((paba < 0.5) ? ra : rb));
    float cay = abs(paba - 0.5) - 0.5;
    float k = rba * rba + baba;
    float f = clamp((rba * (x - ra) + paba * baba) / k, 0.0, 1.0);
    float cbx = x - ra - f * rba;
    float cby = paba - f;
    float s = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;
    return s * sqrt(min(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba));
}
float sd_torus(in vec3 p, in vec2 t) {
    return length(vec2(length(p.xy) - t.x, p.z)) - t.y;
}
float sd_octahedron(in vec3 p, in float s) {
    p = abs(p);
    float m = p.x + p.y + p.z - s;
    vec3 q;
    if (3.0 * p.x < m)
        q = p.xyz;
    else if (3.0 * p.y < m)
        q = p.yzx;
    else if (3.0 * p.z < m)
        q = p.zxy;
    else
        return m * 0.57735027;
    float k = clamp(0.5 * (q.z - q.y + s), 0.0, s);
    return length(vec3(q.x, q.y - s + k, q.z - k));
}
float sd_pyramid(in vec3 p, in float r, in float h) {
    h = h / r;
    p = p / r;
    float m2 = h * h + 0.25;
    p.xy = abs(p.xy);
    p.xy = (p.y > p.x) ? p.yx : p.xy;
    p.xy -= 0.5;
    vec3 q = vec3(p.y, h * p.z - 0.5 * p.x, h * p.x + 0.5 * p.z);
    float s = max(-q.x, 0.0);
    float t = clamp((q.y - 0.5 * p.y) / (m2 + 0.25), 0.0, 1.0);
    float a = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
    float b = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);
    float d2 = min(q.y, -q.x * m2 - q.y * 0.5) > 0.0 ? 0.0 : min(a, b);
    return sqrt((d2 + q.z * q.z) / m2) * sign(max(q.z, -p.z)) * r;
}