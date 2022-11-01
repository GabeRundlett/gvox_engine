#pragma once

#include <shared/shapes.inl>

#define MAX_SD 10000.0

f32 sd_plane(in f32vec3 p) { return p.z; }
f32 sd_sphere(in f32vec3 p, in f32 r) { return length(p) - r; }
f32 sd_box(in f32vec3 p, in f32vec3 size) {
    f32vec3 d = abs(p) - size;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}
f32 sd_box(in f32vec3 p, in Box box) {
    return sd_box(p - (box.bound_max + box.bound_min) * 0.5, (box.bound_max - box.bound_min) * 0.5);
}
f32 sd_cylinder(in f32vec3 p, in f32 h, in f32 r) {
    f32vec2 d = abs(f32vec2(length(p.xy), p.z)) - f32vec2(h, r);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

f32 sd_triangular_prism(in f32vec3 p, in f32 h, in f32 s) {
    p = p.yzx;
    const f32 k = sqrt(3.0);
    h *= 0.5 * k;
    p.xy /= h;
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0 / k;
    if (p.x + k * p.y > 0.0)
        p.xy = f32vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
    p.x -= clamp(p.x, -2.0, 0.0);
    f32 d1 = length(p.xy) * sign(-p.y) * h;
    f32 d2 = abs(p.z) - s;
    return length(max(f32vec2(d1, d2), 0.0)) + min(max(d1, d2), 0.);
}

f32 sd_capsule(in f32vec3 p, in f32vec3 a, in f32vec3 b, in f32 r) {
    f32vec3 pa = p - a, ba = b - a;
    f32 h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

f32 sd_round_cone(f32vec3 p, f32vec3 a, f32vec3 b, f32 r1, f32 r2) {
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

#if 0
f32 dot2(in f32vec2 v) { return dot(v, v); }
f32 dot2(in f32vec3 v) { return dot(v, v); }
f32 ndot(in f32vec2 a, in f32vec2 b) { return a.x * b.x - a.y * b.y; }

f32 sdPlane(f32vec3 p) {
    return p.y;
}

f32 sdSphere(f32vec3 p, f32 s) {
    return length(p) - s;
}

f32 sdBox(f32vec3 p, f32vec3 b) {
    f32vec3 d = abs(p) - b;
    return 
}

f32 sdBoxFrame(f32vec3 p, f32vec3 b, f32 e) {
    p = abs(p) - b;
    f32vec3 q = abs(p + e) - e;

    return min(min(
                   length(max(f32vec3(p.x, q.y, q.z), 0.0)) + min(max(p.x, max(q.y, q.z)), 0.0),
                   length(max(f32vec3(q.x, p.y, q.z), 0.0)) + min(max(q.x, max(p.y, q.z)), 0.0)),
               length(max(f32vec3(q.x, q.y, p.z), 0.0)) + min(max(q.x, max(q.y, p.z)), 0.0));
}
f32 sdEllipsoid(in f32vec3 p, in f32vec3 r) {
    f32 k0 = length(p / r);
    f32 k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}

f32 sdTorus(f32vec3 p, f32vec2 t) {
    return length(f32vec2(length(p.xz) - t.x, p.y)) - t.y;
}

f32 sdCappedTorus(in f32vec3 p, in f32vec2 sc, in f32 ra, in f32 rb) {
    p.x = abs(p.x);
    f32 k = (sc.y * p.x > sc.x * p.y) ? dot(p.xy, sc) : length(p.xy);
    return sqrt(dot(p, p) + ra * ra - 2.0 * ra * k) - rb;
}

f32 sdHexPrism(f32vec3 p, f32vec2 h) {
    f32vec3 q = abs(p);

    const f32vec3 k = f32vec3(-0.8660254, 0.5, 0.57735);
    p = abs(p);
    p.xy -= 2.0 * min(dot(k.xy, p.xy), 0.0) * k.xy;
    f32vec2 d = f32vec2(
        length(p.xy - f32vec2(clamp(p.x, -k.z * h.x, k.z * h.x), h.x)) * sign(p.y - h.x),
        p.z - h.y);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

f32 sdOctogonPrism(in f32vec3 p, in f32 r, f32 h) {
    const f32vec3 k = f32vec3(-0.9238795325, // sqrt(2+sqrt(2))/2
                        0.3826834323,  // sqrt(2-sqrt(2))/2
                        0.4142135623); // sqrt(2)-1
    // reflections
    p = abs(p);
    p.xy -= 2.0 * min(dot(f32vec2(k.x, k.y), p.xy), 0.0) * f32vec2(k.x, k.y);
    p.xy -= 2.0 * min(dot(f32vec2(-k.x, k.y), p.xy), 0.0) * f32vec2(-k.x, k.y);
    // polygon side
    p.xy -= f32vec2(clamp(p.x, -k.z * r, k.z * r), r);
    f32vec2 d = f32vec2(length(p.xy) * sign(p.y), p.z - h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

f32 sdCapsule(f32vec3 p, f32vec3 a, f32vec3 b, f32 r) {
    f32vec3 pa = p - a, ba = b - a;
    f32 h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

f32 sdRoundCone(in f32vec3 p, in f32 r1, f32 r2, f32 h) {
    f32vec2 q = f32vec2(length(p.xz), p.y);

    f32 b = (r1 - r2) / h;
    f32 a = sqrt(1.0 - b * b);
    f32 k = dot(q, f32vec2(-b, a));

    if (k < 0.0)
        return length(q) - r1;
    if (k > a * h)
        return length(q - f32vec2(0.0, h)) - r2;

    return dot(q, f32vec2(a, b)) - r1;
}

f32 sdRoundCone(f32vec3 p, f32vec3 a, f32vec3 b, f32 r1, f32 r2) {
    // sampling independent computations (only depend on shape)
    f32vec3 ba = b - a;
    f32 l2 = dot(ba, ba);
    f32 rr = r1 - r2;
    f32 a2 = l2 - rr * rr;
    f32 il2 = 1.0 / l2;

    // sampling dependant computations
    f32vec3 pa = p - a;
    f32 y = dot(pa, ba);
    f32 z = y - l2;
    f32 x2 = dot2(pa * l2 - ba * y);
    f32 y2 = y * y * l2;
    f32 z2 = z * z * l2;

    // single square root!
    f32 k = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k)
        return sqrt(x2 + z2) * il2 - r2;
    if (sign(y) * a2 * y2 < k)
        return sqrt(x2 + y2) * il2 - r1;
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}

f32 sdTriPrism(f32vec3 p, f32vec2 h) {
    const f32 k = sqrt(3.0);
    h.x *= 0.5 * k;
    p.xy /= h.x;
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0 / k;
    if (p.x + k * p.y > 0.0)
        p.xy = f32vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
    p.x -= clamp(p.x, -2.0, 0.0);
    f32 d1 = length(p.xy) * sign(-p.y) * h.x;
    f32 d2 = abs(p.z) - h.y;
    return length(max(f32vec2(d1, d2), 0.0)) + min(max(d1, d2), 0.);
}

// vertical
f32 sdCylinder(f32vec3 p, f32vec2 h) {
    f32vec2 d = abs(f32vec2(length(p.xz), p.y)) - h;
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

// arbitrary orientation
f32 sdCylinder(f32vec3 p, f32vec3 a, f32vec3 b, f32 r) {
    f32vec3 pa = p - a;
    f32vec3 ba = b - a;
    f32 baba = dot(ba, ba);
    f32 paba = dot(pa, ba);

    f32 x = length(pa * baba - ba * paba) - r * baba;
    f32 y = abs(paba - baba * 0.5) - baba * 0.5;
    f32 x2 = x * x;
    f32 y2 = y * y * baba;
    f32 d = (max(x, y) < 0.0) ? -min(x2, y2) : (((x > 0.0) ? x2 : 0.0) + ((y > 0.0) ? y2 : 0.0));
    return sign(d) * sqrt(abs(d)) / baba;
}

// vertical
f32 sdCone(in f32vec3 p, in f32vec2 c, f32 h) {
    f32vec2 q = h * f32vec2(c.x, -c.y) / c.y;
    f32vec2 w = f32vec2(length(p.xz), p.y);

    f32vec2 a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
    f32vec2 b = w - q * f32vec2(clamp(w.x / q.x, 0.0, 1.0), 1.0);
    f32 k = sign(q.y);
    f32 d = min(dot(a, a), dot(b, b));
    f32 s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
    return sqrt(d) * sign(s);
}

f32 sdCappedCone(in f32vec3 p, in f32 h, in f32 r1, in f32 r2) {
    f32vec2 q = f32vec2(length(p.xz), p.y);

    f32vec2 k1 = f32vec2(r2, h);
    f32vec2 k2 = f32vec2(r2 - r1, 2.0 * h);
    f32vec2 ca = f32vec2(q.x - min(q.x, (q.y < 0.0) ? r1 : r2), abs(q.y) - h);
    f32vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot2(k2), 0.0, 1.0);
    f32 s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return s * sqrt(min(dot2(ca), dot2(cb)));
}

f32 sdCappedCone(f32vec3 p, f32vec3 a, f32vec3 b, f32 ra, f32 rb) {
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

    return s * sqrt(min(cax * cax + cay * cay * baba,
                        cbx * cbx + cby * cby * baba));
}

// c is the sin/cos of the desired cone angle
f32 sdSolidAngle(f32vec3 pos, f32vec2 c, f32 ra) {
    f32vec2 p = f32vec2(length(pos.xz), pos.y);
    f32 l = length(p) - ra;
    f32 m = length(p - c * clamp(dot(p, c), 0.0, ra));
    return max(l, m * sign(c.y * p.x - c.x * p.y));
}

f32 sdOctahedron(f32vec3 p, f32 s) {
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

f32 sdPyramid(in f32vec3 p, in f32 h) {
    f32 m2 = h * h + 0.25;

    // symmetry
    p.xz = abs(p.xz);
    p.xz = (p.z > p.x) ? p.zx : p.xz;
    p.xz -= 0.5;

    // project into face plane (2D)
    f32vec3 q = f32vec3(p.z, h * p.y - 0.5 * p.x, h * p.x + 0.5 * p.y);

    f32 s = max(-q.x, 0.0);
    f32 t = clamp((q.y - 0.5 * p.z) / (m2 + 0.25), 0.0, 1.0);

    f32 a = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
    f32 b = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);

    f32 d2 = min(q.y, -q.x * m2 - q.y * 0.5) > 0.0 ? 0.0 : min(a, b);

    // recover 3D and scale, and add sign
    return sqrt((d2 + q.z * q.z) / m2) * sign(max(q.z, -p.y));
    ;
}

// la,lb=semi axis, h=height, ra=corner
f32 sdRhombus(f32vec3 p, f32 la, f32 lb, f32 h, f32 ra) {
    p = abs(p);
    f32vec2 b = f32vec2(la, lb);
    f32 f = clamp((ndot(b, b - 2.0 * p.xz)) / dot(b, b), -1.0, 1.0);
    f32vec2 q = f32vec2(length(p.xz - 0.5 * b * f32vec2(1.0 - f, 1.0 + f)) * sign(p.x * b.y + p.z * b.x - b.x * b.y) - ra, p.y - h);
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0));
}

f32 sdHorseshoe(in f32vec3 p, in f32vec2 c, in f32 r, in f32 le, f32vec2 w) {
    p.x = abs(p.x);
    f32 l = length(p.xy);
    p.xy = mat2(-c.x, c.y,
                c.y, c.x) *
           p.xy;
    p.xy = f32vec2((p.y > 0.0 || p.x > 0.0) ? p.x : l * sign(-c.x),
                (p.x > 0.0) ? p.y : l);
    p.xy = f32vec2(p.x, abs(p.y - r)) - f32vec2(le, 0.0);

    f32vec2 q = f32vec2(length(max(p.xy, 0.0)) + min(0.0, max(p.x, p.y)), p.z);
    f32vec2 d = abs(q) - w;
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

f32 sdU(in f32vec3 p, in f32 r, in f32 le, f32vec2 w) {
    p.x = (p.y > 0.0) ? abs(p.x) : length(p.xy);
    p.x = abs(p.x - r);
    p.y = p.y - le;
    f32 k = max(p.x, p.y);
    f32vec2 q = f32vec2((k < 0.0) ? -k : length(max(p.xy, 0.0)), abs(p.z)) - w;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
}

f32vec2 opU(f32vec2 d1, f32vec2 d2) {
    return (d1.x < d2.x) ? d1 : d2;
}

f32vec2 map(in f32vec3 pos) {
    f32vec2 res = f32vec2(pos.y, 0.0);

    // bounding box
    if (sdBox(pos - f32vec3(-2.0, 0.3, 0.25), f32vec3(0.3, 0.3, 1.0)) < res.x) {
        res = opU(res, f32vec2(sdSphere(pos - f32vec3(-2.0, 0.25, 0.0), 0.25), 26.9));
        res = opU(res, f32vec2(sdRhombus((pos - f32vec3(-2.0, 0.25, 1.0)).xzy, 0.15, 0.25, 0.04, 0.08), 17.0));
    }

    // bounding box
    if (sdBox(pos - f32vec3(0.0, 0.3, -1.0), f32vec3(0.35, 0.3, 2.5)) < res.x) {
        res = opU(res, f32vec2(sdCappedTorus((pos - f32vec3(0.0, 0.30, 1.0)) * f32vec3(1, -1, 1), f32vec2(0.866025, -0.5), 0.25, 0.05), 25.0));
        res = opU(res, f32vec2(sdBoxFrame(pos - f32vec3(0.0, 0.25, 0.0), f32vec3(0.3, 0.25, 0.2), 0.025), 16.9));
        res = opU(res, f32vec2(sdCone(pos - f32vec3(0.0, 0.45, -1.0), f32vec2(0.6, 0.8), 0.45), 55.0));
        res = opU(res, f32vec2(sdCappedCone(pos - f32vec3(0.0, 0.25, -2.0), 0.25, 0.25, 0.1), 13.67));
        res = opU(res, f32vec2(sdSolidAngle(pos - f32vec3(0.0, 0.00, -3.0), f32vec2(3, 4) / 5.0, 0.4), 49.13));
    }

    // bounding box
    if (sdBox(pos - f32vec3(1.0, 0.3, -1.0), f32vec3(0.35, 0.3, 2.5)) < res.x) {
        res = opU(res, f32vec2(sdTorus((pos - f32vec3(1.0, 0.30, 1.0)).xzy, f32vec2(0.25, 0.05)), 7.1));
        res = opU(res, f32vec2(sdBox(pos - f32vec3(1.0, 0.25, 0.0), f32vec3(0.3, 0.25, 0.1)), 3.0));
        res = opU(res, f32vec2(sdCapsule(pos - f32vec3(1.0, 0.00, -1.0), f32vec3(-0.1, 0.1, -0.1), f32vec3(0.2, 0.4, 0.2), 0.1), 31.9));
        res = opU(res, f32vec2(sdCylinder(pos - f32vec3(1.0, 0.25, -2.0), f32vec2(0.15, 0.25)), 8.0));
        res = opU(res, f32vec2(sdHexPrism(pos - f32vec3(1.0, 0.2, -3.0), f32vec2(0.2, 0.05)), 18.4));
    }

    // bounding box
    if (sdBox(pos - f32vec3(-1.0, 0.35, -1.0), f32vec3(0.35, 0.35, 2.5)) < res.x) {
        res = opU(res, f32vec2(sdPyramid(pos - f32vec3(-1.0, -0.6, -3.0), 1.0), 13.56));
        res = opU(res, f32vec2(sdOctahedron(pos - f32vec3(-1.0, 0.15, -2.0), 0.35), 23.56));
        res = opU(res, f32vec2(sdTriPrism(pos - f32vec3(-1.0, 0.15, -1.0), f32vec2(0.3, 0.05)), 43.5));
        res = opU(res, f32vec2(sdEllipsoid(pos - f32vec3(-1.0, 0.25, 0.0), f32vec3(0.2, 0.25, 0.05)), 43.17));
        res = opU(res, f32vec2(sdHorseshoe(pos - f32vec3(-1.0, 0.25, 1.0), f32vec2(cos(1.3), sin(1.3)), 0.2, 0.3, f32vec2(0.03, 0.08)), 11.5));
    }

    // bounding box
    if (sdBox(pos - f32vec3(2.0, 0.3, -1.0), f32vec3(0.35, 0.3, 2.5)) < res.x) {
        res = opU(res, f32vec2(sdOctogonPrism(pos - f32vec3(2.0, 0.2, -3.0), 0.2, 0.05), 51.8));
        res = opU(res, f32vec2(sdCylinder(pos - f32vec3(2.0, 0.14, -2.0), f32vec3(0.1, -0.1, 0.0), f32vec3(-0.2, 0.35, 0.1), 0.08), 31.2));
        res = opU(res, f32vec2(sdCappedCone(pos - f32vec3(2.0, 0.09, -1.0), f32vec3(0.1, 0.0, 0.0), f32vec3(-0.2, 0.40, 0.1), 0.15, 0.05), 46.1));
        res = opU(res, f32vec2(sdRoundCone(pos - f32vec3(2.0, 0.15, 0.0), f32vec3(0.1, 0.0, 0.0), f32vec3(-0.1, 0.35, 0.1), 0.15, 0.05), 51.7));
        res = opU(res, f32vec2(sdRoundCone(pos - f32vec3(2.0, 0.20, 1.0), 0.2, 0.1, 0.3), 37.0));
    }

    return res;
}
#endif
