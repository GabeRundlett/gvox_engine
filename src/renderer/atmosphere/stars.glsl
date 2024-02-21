#pragma once

#include <renderer/kajiya/inc/math_const.glsl>

#define LAYERS 5.0

// License: Unknown, author: Unknown, found: don't remember
float tanh_approx(float x) {
    //  Found this somewhere on the interwebs
    //  return tanh(x);
    float x2 = x * x;
    return clamp(x * (27.0 + x2) / (27.0 + 9.0 * x2), -1.0, 1.0);
}

// License: MIT OR CC-BY-NC-4.0, author: mercury, found: https://mercury.sexy/hg_sdf/
vec2 mod2(inout vec2 p, vec2 size) {
    vec2 c = floor((p + size * 0.5) / size);
    p = mod(p + size * 0.5, size) - size * 0.5;
    return c;
}

// License: Unknown, author: Unknown, found: don't remember
vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453123);
}

vec3 toSpherical(vec3 p) {
    float r = length(p);
    float t = acos(p.z / r);
    float ph = atan(p.y, p.x);
    return vec3(r, t, ph);
}

// License: CC BY-NC-SA 3.0, author: Stephane Cuillerdier - Aiekick/2015 (twitter:@aiekick), found: https://www.shadertoy.com/view/Mt3GW2
vec3 blackbody(float Temp) {
    vec3 col = vec3(255.);
    col.x = 56100000. * pow(Temp, (-3. / 2.)) + 148.;
    col.y = 100.04 * log(Temp) - 623.6;
    if (Temp > 6500.)
        col.y = 35200000. * pow(Temp, (-3. / 2.)) + 184.;
    col.z = 194.18 * log(Temp) - 1448.6;
    col = clamp(col, 0., 255.) / 255.;
    if (Temp < 1000.)
        col *= Temp / 1000.;
    return col;
}

// https://www.shadertoy.com/view/stBcW1
// License CC0: Stars and galaxy
// Bit of sunday tinkering lead to stars and a galaxy
// Didn't turn out as I envisioned but it turned out to something
// that I liked so sharing it.
vec3 stars(vec3 ro, vec3 rd, vec2 sp, float hh) {
    vec3 col = vec3(0.0);

    const float m = LAYERS;
    hh = tanh_approx(20.0 * hh);

    for (float i = 0.0; i < m; ++i) {
        vec2 pp = sp + 0.5 * i;
        float s = i / (m - 1.0);
        vec2 dim = vec2(mix(0.05, 0.003, s) * M_PI);
        vec2 np = mod2(pp, dim);
        vec2 h = hash2(np + 127.0 + i);
        vec2 o = -1.0 + 2.0 * h;
        float y = sin(sp.x);
        pp += o * dim * 0.5;
        pp.y *= y;
        float l = length(pp);

        float h1 = fract(h.x * 1667.0);
        float h2 = fract(h.x * 1887.0);
        float h3 = fract(h.x * 2997.0);

        vec3 scol = mix(8.0 * h2, 0.25 * h2 * h2, s) * blackbody(mix(3000.0, 22000.0, h1 * h1));

        vec3 ccol = col + exp(-(mix(6000.0, 2000.0, hh) / mix(2.0, 0.25, s)) * max(l - 0.001, 0.0)) * scol;
        col = h3 < y ? ccol : col;
    }

    return col;
}

vec3 get_star_radiance(daxa_BufferPtr(GpuInput) gpu_input, vec3 view_direction) {
    vec3 ro = vec3(0.0, 0.0, 0.0);
    vec2 sp = toSpherical(view_direction.xzy).yz;
    float sf = 0.0;

    return stars(ro, view_direction, sp, sf) * 0.001;
}
