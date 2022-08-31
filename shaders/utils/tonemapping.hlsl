#pragma once

#define GAMMA_VALUE 2.2

template <int i>
float3 tonemap_filmic(float3 color);
template <>
float3 tonemap_filmic<1>(float3 color) {
    color = max(color, float3(0, 0, 0));
    color = (color * (6.2 * color + 0.5)) / (color * (6.2 * color + 1.7) + 0.06);
    return color;
}
template <>
float3 tonemap_filmic<-1>(float3 color) {
    color = max(color, float3(0, 0, 0));
    color = (-sqrt(5) * sqrt(701 * color * color - 106 * color + 125) - 85 * color + 25) / (620 * (color - 1));
    return color;
}

template <int i>
float3 tonemap_reinhard(float3 color);
template <>
float3 tonemap_reinhard<1>(float3 color) {
    float3 i = float3(0.2126, 0.7152, 0.0722);
    float l = dot(color, i);
    color *= 1.0 / (1.0 + l);
    color = pow(color, 1.0 / GAMMA_VALUE);
    return color;
}
template <>
float3 tonemap_reinhard<-1>(float3 color) {
    color = pow(color, GAMMA_VALUE);
    float3 t = color;
    float3 i = float3(0.2126, 0.7152, 0.0722);
    // This is a recursive definition. I'm not smart enough to
    // figure out whether there's an analytical solution, so I
    // just do a few iterations and it looks good enough
    //      l = dot(t * (1 + l), i)
    float l = dot(t * (1.0 + dot(t * (1.0 + dot(t * (1.0 + dot(t * (1.0 + i), i)), i)), i)), i);
    color = t * (1.0 + l);
    return color;
}

template <int i>
float3 tonemap_linear(float3 color);
template <>
float3 tonemap_linear<1>(float3 color) {
    return pow(color, 1.0 / GAMMA_VALUE);
}
template <>
float3 tonemap_linear<-1>(float3 color) {
    return pow(color, GAMMA_VALUE);
}

template <int i>
float3 tonemap_none(float3 color) {
    return color;
}

template <int i>
float3 tonemap(float3 color) {
    // tonemap_reinhard
    // tonemap_filmic
    // tonemap_linear
    // tonemap_none
    return tonemap_filmic<i>(color);
}
