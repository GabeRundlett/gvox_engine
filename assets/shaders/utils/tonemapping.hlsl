#pragma once

template <int i>
float3 filmic(float3 color);

template <>
float3 filmic<1>(float3 color) {
    color = max(color, float3(0, 0, 0));
    color = (color * (6.2 * color + 0.5)) / (color * (6.2 * color + 1.7) + 0.06);
    return color;
}
template <>
float3 filmic<-1>(float3 color) {
    color = max(color, float3(0, 0, 0));
    color = (-sqrt(5) * sqrt(701 * color * color - 106 * color + 125) - 85 * color + 25) / (620 * (color - 1));
    return color;
}

#define GAMMA_VALUE 2.2
template <int i>
float3 gamma(float3 color);
template <>
float3 gamma<1>(float3 color) {
    return pow(color, 1.0 / GAMMA_VALUE);
}
template <>
float3 gamma<-1>(float3 color) {
    return pow(color, GAMMA_VALUE);
}

template <int i>
float3 tonemap_none(float3 color) {
    return color;
}

template <int i>
float3 tonemap(float3 color) {
    return tonemap_none<i>(color);
}
