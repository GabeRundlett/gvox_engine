#pragma once

#include "brdf.glsl"

struct SpecularBrdfEnergyPreservation {
    vec3 preintegrated_reflection;
    vec3 preintegrated_reflection_mult;
    vec3 preintegrated_transmission_fraction;
    float valid_sample_fraction;
};

vec3 SpecularBrdfEnergyPreservation_sample_fg_lut(float ndotv, float roughness) {
    vec2 uv = vec2(ndotv, roughness) * BRDF_FG_LUT_UV_SCALE + BRDF_FG_LUT_UV_BIAS;
    // TODO
    // return bindless_textures[BINDLESS_LUT_BRDF_FG].SampleLevel(sampler_lnc, uv, 0).xyz;
    return vec3(0.5);
}

SpecularBrdfEnergyPreservation SpecularBrdfEnergyPreservation_from_brdf_ndotv(SpecularBrdf brdf, float ndotv) {
    const float roughness = brdf.roughness;
    const vec3 specular_albedo = brdf.albedo;

    vec3 fg = SpecularBrdfEnergyPreservation_sample_fg_lut(ndotv, roughness);
    vec3 single_scatter = specular_albedo * fg.x + fg.y;

    SpecularBrdfEnergyPreservation res;
    res.valid_sample_fraction = fg.z;

    // In retrospect, this is just a special case of "Eﬀicient Rendering of Layered Materials using an
    // Atomic Decomposition with Statistical Operators" by Laurent Belcour, specifically section 5.1:
    // https://hal.archives-ouvertes.fr/hal-01785457/document

    float e_ss = fg.x + fg.y;
    vec3 f_ss = single_scatter / e_ss;
    // Ad-hoc shift towards F90 for subsequent bounces
    vec3 f_ss_tail = mix(f_ss, vec3(1.0), 0.4);
    vec3 bounce_radiance = (1.0 - e_ss) * f_ss_tail;
    vec3 mult = 1.0 + bounce_radiance / (1.0 - bounce_radiance);

    res.preintegrated_reflection = single_scatter * mult;
    res.preintegrated_reflection_mult = mult;
    res.preintegrated_transmission_fraction = 1 - res.preintegrated_reflection;

    return res;
}