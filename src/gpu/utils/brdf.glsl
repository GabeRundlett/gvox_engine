#pragma once

#include <utils/math.glsl>

const uvec2 BRDF_FG_LUT_DIMS = uvec2(64, 64);
const vec2 BRDF_FG_LUT_UV_SCALE = (BRDF_FG_LUT_DIMS - 1.0) / BRDF_FG_LUT_DIMS;
const vec2 BRDF_FG_LUT_UV_BIAS = vec2(0.5) / BRDF_FG_LUT_DIMS;

const float BRDF_SAMPLING_MIN_COS = 1e-5;

#define USE_GGX_VNDF_SAMPLING 1
#define USE_GGX_CORRELATED_MASKING 1

// Defined wrt the projected solid angle metric
struct BrdfValue {
    vec3 value_over_pdf;
    vec3 value;
    float pdf;

    vec3 transmission_fraction;
};

BrdfValue BrdfValue_invalid() {
    BrdfValue res;
    res.value_over_pdf = vec3(0.0);
    res.pdf = 0.0;
    res.transmission_fraction = vec3(0.0);
    return res;
}

// Defined wrt the projected solid angle metric
struct BrdfSample {
    vec3 value_over_pdf;
    vec3 value;
    float pdf;

    vec3 transmission_fraction;

    vec3 wi;

    // For filtering / firefly suppression
    float approx_roughness;
};

BrdfSample BrdfSample_invalid() {
    BrdfSample res;
    res.value_over_pdf = vec3(0.0);
    res.pdf = 0.0;
    res.wi = vec3(0.0, 0.0, -1.0);
    res.transmission_fraction = vec3(0.0);
    res.approx_roughness = 0;
    return res;
}
bool is_valid(BrdfSample self) {
    return self.wi.z > 1e-6;
}

struct DiffuseBrdf {
    vec3 albedo;
    // vec3 emission;
};

BrdfSample sample_brdf(DiffuseBrdf self, vec3 _wo, vec2 urand) {
    float phi = urand.x * M_TAU;
    float cos_theta = sqrt(max(0.0, 1.0 - urand.y));
    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));

    BrdfSample res;
    float sin_phi = sin(phi);
    float cos_phi = cos(phi);

    res.wi = vec3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);
    res.pdf = M_FRAC_1_PI;
    res.value_over_pdf = self.albedo;
    res.value = res.value_over_pdf * res.pdf;
    res.transmission_fraction = vec3(0.0);
    res.approx_roughness = 1.0;

    return res;
}

BrdfValue evaluate(DiffuseBrdf self, vec3 _wo, vec3 wi) {
    BrdfValue res;
    res.pdf = select(wi.z > 0.0, M_FRAC_1_PI, 0.0);
    res.value_over_pdf = select(bvec3(wi.z > 0.0), self.albedo, vec3(0.0));
    res.value = res.value_over_pdf * res.pdf;
    res.transmission_fraction = vec3(0.0);
    return res;
}

vec2 wi_to_primary_sample_space(DiffuseBrdf self, vec3 wi) {
    const float cos_theta = wi.z;
    // cos_theta = sqrt(max(0.0, 1.0 - urand.y));
    // cos_theta * cos_theta = 1.0 - urand.y
    // urand.y = 1.0 - cos_theta * cos_theta
    const float y = saturate(1.0 - cos_theta * cos_theta);
    const float x = fract(atan(wi.y, wi.x) / M_TAU);
    return vec2(x, y);
}

vec3 eval_fresnel_schlick(vec3 f0, vec3 f90, float cos_theta) {
    return mix(f0, f90, pow(max(0.0, 1.0 - cos_theta), 5));
}

struct NdfSample {
    vec3 m;
    float pdf;
};

struct SmithShadowingMasking {
    float g;
    float g_over_g1_wo;
};

float SmithShadowingMasking_g_smith_ggx_correlated(float ndotv, float ndotl, float a2) {
    float lambda_v = ndotl * sqrt((-ndotv * a2 + ndotv) * ndotv + a2);
    float lambda_l = ndotv * sqrt((-ndotl * a2 + ndotl) * ndotl + a2);

    return 2.0 * ndotl * ndotv / (lambda_v + lambda_l);
}

float SmithShadowingMasking_g_smith_ggx1(float ndotv, float a2) {
    float tan2_v = (1.0 - ndotv * ndotv) / (ndotv * ndotv);
    return 2.0 / (1.0 + sqrt(1.0 + a2 * tan2_v));
}

float SmithShadowingMasking_g_smith_ggx(float ndotv, float ndotl, float a2) {
#if USE_GGX_CORRELATED_MASKING
    return SmithShadowingMasking_g_smith_ggx_correlated(ndotv, ndotl, a2);
#else
    return SmithShadowingMasking_g_smith_ggx1(ndotl, a2) * SmithShadowingMasking_g_smith_ggx1(ndotv, a2);
#endif
}

SmithShadowingMasking SmithShadowingMasking_eval(float ndotv, float ndotl, float a2) {
    SmithShadowingMasking res;
#if USE_GGX_CORRELATED_MASKING
    res.g = SmithShadowingMasking_g_smith_ggx_correlated(ndotv, ndotl, a2);
    res.g_over_g1_wo = res.g / SmithShadowingMasking_g_smith_ggx1(ndotv, a2);
#else
    res.g = SmithShadowingMasking_g_smith_ggx1(ndotl, a2) * SmithShadowingMasking_g_smith_ggx1(ndotv, a2);
    res.g_over_g1_wo = SmithShadowingMasking_g_smith_ggx1(ndotl, a2);
#endif
    return res;
}

struct SpecularBrdf {
    float roughness;
    vec3 albedo;
    // vec3 emission;
};

float SpecularBrdf_ggx_ndf(float a2, float cos_theta) {
    float denom_sqrt = cos_theta * cos_theta * (a2 - 1.0) + 1.0;
    return a2 / (M_PI * denom_sqrt * denom_sqrt);
}

// Like the GGX NDF, but scaled to peak at 1.0. Never _quite_ reaches zero.
float SpecularBrdf_ggx_ndf_0_1(float a2, float cos_theta) {
    float denom_sqrt = cos_theta * cos_theta * (a2 - 1.0) + 1.0;
    return a2 * a2 / (denom_sqrt * denom_sqrt);
}

float SpecularBrdf_pdf_ggx(float a2, float cos_theta) {
    return SpecularBrdf_ggx_ndf(a2, cos_theta) * cos_theta;
}

float SpecularBrdf_pdf_ggx_vn(float a2, vec3 wo, vec3 h) {
    float g1 = SmithShadowingMasking_g_smith_ggx1(wo.z, a2);
    float d = SpecularBrdf_ggx_ndf(a2, h.z);
    return g1 * d * max(0.f, dot(wo, h)) / wo.z;
}

NdfSample sample_ndf(inout SpecularBrdf self, vec2 urand) {
    const float a2 = self.roughness * self.roughness;

    const float cos2_theta = (1 - urand.x) / (1 - urand.x + a2 * urand.x);
    const float cos_theta = sqrt(cos2_theta);
    const float phi = M_TAU * urand.y;

    const float sin_theta = sqrt(max(0.0, 1.0 - cos2_theta));

    NdfSample res;
    res.m = vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    res.pdf = SpecularBrdf_pdf_ggx(a2, cos_theta);

    return res;
}

// From https://github.com/NVIDIAGameWorks/Falcor/blob/c0729e806045731d71cfaae9d31a992ac62070e7/Source/Falcor/Experimental/Scene/Material/Microfacet.slang
// https://jcgt.org/published/0007/04/01/paper.pdf
NdfSample sample_vndf(inout SpecularBrdf self, float alpha, vec3 wo, vec2 urand) {
    float alpha_x = alpha, alpha_y = alpha;
    float a2 = alpha_x * alpha_y;

    // Transform the view vector to the hemisphere configuration.
    vec3 Vh = normalize(vec3(alpha_x * wo.x, alpha_y * wo.y, wo.z));

    // Construct orthonormal basis (Vh,T1,T2).
    vec3 T1 = select(bvec3(Vh.z < 0.9999f), normalize(cross(vec3(0, 0, 1), Vh)), vec3(1, 0, 0)); // TODO: fp32 precision
    vec3 T2 = cross(Vh, T1);

    // Parameterization of the projected area of the hemisphere.
    float r = sqrt(urand.x);
    float phi = (2.f * M_PI) * urand.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.f + Vh.z);
    t2 = (1.f - s) * sqrt(1.f - t1 * t1) + s * t2;

    // Reproject onto hemisphere.
    vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.f, 1.f - t1 * t1 - t2 * t2)) * Vh;

    // Transform the normal back to the ellipsoid configuration. This is our half vector.
    vec3 h = normalize(vec3(alpha_x * Nh.x, alpha_y * Nh.y, max(0.f, Nh.z)));
    float pdf = SpecularBrdf_pdf_ggx_vn(a2, wo, h);

    NdfSample res;
    res.m = h;
    res.pdf = pdf;
    return res;
}

BrdfSample sample_brdf(inout SpecularBrdf self, vec3 wo, vec2 urand) {
#if USE_GGX_VNDF_SAMPLING
    NdfSample ndf_sample = sample_vndf(self, self.roughness, wo, urand);
#else
    NdfSample ndf_sample = sample_ndf(self, urand);
#endif

    const vec3 wi = reflect(-wo, ndf_sample.m);

    if (ndf_sample.m.z <= BRDF_SAMPLING_MIN_COS || wi.z <= BRDF_SAMPLING_MIN_COS || wo.z <= BRDF_SAMPLING_MIN_COS) {
        return BrdfSample_invalid();
    }

    // Change of variables from the half-direction space to regular lighting geometry.
    const float jacobian = 1.0 / (4.0 * dot(wi, ndf_sample.m));

    const vec3 fresnel = eval_fresnel_schlick(self.albedo, vec3(1.0), dot(ndf_sample.m, wi));
    const float a2 = self.roughness * self.roughness;
    const float cos_theta = ndf_sample.m.z;

    SmithShadowingMasking shadowing_masking = SmithShadowingMasking_eval(wo.z, wi.z, a2);

    BrdfSample res;
    res.pdf = ndf_sample.pdf * jacobian / wi.z;
    res.wi = wi;
    res.transmission_fraction = vec3(1) - fresnel;
    res.approx_roughness = self.roughness;

#if USE_GGX_VNDF_SAMPLING
    res.value_over_pdf =
        fresnel * shadowing_masking.g_over_g1_wo;
#else
    res.value_over_pdf =
        fresnel / (cos_theta * jacobian) * shadowing_masking.g / (4 * wo.z);
#endif

    res.value =
        fresnel * shadowing_masking.g * SpecularBrdf_ggx_ndf(a2, cos_theta) / (4 * wo.z * wi.z);

    return res;
}

BrdfValue evaluate(SpecularBrdf self, vec3 wo, vec3 wi) {
    if (wi.z <= 0.0 || wo.z <= 0.0) {
        return BrdfValue_invalid();
    }

    const float a2 = self.roughness * self.roughness;

    const vec3 m = normalize(wo + wi);

    const float cos_theta = m.z;

#if USE_GGX_VNDF_SAMPLING
    const float pdf_h = SpecularBrdf_pdf_ggx_vn(a2, wo, m);
#else
    const float pdf_h = SpecularBrdf_pdf_ggx(a2, cos_theta);
#endif

    const float jacobian = 1.0 / (4.0 * dot(wi, m));

    const vec3 fresnel = eval_fresnel_schlick(self.albedo, vec3(1.0), dot(m, wi));

    SmithShadowingMasking shadowing_masking = SmithShadowingMasking_eval(wo.z, wi.z, a2);

    BrdfValue res;
    res.pdf = pdf_h * jacobian / wi.z;
    res.transmission_fraction = 1.0.xxx - fresnel;

#if USE_GGX_VNDF_SAMPLING
    res.value_over_pdf =
        fresnel * shadowing_masking.g_over_g1_wo;
#else
    res.value_over_pdf =
        fresnel / (cos_theta * jacobian) * shadowing_masking.g / (4 * wo.z);
#endif

    res.value =
        fresnel * shadowing_masking.g * SpecularBrdf_ggx_ndf(a2, cos_theta) / (4 * wo.z * wi.z);

    return res;
}

// https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
vec3 specular_dominant_direction(vec3 n, vec3 v, float roughness) {
    vec3 r = reflect(-v, n);
    float f = (1.0 - roughness) * (sqrt(1.0 - roughness) + roughness);
    return normalize(mix(n, r, f));
}
