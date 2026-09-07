// Generates the split-sum specular BRDF "FG" lookup table, once at start-up.
//
// Port of kajiya's assets/shaders/lut/brdf_fg.hlsl. The table is indexed by
// (ndotv, roughness) and holds:
//   .x  the albedo-scaled term (F)
//   .y  the albedo-independent bias term (G)
//   .z  the fraction of samples that produced a valid wi
// so that `specular_albedo * fg.x + fg.y` is the single-scatter reflectance.
//
// The destination is the global g_brdf_fg_lut_tex, so this shader needs no
// task-head attachment of its own -- the task graph only has to know that the
// dispatch writes the image. See SpecularBrdfEnergyPreservation_sample_fg_lut
// in inc/brdf_lut.glsl for the consumer.

#include <daxa/daxa.inl>
#include <renderer/globals.glsl>
#include <renderer/kajiya/brdf_fg_lut.inl>
#include <renderer/kajiya/inc/brdf.glsl>
#include <renderer/kajiya/inc/quasi_random.glsl>

vec3 integrate_brdf(float roughness, float ndotv) {
    vec3 wo = vec3(sqrt(1.0 - ndotv * ndotv), 0, ndotv);

    float a = 0;
    float b = 0;

    SpecularBrdf brdf_a;
    brdf_a.roughness = roughness;
    brdf_a.albedo = vec3(1.0);

    SpecularBrdf brdf_b = brdf_a;
    brdf_b.albedo = vec3(0.0);

    // TODO: consider splitting into its own LUT, as hardly anything needs this.
    float valid = 0;

    const uint num_samples = BRDF_FG_LUT_SAMPLE_COUNT;
    for (uint i = 0; i < num_samples; ++i) {
        vec2 urand = hammersley(i, num_samples);
        BrdfSample v_a = sample_brdf(brdf_a, wo, urand);

        if (is_valid(v_a)) {
            BrdfValue v_b = evaluate(brdf_b, wo, v_a.wi);

            a += (v_a.value_over_pdf.x - v_b.value_over_pdf.x);
            b += v_b.value_over_pdf.x;
            valid += 1;
        }
    }

    return vec3(a, b, valid) / float(num_samples);
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    uvec2 pix = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pix, BRDF_FG_LUT_DIMS))) {
        return;
    }

    float ndotv = (float(pix.x) / (BRDF_FG_LUT_DIMS.x - 1.0)) * (1.0 - 1e-3) + 1e-3;
    float roughness = max(1e-5, float(pix.y) / (BRDF_FG_LUT_DIMS.y - 1.0));

    imageStore(daxa_image2D(g_brdf_fg_lut_tex), ivec2(pix), vec4(integrate_brdf(roughness, ndotv), 1.0));
}
