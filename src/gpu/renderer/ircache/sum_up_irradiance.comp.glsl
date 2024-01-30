#include <shared/app.inl>
#include <utils/normal.glsl>
// #include <utils/frame_constants.glsl>
#include <utils/sh.glsl>
#include <utils/quasi_random.glsl>
#include <utils/reservoir.glsl>
#include "ircache_constants.glsl"
#include "ircache_sampler_common.inc.glsl"

struct Contribution {
    vec4 sh_rgb[3];
};

Contribution Contribution_new() {
    Contribution res;
    res.sh_rgb[0] = vec4(0);
    res.sh_rgb[1] = vec4(0);
    res.sh_rgb[2] = vec4(0);
    return res;
}

void add_radiance_in_direction(inout Contribution self, vec3 radiance, vec3 direction) {
    // https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/gdc2018-precomputedgiobalilluminationinfrostbite.pdf
    // `shEvaluateL1`, plus the `4` factor, with `pi` cancelled out in the evaluation code (BRDF).
    vec4 sh = vec4(0.282095, direction * 0.488603) * 4;
    self.sh_rgb[0] += sh * radiance.r;
    self.sh_rgb[1] += sh * radiance.g;
    self.sh_rgb[2] += sh * radiance.b;
}

void scale(inout Contribution self, float value) {
    self.sh_rgb[0] *= value;
    self.sh_rgb[1] *= value;
    self.sh_rgb[2] *= value;
}

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint dispatch_idx = gl_GlobalInvocationID.x;
    if (IRCACHE_FREEZE) {
        return;
    }

    const uint total_alloc_count = deref(ircache_meta_buf).tracing_alloc_count;
    if (dispatch_idx >= total_alloc_count) {
        return;
    }

    const uint entry_idx = deref(ircache_entry_indirection_buf[dispatch_idx]);
    const uint output_idx = entry_idx * IRCACHE_IRRADIANCE_STRIDE;

    Contribution contribution_sum = Contribution_new();
    {
        float valid_samples = 0;

        // TODO: counter distortion
        for (uint octa_idx = 0; octa_idx < IRCACHE_OCTA_DIMS2; ++octa_idx) {
            const vec2 octa_coord = (vec2(octa_idx % IRCACHE_OCTA_DIMS, octa_idx / IRCACHE_OCTA_DIMS) + 0.5) / IRCACHE_OCTA_DIMS;

            const Reservoir1spp r = Reservoir1spp_from_raw(floatBitsToUint(deref(ircache_aux_buf[entry_idx * IRCACHE_AUX_STRIDE + octa_idx]).xy));
            const vec3 dir = direction(SampleParams_from_raw(r.payload));

            const vec4 contrib = deref(ircache_aux_buf[entry_idx * IRCACHE_AUX_STRIDE + IRCACHE_OCTA_DIMS2 + octa_idx]);

            add_radiance_in_direction(contribution_sum,
                                      contrib.rgb * contrib.w,
                                      dir);

            valid_samples += select(contrib.w > 0, 1.0, 0.0);
        }

        scale(contribution_sum, 1.0 / max(1.0, valid_samples));
    }

    for (uint basis_i = 0; basis_i < IRCACHE_IRRADIANCE_STRIDE; ++basis_i) {
        const vec4 new_value = contribution_sum.sh_rgb[basis_i];
        vec4 prev_value =
            deref(ircache_irradiance_buf[entry_idx * IRCACHE_IRRADIANCE_STRIDE + basis_i]) * deref(gpu_input).pre_exposure_delta;

        const bool should_reset = all(equal(vec4(0.0), prev_value));
        if (should_reset) {
            prev_value = new_value;
        }

        float blend_factor_new = 0.25;
        // float blend_factor_new = 1;
        const vec4 blended_value = mix(prev_value, new_value, blend_factor_new);

        deref(ircache_irradiance_buf[entry_idx * IRCACHE_IRRADIANCE_STRIDE + basis_i]) = blended_value;
    }
}
