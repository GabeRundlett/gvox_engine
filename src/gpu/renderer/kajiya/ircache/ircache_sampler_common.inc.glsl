#pragma once
#include <utils/random.glsl>
#include <utils/normal.glsl>

const uint SAMPLER_SEQUENCE_LENGTH = 1024;

struct SampleParams {
    uint value;
};

SampleParams SampleParams_from_spf_entry_sample_frame(uint samples_per_frame, uint entry_idx, uint sample_idx, uint frame_idx) {
    const uint PERIOD = IRCACHE_OCTA_DIMS2 / samples_per_frame;

    uint xy = sample_idx * PERIOD + (frame_idx % PERIOD);

    // Checkerboard
    xy ^= (xy & 4u) >> 2u;

    SampleParams res;
    res.value = xy + ((frame_idx << 16u) ^ (entry_idx)) * IRCACHE_OCTA_DIMS2;

    return res;
}

SampleParams SampleParams_from_raw(uint raw) {
    SampleParams res;
    res.value = raw;
    return res;
}

uint raw(SampleParams self) {
    return self.value;
}

uint octa_idx(SampleParams self) {
    return self.value % IRCACHE_OCTA_DIMS2;
}

uvec2 octa_quant(SampleParams self) {
    uint oi = octa_idx(self);
    return uvec2(oi % IRCACHE_OCTA_DIMS, oi / IRCACHE_OCTA_DIMS);
}

uint rng(SampleParams self) {
    return hash1(self.value >> 4u);
}

vec2 octa_uv(SampleParams self) {
    const uvec2 oq = octa_quant(self);
    const uint r = rng(self);
    const vec2 urand = r2_sequence(r % SAMPLER_SEQUENCE_LENGTH);
    return (vec2(oq) + urand) / 4.0;
}

// TODO: tackle distortion
vec3 direction(SampleParams self) {
    return octa_decode(octa_uv(self));
}
