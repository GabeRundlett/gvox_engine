#pragma once

#include <utils/math.glsl>

struct Reservoir1sppStreamState {
    float p_q_sel;
    float M_sum;
};

Reservoir1sppStreamState Reservoir1sppStreamState_create() {
    Reservoir1sppStreamState res;
    res.p_q_sel = 0;
    res.M_sum = 0;
    return res;
}

struct Reservoir1spp {
    float w_sum; // Doesn't need storing. TODO: maybe move to Reservoir1sppStreamState.
    uint payload;
    float M;
    float W;
};

Reservoir1spp Reservoir1spp_create() {
    Reservoir1spp res;
    res.w_sum = 0;
    res.payload = 0;
    res.M = 0;
    res.W = 0;
    return res;
}

Reservoir1spp Reservoir1spp_from_raw(daxa_u32vec2 raw) {
    Reservoir1spp res;
    res.w_sum = 0;
    res.payload = raw.x;
    const daxa_f32vec2 MW = unpackHalf2x16(raw.y);
    res.M = MW[0];
    res.W = MW[1];
    return res;
}

daxa_u32vec2 as_raw(inout Reservoir1spp self) {
    return daxa_u32vec2(self.payload, packHalf2x16(daxa_f32vec2(self.M, self.W)));
}

bool update(inout Reservoir1spp self, float w, uint sample_payload, inout uint rng) {
    self.w_sum += w;
    self.M += 1;
    const float dart = uint_to_u01_float(hash1_mut(rng));
    const float prob = w / self.w_sum;

    if (prob >= dart) {
        self.payload = sample_payload;
        return true;
    } else {
        return false;
    }
}

bool update_with_stream(
    inout Reservoir1spp self,
    Reservoir1spp r,
    float p_q,
    float weight,
    inout Reservoir1sppStreamState stream_state,
    uint sample_payload,
    inout uint rng) {
    stream_state.M_sum += r.M;

    if (update(self, p_q * weight * r.W * r.M, sample_payload, rng)) {
        stream_state.p_q_sel = p_q;
        return true;
    } else {
        return false;
    }
}

void init_with_stream(
    inout Reservoir1spp self,
    float p_q,
    float weight,
    inout Reservoir1sppStreamState stream_state,
    uint sample_payload) {
    self.payload = sample_payload;
    self.w_sum = p_q * weight;
    self.M = weight != 0 ? 1 : 0;
    self.W = weight;

    stream_state.p_q_sel = p_q;
    stream_state.M_sum = self.M;
}

void finish_stream(inout Reservoir1spp self, Reservoir1sppStreamState state) {
    self.M = state.M_sum;
    self.W = self.w_sum / (max(1e-8, self.M * state.p_q_sel));
}
