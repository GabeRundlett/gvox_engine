#pragma once

// #include <miniaudio.h>
// #include <soloud/soloud.h>
#include <cstdint>

struct AppAudio {
    // ma_device device;
    // ma_waveform waveform;
    // SoLoud::Soloud soloud;

    AppAudio();
    ~AppAudio();

    void set_frequency(float frequency);

  private:
    void callback(void *output, const void *input, uint32_t frame_count);
};
