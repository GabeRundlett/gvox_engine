#include "app_audio.hpp"

#include <cstdint>
#include <cmath>
#include <cassert>

AppAudio::AppAudio() {
    // auto device_config = ma_device_config_init(ma_device_type_playback);
    // device_config.playback.format = ma_format_f32;
    // device_config.playback.channels = 2; // ?
    // device_config.sampleRate = 48000;
    // device_config.dataCallback = [](ma_device *device_ptr, void *output, const void *input, ma_uint32 frame_count) {
    //     if (!device_ptr->pUserData) {
    //         return;
    //     }
    //     static_cast<AppAudio *>(device_ptr->pUserData)->callback(static_cast<float *>(output), input, frame_count);
    // };
    // device_config.pUserData = this;

    // ma_device_init(NULL, &device_config, &device);

    // auto waveform_config = ma_waveform_config_init(
    //     device.playback.format, device.playback.channels, device.sampleRate,
    //     ma_waveform_type_sine, 0.2, 220);
    // ma_waveform_init(&waveform_config, &waveform);

    // ma_device_start(&device);
}

AppAudio::~AppAudio() {
    // ma_waveform_uninit(&waveform);
    // ma_device_uninit(&device);
}

void AppAudio::set_frequency(float) {
    // ma_device_stop(&device);
    // ma_waveform_uninit(&waveform);
    // auto waveform_config = ma_waveform_config_init(
    //     device.playback.format, device.playback.channels, device.sampleRate,
    //     ma_waveform_type_sine, 0.2, static_cast<double>(frequency));
    // ma_waveform_init(&waveform_config, &waveform);
    // ma_device_start(&device);
}

void AppAudio::callback(void *, const void *, uint32_t) {
    // ma_waveform_read_pcm_frames(&waveform, output, frame_count, NULL);

    // for (uint32_t i = 0; i < frame_count; ++i) {
    //     // generate sound data?
    //     float frequency = 440.0f;
    //     float amplitude = 0.5f;
    //     float x = float(i) * frequency / device_config.sampleRate;
    //     output[i] = ((x - std::floorf(x)) * 2.0f - 1.0f) * amplitude;
    // }
}
