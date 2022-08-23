#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "../../src/window.hpp"

#include <soundio/soundio.h>

#include <cmath>
#include <cstdio>

namespace music {
    template <int base_oct, int new_oct>
    constexpr float octave_scale() {
        int oct_offset = new_oct - base_oct;
        if (new_oct < base_oct) {
            return 1.0f / (1 << (-oct_offset));
        } else {
            return 1.0f * (1 << oct_offset);
        }
    }

    template <int octave>
    inline constexpr float A = octave_scale<4, octave>() * 440.0000f;
    template <int octave>
    inline constexpr float B = octave_scale<4, octave>() * 493.8833f;
    template <int octave>
    inline constexpr float C = octave_scale<5, octave>() * 523.2511f;
    template <int octave>
    inline constexpr float D = octave_scale<5, octave>() * 587.3295f;
    template <int octave>
    inline constexpr float E = octave_scale<5, octave>() * 659.2551f;
    template <int octave>
    inline constexpr float F = octave_scale<5, octave>() * 698.4565f;
    template <int octave>
    inline constexpr float G = octave_scale<5, octave>() * 783.9909f;

    namespace wave {
        float sine(float x) {
            return std::sin(x);
        }

        float sawtooth(float x) {
            return std::fmodf(x / (3.14159265f * 2.0f), 1.0f) * 2 - 1;
        }

        float triangle(float x) {
            return std::abs(std::fmodf(x / 3.14159265f, 1.0f) * 2 - 1) * 2 - 1;
        }

        float square(float x) {
            return sawtooth(x) < 0.0f ? -1.0f : 1.0f;
        }

        float piano(float x) {
            float y = wave::sine(x) * std::exp(-0.004f * x);
            y += wave::sine(x * 2.0f) * std::exp(-0.004f * x) / 2.0f;
            y += wave::sine(x * 3.0f) * std::exp(-0.004f * x) / 4.0f;
            y += wave::sine(x * 4.0f) * std::exp(-0.004f * x) / 8.0f;
            y += wave::sine(x * 5.0f) * std::exp(-0.004f * x) / 16.0f;
            y += wave::sine(x * 6.0f) * std::exp(-0.004f * x) / 32.0f;
            return y;
        }
    } // namespace wave
} // namespace music

float audio_func(float x) {
    using namespace music;
    float y = wave::piano(x * C<4>);
    return y * 0.1f;
}

static const float PI = 3.1415926535f;
static float seconds_offset = 0.0f;

static void write_callback(SoundIoOutStream *outstream, int frame_count_min, int frame_count_max) {
    const SoundIoChannelLayout &layout = outstream->layout;
    float float_sample_rate = outstream->sample_rate;
    float seconds_per_frame = 1.0f / float_sample_rate;
    struct SoundIoChannelArea *areas;
    int frames_left = frame_count_max;
    int err;
    while (frames_left > 0) {
        int frame_count = frames_left;
        if ((err = soundio_outstream_begin_write(outstream, &areas, &frame_count))) {
            std::fprintf(stderr, "%s\n", soundio_strerror(err));
            exit(1);
        }
        if (!frame_count)
            break;
        for (int frame = 0; frame < frame_count; frame += 1) {
            float x = (seconds_offset + frame * seconds_per_frame) * 2.0f * PI;
            float sample = audio_func(x);
            for (int channel = 0; channel < layout.channel_count; channel += 1) {
                float *ptr = (float *)(areas[channel].ptr + areas[channel].step * frame);
                *ptr = sample;
            }
        }
        seconds_offset = (seconds_offset + seconds_per_frame * frame_count);
        if ((err = soundio_outstream_end_write(outstream))) {
            std::fprintf(stderr, "%s\n", soundio_strerror(err));
            exit(1);
        }
        frames_left -= frame_count;
    }
}

struct App {
    Window window;

    App() {
        window.set_user_pointer<App>(this);
    }

    void update() {
        window.update();
    }

    Window &get_window() { return window; }
    void on_mouse_move(const glm::dvec2 m) {
    }
    void on_mouse_scroll(const glm::dvec2 offset) {
    }
    void on_mouse_button(int button, int action) {
    }
    void on_key(int key, int action) {
    }
    void on_resize() { update(); }
};

static std::atomic_bool keep_running = true;

void audio_main() {
    SoundIo *soundio = soundio_create();
    soundio_connect(soundio);
    soundio_flush_events(soundio);
    int default_out_device_index = soundio_default_output_device_index(soundio);
    SoundIoDevice *device = soundio_get_output_device(soundio, default_out_device_index);
    SoundIoOutStream *outstream = soundio_outstream_create(device);
    outstream->format = SoundIoFormatFloat32NE;
    outstream->write_callback = write_callback;
    soundio_outstream_open(outstream);
    soundio_outstream_start(outstream);
    while (keep_running) {
        soundio_wait_events(soundio);
        keep_running = false;
    }
    soundio_outstream_destroy(outstream);
    soundio_device_unref(device);
    soundio_destroy(soundio);
}

int main() {
    auto audio_thread = std::thread(audio_main);
    {
        App app;
        while (true) {
            app.update();
            if (app.window.should_close())
                break;
        }
        keep_running = false;
    }
    audio_thread.join();
}
