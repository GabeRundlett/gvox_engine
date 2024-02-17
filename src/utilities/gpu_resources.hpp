#pragma once

#include <daxa/daxa.hpp>
#include <daxa/utils/task_graph.hpp>

#include "record_context.hpp"

struct GpuResources {
    daxa::ImageId value_noise_image;
    daxa::ImageId blue_noise_vec2_image;
    daxa::ImageId debug_texture;
    daxa::ImageId test_texture;
    daxa::ImageId test_texture2;

    daxa::BufferId input_buffer;
    daxa::BufferId output_buffer;
    daxa::BufferId staging_output_buffer;
    daxa::BufferId globals_buffer;

    daxa::SamplerId sampler_nnc;
    daxa::SamplerId sampler_lnc;
    daxa::SamplerId sampler_llc;
    daxa::SamplerId sampler_llr;

    daxa::TaskImage task_value_noise_image{{.name = "task_value_noise_image"}};
    daxa::TaskImage task_blue_noise_vec2_image{{.name = "task_blue_noise_vec2_image"}};
    daxa::TaskImage task_debug_texture{{.name = "task_debug_texture"}};
    daxa::TaskImage task_test_texture{{.name = "task_test_texture"}};
    daxa::TaskImage task_test_texture2{{.name = "task_test_texture2"}};

    daxa::TaskBuffer task_input_buffer{{.name = "task_input_buffer"}};
    daxa::TaskBuffer task_output_buffer{{.name = "task_output_buffer"}};
    daxa::TaskBuffer task_staging_output_buffer{{.name = "task_staging_output_buffer"}};
    daxa::TaskBuffer task_globals_buffer{{.name = "task_globals_buffer"}};

    void create(daxa::Device &device);
    void destroy(daxa::Device &device) const;

    void use_resources(RecordContext &record_ctx);
    void update_seeded_value_noise(daxa::Device &device, uint64_t seed);
};
