#include "scene.hpp"
#include "application/input.inl"
#include "renderer/render_scene.hpp"
#include "renderer/render_voxel_object.hpp"
#include "renderer/renderer.hpp"
#include "voxels/voxel_allocator.hpp"
#include "voxels/voxel_object.hpp"
#include "voxels/voxel_brick.hpp"
#include <glm/geometric.hpp>
#include "voxels/pack_unpack.inl"
#include "voxels/voxel_world.hpp"
#include "voxels/animation_playground/animation_playground.hpp"
#include <base/profiler.hpp>

glm::vec3 hsv2rgb(glm::vec3 c) {
    glm::vec4 k = glm::vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    glm::vec3 p = abs(glm::fract(glm::vec3(c.x, c.x, c.x) + glm::vec3(k.x, k.y, k.z)) * 6.0f - k.w);
    return c.z * glm::mix(glm::vec3(k.x), glm::clamp(p - k.x, glm::vec3(0.0), glm::vec3(1.0)), glm::vec3(c.y));
}

vec3 uniform_sample_cone(vec2 urand, float cos_theta_max) {
    float cos_theta = (1.0f - urand.x) + urand.x * cos_theta_max;
    float sin_theta = sqrt(clamp(1.0f - cos_theta * cos_theta, 0.0f, 1.0f));
    float phi = urand.y * (M_PI * 2.0f);
    return vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}
// Building an Orthonormal Basis, Revisited
// http://jcgt.org/published/0006/01/01/
mat3 build_orthonormal_basis(vec3 n) {
    vec3 b1;
    vec3 b2;

    if (n.z < 0.0) {
        const float a = 1.0f / (1.0f - n.z);
        const float b = n.x * n.y * a;
        b1 = vec3(1.0 - n.x * n.x * a, -b, n.x);
        b2 = vec3(b, n.y * n.y * a - 1.0, -n.y);
    } else {
        const float a = 1.0f / (1.0f + n.z);
        const float b = -n.x * n.y * a;
        b1 = vec3(1.0 - n.x * n.x * a, b, -n.x);
        b2 = vec3(b, 1.0 - n.y * n.y * a, -n.y);
    }

    return mat3(b1, b2, n);
}
uint good_rand_hash(uint x) {
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}
uint good_rand_hash(uvec2 v) { return good_rand_hash(v.x ^ good_rand_hash(v.y)); }
uint good_rand_hash(uvec3 v) {
    return good_rand_hash(v.x ^ good_rand_hash(v.y) ^ good_rand_hash(v.z));
}

uint _rand_state;
void rand_seed(uint seed) {
    _rand_state = seed;
}

float rand_() {
    // https://www.pcg-random.org/
    _rand_state = _rand_state * 747796405u + 2891336453u;
    uint result = ((_rand_state >> ((_rand_state >> 28u) + 4u)) ^ _rand_state) * 277803737u;
    result = (result >> 22u) ^ result;
    return result / 4294967295.0;
}

Scene::Scene(GpuContext &gpu_context) : gpu_context(gpu_context) {
    PROFILE_FUNC();

    render_scene = create_render_scene(gpu_context);
    voxel_allocator = create_voxel_allocator();

    float radii[8] = {0.5f, 0.60f, 0.70f, 0.80f, 0.90f, 0.67f, 0.55f, 0.45f};

    for (int frame_i = 0; frame_i < countof(ball_frames); ++frame_i) {
        VoxelObject *voxel_object = new VoxelObject();
        voxel_object->allocator = voxel_allocator;

        voxel_object->brick_min = {0, 0, 0};
        voxel_object->brick_max = {7, 7, 7};

        auto grid_size = voxel_object->brick_max - voxel_object->brick_min + 1;
        voxel_object->brick_grid.resize(grid_size.x * grid_size.y * grid_size.z);

        for (int czi = voxel_object->brick_min.z; czi <= voxel_object->brick_max.z; ++czi) {
            for (int cyi = voxel_object->brick_min.y; cyi <= voxel_object->brick_max.y; ++cyi) {
                for (int cxi = voxel_object->brick_min.x; cxi <= voxel_object->brick_max.x; ++cxi) {
                    glm::ivec3 brick_pos = {cxi, cyi, czi};

                    VoxelBrick *brick = voxel_object->alloc_brick();
                    brick->voxel_min = {BRICK_SIZE, BRICK_SIZE, BRICK_SIZE};
                    brick->voxel_max = {0, 0, 0};
                    brick->render_attribs = voxel_object->alloc_render_brick();
                    brick->brick_i = brick_pos;

                    for (int vzi = 0; vzi < BRICK_SIZE; ++vzi) {
                        brick->bitmask[vzi] = 0;
                        for (int vyi = 0; vyi < BRICK_SIZE; ++vyi) {
                            for (int vxi = 0; vxi < BRICK_SIZE; ++vxi) {
                                int i = vxi + vyi * BRICK_SIZE + vzi * BRICK_SIZE * BRICK_SIZE;
                                glm::ivec3 voxel_pos = brick_pos * BRICK_SIZE + glm::ivec3(vxi, vyi, vzi);
                                vec3 p = (glm::vec3(voxel_pos) + 0.5f) / glm::vec3(grid_size) / float(BRICK_SIZE) * 2.0f - 1.0f;
                                glm::vec3 col = glm::vec3(1.0); // glm::vec3(vxi, vyi, vzi) / glm::vec3(BRICK_SIZE);
                                glm::vec3 nrm = glm::normalize(p);

                                rand_seed(good_rand_hash(floatBitsToUint(nrm)));
                                const mat3 basis = build_orthonormal_basis(normalize(nrm));
                                nrm = basis * uniform_sample_cone(vec2(rand_(), rand_()), cos(0.19f * 0.5f));
                                nrm = glm::normalize(nrm);

                                auto voxel = Voxel{
                                    daxa_f32vec3(col.r, col.g, col.b),
                                    daxa_f32vec3(nrm.r, nrm.g, nrm.b),
                                    0.5f,
                                    0u,
                                };

                                brick->render_attribs->voxels[i] = pack_voxel(voxel);
                                if (dot(p, p) < radii[frame_i]) {
                                    brick->bitmask[vzi] |= 1ull << i;
                                    brick->voxel_min = glm::min(brick->voxel_min, glm::u8vec3(vxi, vyi, vzi));
                                    brick->voxel_max = glm::max(brick->voxel_max, glm::u8vec3(vxi, vyi, vzi));
                                }
                            }
                        }
                    }

                    if (brick->voxel_min.x > brick->voxel_max.x) {
                        voxel_object->free_brick(brick);
                    } else {
                        voxel_object->brick_grid[voxel_object->get_brick_index(brick_pos)] = brick;
                    }
                }
            }
        }

        voxel_object->render_voxel_object = create_render_voxel_object(render_scene);
        voxel_object->render_dirty = true;

        ball_frames[frame_i] = voxel_object;
        voxel_objects.push_back(voxel_object);
    }

    voxel_world = create_voxel_world(this);

    animation_playground = new AnimationPlayground(gpu_context, render_scene, voxel_allocator);
}

Scene::~Scene() {
    delete animation_playground;

    for (auto voxel_object : voxel_objects) {
        if (voxel_object) {
            destroy_render_voxel_object(gpu_context, voxel_object->render_voxel_object);
            delete voxel_object;
        }
    }

    destroy_voxel_world(voxel_world);
    destroy_render_scene(gpu_context, render_scene);
    // Last: everything above returns bricks to it on the way out.
    destroy_voxel_allocator(voxel_allocator);
}

void Scene::update(Renderer &renderer, GpuInput &gpu_input) {
    render_scene_begin(gpu_context, render_scene);
    for (auto voxel_object : voxel_objects)
        update_render_voxel_object(gpu_context, voxel_object);

    update_voxel_world(gpu_context, renderer, gpu_input, voxel_world);

    srand(0);
    for (int zi = 0; zi < 1; zi += 1)
        for (int yi = 0; yi < 1; yi += 1)
            for (int xi = 0; xi < 1; xi += 1) {
                auto voxel_object = ball_frames[int(gpu_input.time * 12 + rand()) % countof(ball_frames)];
                auto grid_size = voxel_object->brick_max - voxel_object->brick_min + 1;
                auto pos = glm::vec3(xi, yi, zi) * float(BRICK_SIZE) * VOXEL_SIZE * glm::vec3(grid_size);
                auto tint = hsv2rgb(glm::vec3(float(rand() % 100) / 100, 0.9f + float(rand() % 100) / 1000, 0.9));
                // auto tint = glm::vec3(1);
                draw_voxel_object(voxel_object, pos - 1000.0f, glm::quat(1, 0, 0, 0), VOXEL_SIZE, tint);

                // Box box;
                // box.p0_x = pos.x;
                // box.p0_y = pos.y;
                // box.p0_z = pos.z;
                // box.p1_x = pos.x + VOXEL_SIZE * BRICK_SIZE * grid_size.x;
                // box.p1_y = pos.y + VOXEL_SIZE * BRICK_SIZE * grid_size.y;
                // box.p1_z = pos.z + VOXEL_SIZE * BRICK_SIZE * grid_size.z;
                // box.r = 1.0f;
                // box.g = 0.2f;
                // box.b = 0.7f;
                // renderer.submit_debug_box_lines(&box, 1);
            }

    animation_playground->update(renderer, gpu_input);

    render_scene_end(gpu_context, render_scene);
}
