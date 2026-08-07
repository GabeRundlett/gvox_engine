#include "scene.hpp"
#include "renderer/render_scene.hpp"
#include "renderer/render_voxel_object.hpp"
#include "voxels/voxel_object.hpp"
#include <glm/geometric.hpp>
#include "voxels/pack_unpack.inl"

Scene::Scene(GpuContext &gpu_context) : gpu_context(gpu_context) {
    VoxelObject *voxel_object = new VoxelObject();

    VoxelBrick *brick = new VoxelBrick();
    brick->voxel_min = {0, 0, 0};
    brick->voxel_max = {BRICK_SIZE - 1, BRICK_SIZE - 1, BRICK_SIZE - 1};
    brick->render_attribs = new VoxelRenderBrick();

    for (int zi = brick->voxel_min.z; zi <= brick->voxel_max.z; ++zi) {
        for (int yi = brick->voxel_min.y; yi <= brick->voxel_max.y; ++yi) {
            for (int xi = brick->voxel_min.x; xi <= brick->voxel_max.x; ++xi) {
                int i = xi + yi * BRICK_SIZE + zi * BRICK_SIZE * BRICK_SIZE;
                uint64_t packed_attribs = 0;

                glm::vec3 col = glm::vec3(xi, yi, zi) / glm::vec3(BRICK_SIZE);
                glm::vec3 nrm = glm::normalize(glm::vec3(xi, yi, zi) - glm::vec3(4));
                auto voxel = GpuVoxel{
                    daxa_f32vec3(col.r, col.g, col.b),
                    daxa_f32vec3(nrm.r, nrm.g, nrm.b),
                    1.0,
                    0u,
                };
                brick->render_attribs->packed_attribs[i] = pack_voxel(voxel).data;
            }
        }
        brick->bitmask[zi] = 0xffff'ffff'ffff'ffffull;
    }

    render_scene = create_render_scene(gpu_context);
    voxel_object->render_voxel_object = create_render_voxel_object(render_scene);
    voxel_object->render_dirty = true;

    voxel_objects.push_back(voxel_object);
}

Scene::~Scene() {
    for (auto voxel_object : voxel_objects) {
        if (voxel_object) {
            destroy_render_voxel_object(gpu_context, voxel_object->render_voxel_object);
            delete voxel_object;
        }
    }

    destroy_render_scene(gpu_context, render_scene);
}
