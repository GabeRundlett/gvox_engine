#pragma once

#include <daxa/daxa.hpp>
#include <daxa/utils/task_graph.hpp>
#include <utilities/mesh/mesh_model.inl>

struct Texture {
    std::filesystem::path path;
    daxa::ImageId image_id;
    daxa::TaskImage task_image;
    daxa_u32 size_x, size_y;
    daxa_i32 channel_n;
    uint8_t const *pixels;
};
struct Mesh {
    std::vector<MeshVertex> verts;
    std::vector<std::shared_ptr<Texture>> textures;
    daxa_f32mat4x4 modl_mat;
    daxa::BufferId vertex_buffer;
    daxa::BufferId normal_buffer;
};
using TextureMap = std::unordered_map<std::string, std::shared_ptr<Texture>>;
struct MeshModel {
    TextureMap textures;
    std::vector<Mesh> meshes;
    daxa_f32vec3 bound_min;
    daxa_f32vec3 bound_max;
};

void open_mesh_model(daxa::Device device, MeshModel &model, std::filesystem::path const &filepath, std::string const &name);
