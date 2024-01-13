#pragma once

#include <cpu/mesh_model.hpp>

#include <daxa/gpu_resources.hpp>
#include <daxa/utils/task_graph.hpp>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <glm/glm.hpp>

#include <daxa/utils/task_graph_types.hpp>
#include <FreeImage.h>

namespace {
    void load_textures(Mesh &mesh, TextureMap &global_textures, aiMaterial *mat, std::filesystem::path const &rootdir) {
        auto texture_n = std::min(1u, mat->GetTextureCount(aiTextureType_DIFFUSE));
        if (texture_n == 0) {
            mesh.textures.push_back(global_textures.at("#default_texture"));
        } else {
            for (uint32_t i = 0; i < texture_n; i++) {
                aiString str;
                mat->GetTexture(aiTextureType_DIFFUSE, i, &str);
                auto path = (rootdir / str.C_Str()).make_preferred();
                auto texture_iter = global_textures.find(path.string());
                if (texture_iter != global_textures.end()) {
                    mesh.textures.push_back(texture_iter->second);
                } else {
                    mesh.textures.push_back(global_textures[path.string()] = std::make_shared<Texture>(Texture{.path = path}));
                }
            }
        }
    }

    void process_node(daxa::Device device, MeshModel &model, aiNode *node, aiScene const *scene, std::filesystem::path const &rootdir, glm::mat4 const &parent_transform) {
        auto transform = *reinterpret_cast<glm::mat4 *>(&node->mTransformation);
        transform = transform * parent_transform;
        auto transposed_transform = glm::transpose(transform);
        for (uint32_t mesh_i = 0; mesh_i < node->mNumMeshes; ++mesh_i) {
            aiMesh *aimesh = scene->mMeshes[node->mMeshes[mesh_i]];
            // TODO: Fix. Min and Max bounds seem to be broken.
            auto g_min = transposed_transform * glm::vec4(aimesh->mAABB.mMin.x, aimesh->mAABB.mMin.y, aimesh->mAABB.mMin.z, 1.0);
            auto g_max = transposed_transform * glm::vec4(aimesh->mAABB.mMax.x, aimesh->mAABB.mMax.y, aimesh->mAABB.mMax.z, 1.0);
            model.bound_min.x = std::min(g_min.x, model.bound_min.x);
            model.bound_min.y = std::min(g_min.y, model.bound_min.y);
            model.bound_min.z = std::min(g_min.z, model.bound_min.z);
            model.bound_max.x = std::max(g_max.x, model.bound_max.x);
            model.bound_max.y = std::max(g_max.y, model.bound_max.y);
            model.bound_max.z = std::max(g_max.z, model.bound_max.z);
            model.meshes.push_back({});
            auto &o_mesh = model.meshes.back();
            o_mesh.modl_mat = *reinterpret_cast<daxa_f32mat4x4 *>(&transposed_transform);
            auto &verts = o_mesh.verts;
            verts.reserve(aimesh->mNumFaces * 3);
            for (size_t face_i = 0; face_i < aimesh->mNumFaces; ++face_i) {
                for (uint32_t index_i = 0; index_i < 3; ++index_i) {
                    auto vert_i = aimesh->mFaces[face_i].mIndices[index_i];
                    auto pos = aimesh->mVertices[vert_i];
                    auto tex = aimesh->mTextureCoords[0][vert_i];
                    verts.push_back({
                        .pos = {pos.x, pos.y, pos.z},
                        .tex = {tex.x, tex.y},
                    });
                }
            }
            aiMaterial *material = scene->mMaterials[aimesh->mMaterialIndex];
            load_textures(o_mesh, model.textures, material, rootdir);
            o_mesh.vertex_buffer = device.create_buffer(daxa::BufferInfo{
                .size = static_cast<uint32_t>(sizeof(MeshVertex) * o_mesh.verts.size()),
                .name = "vertex_buffer",
            });
            o_mesh.normal_buffer = device.create_buffer(daxa::BufferInfo{
                .size = static_cast<uint32_t>(sizeof(MeshVertex) * o_mesh.verts.size() / 3),
                .name = "normal_buffer",
            });
        }
        for (uint32_t i = 0; i < node->mNumChildren; ++i) {
            process_node(device, model, node->mChildren[i], scene, rootdir, transform);
        }
    }

    constexpr auto default_texture_pixels = std::array<uint32_t, 16 * 16>{
        // clang-format off
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,

        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,

        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,

        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
        // clang-format on
    };
} // namespace

void open_mesh_model(daxa::Device device, MeshModel &model, std::filesystem::path const &filepath, std::string const &name) {
    Assimp::Importer import{};
    aiScene const *scene = import.ReadFile(filepath.string(), aiProcess_Triangulate | aiProcess_GenBoundingBoxes);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        return;
    }
    model.textures["#default_texture"] = std::make_shared<Texture>();
    model.bound_min = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    model.bound_max = {std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min()};
    process_node(device, model, scene->mRootNode, scene, filepath.parent_path(), {{1, 0, 0, 0}, {0, 0, -1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}});
    auto texture_staging_buffers = std::vector<daxa::BufferId>{};
    daxa::TaskGraph mip_task_list = daxa::TaskGraph({
        .device = device,
        .name = "mesh upload task list",
    });
    auto fi_bitmaps = std::vector<FIBITMAP *>{};
    for (auto &[key, texture] : model.textures) {
        if (key == "#default_texture") {
            texture->pixels = reinterpret_cast<uint8_t const *>(default_texture_pixels.data());
            texture->size_x = static_cast<uint32_t>(16);
            texture->size_y = static_cast<uint32_t>(16);
        } else {
            auto fi_file_desc = FreeImage_GetFileType(texture->path.string().c_str(), 0);
            FIBITMAP *fi_bitmap = FreeImage_Load(fi_file_desc, texture->path.string().c_str());
            auto pixel_size = FreeImage_GetBPP(fi_bitmap);
            if (pixel_size != 32) {
                auto *temp = FreeImage_ConvertTo32Bits(fi_bitmap);
                FreeImage_Unload(fi_bitmap);
                fi_bitmap = temp;
            }
            texture->size_x = static_cast<uint32_t>(FreeImage_GetWidth(fi_bitmap));
            texture->size_y = static_cast<uint32_t>(FreeImage_GetHeight(fi_bitmap));
            texture->pixels = FreeImage_GetBits(fi_bitmap);
            assert(texture->pixels != nullptr && "Failed to load image");
            fi_bitmaps.push_back(fi_bitmap);
        }
        auto src_channel_n = 4u;
        auto dst_channel_n = 4u;
        texture->image_id = device.create_image({
            .format = daxa::Format::B8G8R8A8_SRGB,
            .size = {texture->size_x, texture->size_y, 1},
            .mip_level_count = 4,
            .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::TRANSFER_DST,
            .name = "image",
        });
        auto sx = static_cast<uint32_t>(texture->size_x);
        auto sy = static_cast<uint32_t>(texture->size_y);
        auto data = texture->pixels;
        auto &image_id = texture->image_id;
        size_t image_size = sx * sy * sizeof(uint8_t) * dst_channel_n;
        auto texture_staging_buffer = device.create_buffer({
            .size = static_cast<uint32_t>(image_size),
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "texture_staging_buffer",
        });
        auto *staging_buffer_ptr = device.get_host_address_as<uint8_t>(texture_staging_buffer).value();
        for (size_t i = 0; i < sx * sy; ++i) {
            size_t src_offset = i * src_channel_n;
            size_t dst_offset = i * dst_channel_n;
            for (size_t ci = 0; ci < std::min(src_channel_n, dst_channel_n); ++ci) {
                staging_buffer_ptr[ci + dst_offset] = data[ci + src_offset];
            }
        }
        texture_staging_buffers.push_back(texture_staging_buffer);

        texture->task_image = daxa::TaskImage(daxa::TaskImageInfo{
            .initial_images = {
                .images = std::array{image_id},
                .latest_slice_states = std::array{daxa::ImageSliceState{
                    .latest_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                    .slice = {.level_count = 4},
                }},
            },
            .name = name + key,
        });
        mip_task_list.use_persistent_image(texture->task_image);
        auto task_image_mip_view = texture->task_image.view().view({.base_mip_level = 0, .level_count = 4});

        mip_task_list.add_task({
            .uses = {
                daxa::ImageTransferWrite<daxa::ImageViewType::REGULAR_2D>{task_image_mip_view},
            },
            .task = [texture_staging_buffer, task_image_mip_view, sx, sy](daxa::TaskInterface const &ti) {
                auto &cmd_list = ti.get_recorder();
                cmd_list.copy_buffer_to_image({
                    .buffer = texture_staging_buffer,
                    .image = ti.uses[task_image_mip_view].image(),
                    .image_layout = ti.uses[task_image_mip_view].layout(),
                    .image_offset = {0, 0, 0},
                    .image_extent = {sx, sy, 1},
                });
            },
            .name = "upload",
        });
        for (uint32_t i = 0; i < 3; ++i) {
            auto view_a = texture->task_image.view().view({.base_mip_level = i});
            auto view_b = texture->task_image.view().view({.base_mip_level = i + 1});
            mip_task_list.add_task({
                .uses = {
                    daxa::ImageTransferRead<daxa::ImageViewType::REGULAR_2D>{view_a},
                    daxa::ImageTransferWrite<daxa::ImageViewType::REGULAR_2D>{view_b},
                },
                .task = [=, &device](daxa::TaskInterface const &runtime) {
                    auto &cmd_list = runtime.get_recorder();
                    auto image_a = runtime.uses[view_a].image();
                    auto image_b = runtime.uses[view_b].image();
                    auto image_info = device.info_image(image_a).value();
                    auto mip_size = std::array<int32_t, 3>{std::max<int32_t>(1, static_cast<int32_t>(image_info.size.x)), std::max<int32_t>(1, static_cast<int32_t>(image_info.size.y)), std::max<int32_t>(1, static_cast<int32_t>(image_info.size.z))};
                    for (uint32_t j = 0; j < i; ++j) {
                        mip_size = {std::max<int32_t>(1, mip_size[0] / 2), std::max<int32_t>(1, mip_size[1] / 2), std::max<int32_t>(1, mip_size[2] / 2)};
                    }
                    auto next_mip_size = std::array<int32_t, 3>{std::max<int32_t>(1, mip_size[0] / 2), std::max<int32_t>(1, mip_size[1] / 2), std::max<int32_t>(1, mip_size[2] / 2)};
                    cmd_list.blit_image_to_image({
                        .src_image = image_a,
                        .src_image_layout = runtime.uses[view_a].layout(),
                        .dst_image = image_b,
                        .dst_image_layout = runtime.uses[view_b].layout(),
                        .src_slice = {
                            .mip_level = i,
                            .base_array_layer = 0,
                            .layer_count = 1,
                        },
                        .src_offsets = {{{0, 0, 0}, {mip_size[0], mip_size[1], mip_size[2]}}},
                        .dst_slice = {
                            .mip_level = i + 1,
                            .base_array_layer = 0,
                            .layer_count = 1,
                        },
                        .dst_offsets = {{{0, 0, 0}, {next_mip_size[0], next_mip_size[1], next_mip_size[2]}}},
                        .filter = daxa::Filter::LINEAR,
                    });
                },
                .name = "mip_level_" + std::to_string(i),
            });
        }
        mip_task_list.add_task({
            .uses = {
                daxa::ImageFragmentShaderSampled<daxa::ImageViewType::REGULAR_2D>{texture->task_image.view().view({.base_mip_level = 0, .level_count = 4})},
            },
            .task = [](daxa::TaskInterface const &) {},
            .name = "Transition",
        });
    }

    mip_task_list.submit({});
    mip_task_list.complete({});
    mip_task_list.execute({});
    device.wait_idle();
    for (auto texture_staging_buffer : texture_staging_buffers) {
        device.destroy_buffer(texture_staging_buffer);
    }
    for (auto *fi_bitmap : fi_bitmaps) {
        FreeImage_Unload(fi_bitmap);
    }
}
