#pragma once

#include <glm/glm.hpp>
#include <array>

#include "defines.inl"

struct Chunk {
    static constexpr glm::ivec3 DIM = glm::ivec3(CHUNK_SIZE);
};

struct World {
    static constexpr glm::ivec3 DIM = glm::ivec3{CHUNK_NX, CHUNK_NY, CHUNK_NZ};
};

template <typename T>
using BlockArray = std::array<std::array<std::array<T, Chunk::DIM.x>, Chunk::DIM.y>, Chunk::DIM.z>;
template <typename T>
using ChunkArray = std::array<std::array<std::array<T, World::DIM.x>, World::DIM.y>, World::DIM.z>;
template <typename T>
using ChunkIndexArray = std::array<std::array<std::array<T, World::DIM.x>, World::DIM.y>, World::DIM.z>;
