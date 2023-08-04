#pragma once

#include <shared/input.inl>
#include <shared/globals.inl>
#include <shared/voxels.inl>

#include <shared/core.inl>

#include <shared/tasks/startup.inl>
#include <shared/tasks/perframe.inl>
#include <shared/tasks/chunk_edit.inl>
#include <shared/tasks/chunk_opt.inl>
#include <shared/tasks/chunk_alloc.inl>
#include <shared/tasks/per_chunk.inl>
#include <shared/tasks/voxel_particle_sim.inl>

#include <shared/tasks/renderer/voxel_particle_raster.inl>

#include <shared/tasks/renderer/downscale.inl>
#include <shared/tasks/renderer/trace_primary.inl>
#include <shared/tasks/renderer/calculate_reprojection_map.inl>
#include <shared/tasks/renderer/ssao.inl>
#include <shared/tasks/renderer/trace_secondary.inl>
#include <shared/tasks/renderer/diffuse_gi.inl>
#include <shared/tasks/renderer/taa.inl>
#include <shared/tasks/renderer/postprocessing.inl>
