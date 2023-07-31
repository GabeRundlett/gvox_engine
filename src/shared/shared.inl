#pragma once

#include <shared/input.inl>
#include <shared/globals.inl>
#include <shared/voxels.inl>

#include <shared/core.inl>

#if STARTUP_COMPUTE || defined(__cplusplus)
#include <shared/tasks/startup.inl>
#endif
#if PERFRAME_COMPUTE || defined(__cplusplus)
#include <shared/tasks/perframe.inl>
#endif
#if CHUNK_EDIT_COMPUTE || defined(__cplusplus)
#include <shared/tasks/chunk_edit.inl>
#endif
#if CHUNK_OPT_COMPUTE || defined(__cplusplus)
#include <shared/tasks/chunk_opt.inl>
#endif
#if CHUNK_ALLOC_COMPUTE || defined(__cplusplus)
#include <shared/tasks/chunk_alloc.inl>
#endif
#if PER_CHUNK_COMPUTE || defined(__cplusplus)
#include <shared/tasks/per_chunk.inl>
#endif
#if VOXEL_PARTICLE_SIM_COMPUTE || defined(__cplusplus)
#include <shared/tasks/voxel_particle_sim.inl>
#endif

#if VOXEL_PARTICLE_RASTER || defined(__cplusplus)
#include <shared/tasks/renderer/voxel_particle_raster.inl>
#endif

#include <shared/tasks/renderer/downscale.inl>
#include <shared/tasks/renderer/trace_primary.inl>
#include <shared/tasks/renderer/calculate_reprojection_map.inl>
#include <shared/tasks/renderer/ssao.inl>
#include <shared/tasks/renderer/trace_secondary.inl>
#include <shared/tasks/renderer/postprocessing.inl>
