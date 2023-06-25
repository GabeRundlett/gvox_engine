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
#if TRACE_PRIMARY_COMPUTE || defined(__cplusplus)
#include <shared/tasks/trace_primary.inl>
#endif
#if COLOR_SCENE_COMPUTE || defined(__cplusplus)
#include <shared/tasks/color_scene.inl>
#endif
#if POSTPROCESSING_RASTER || defined(__cplusplus)
#include <shared/tasks/postprocessing.inl>
#endif

#if CHUNK_HIERARCHY_COMPUTE || defined(__cplusplus)
#include <shared/tasks/chunk_hierarchy.inl>
#endif

#if VOXEL_PARTICLE_SIM_COMPUTE || defined(__cplusplus)
#include <shared/tasks/voxel_particle_sim.inl>
#endif
#if VOXEL_PARTICLE_RASTER || defined(__cplusplus)
#include <shared/tasks/voxel_particle_raster.inl>
#endif
