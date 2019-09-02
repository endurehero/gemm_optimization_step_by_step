#The following are set after configuration is done:
# ATLAS_FOUND
# ATLAS_INCLUD_DIRS
# ATLAS_LIBRARIES

set(atlas_include_search_paths
    /usr/include/atlas
    /usr/include/atlas-base
    $ENV{Atlas_ROOT_DIR}
    $ENV{Atlas_ROOT_DIR}/include)


set(atlas_lib_search_paths
    /usr/lib/atlas
    /usr/lib/atlas-base
    $ENV{Atlas_ROOT_DIR}
    $ENV{Atlas_ROOT_DIR}/lib)

find_path(atlas_cblas_include_dir NAMES cblas.h PATHS ${atlas_include_search_paths})
find_library(atlas_cblas_lib NAMES ptcblas_r ptcblas cblas_r cblas PATHS ${atlas_lib_search_paths})

set(looked_for
    atlas_cblas_include_dir

    atlas_cblas_lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Atlas DEFAULT ${looked_for})

if(ATLAS_FOUND)
    set(ATLAS_INCLUDE_DIRS ${atlas_cblas_include_dir})
    set(ATLAS_LIBRARIES  ${atlas_cblas_lib})
    mark_as_advanced(${looked_for})
    
    message(STATUS "Found Atlas (include: ${atlas_cblas_include_dir} library: ${atlas_cblas_lib}")
endif()