include(cmake/utils.cmake)

set(src_file "")
fetch_files_with_suffix_recursively(${GEMM_SOURCE_DIR}/impl "*" src_file)

CUDA_COMPILE(cuda_c_src_objs SHARED ${src_file})

add_library(IMPL SHARED ${cuda_c_src_objs})