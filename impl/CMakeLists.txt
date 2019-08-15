message(STATUS "execute impl/CMakeLists.txt ...")

set(src_file "")
fetch_files_with_suffix_recursively(${GEMM_SOURCE_DIR}/impl "cpp" src_file)

if(USE_GPU)
    fetch_files_with_suffix_recursively(${GEMM_SOURCE_DIR}/impl "cu" src_file)
endif()


if(USE_GPU)
    cuda_compile(cuda_c_src_objs SHARED ${src_file})
endif()


set(LIBRARY_OUTPUT_PATH ${GEMM_BINARY_DIR}/lib)
set(ARCHIVE_OUTPUT_DIRECTORY ${GEMM_BINARY_DIR}/lib)


add_library(impl SHARED ${src_file})
add_library(impl_static STATIC ${src_file})
set_target_properties(impl_static PROPERTIES OUTPUT_NAME "impl")

set_target_properties(impl PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(impl_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)

set_target_properties(impl PROPERTIES VERSION 1.0 SOVERSION 1)


