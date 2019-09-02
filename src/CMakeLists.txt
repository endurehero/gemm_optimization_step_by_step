set(src_file "")
fetch_files_with_suffix(${GEMM_SOURCE_DIR}/src "cpp" src_file)
fetch_files_with_suffix(${GEMM_SOURCE_DIR}/src/cpu "cpp" src_file)

if(USE_GPU)
    set(cuda_src_file "")
    add_subdirectory(${GEMM_SOURCE_DIR}/src/gpu)
    list(APPEND src_file ${cuda_src_file})
endif()

cuda_add_executable(GEMM ${src_file})

if(BUILD_SHARED)
    target_link_libraries(GEMM ${LINKER_LIBS})
    if(USE_GPU)
        target_link_libraries(GEMM ${CUDA_LINKER_LIBS})
    endif()
endif()

message(STATUS "Executable output: ${EXEC_OUTPUT}")
set_target_properties(GEMM PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXEC_OUTPUT})