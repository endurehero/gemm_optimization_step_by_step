if(USE_GPU)
    fetch_files_with_suffix_recursively(${GEMM_SOURCE_DIR}/src/gpu "cu" cuda_src_file)
    set(cuda_src_file ${cuda_src_file} PARENT_SCOPE) 
endif()