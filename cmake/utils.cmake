function(fetch_files_with_suffix search_dir suffix outputs)
    exec_program(ls ${search_dir}
                ARGS "*.${suffix}"
                OUTPUT_VARIABLE OUTPUT
                RETURN_VALUE VALUE)
    
    if(NOT VALUE)
        string(REPLACE "\n" ";" OUTPUT_LIST "${OUTPUT}")
        set(abs_dir "")
        foreach(var ${OUTPUT_LIST})
            set(abs_dir ${abs_dir} ${search_dir}/${var})
        endforeach()

        set(${outputs} ${${outputs}} ${abs_dir} PARENT_SCOPE)
    endif()
endfunction()

function(fetch_files_with_suffix_recursively search_dir suffix outputs)
    set(dir "")
    file(GLOB_RECURSE dir RELATIVE ${search_dir} "*.${suffix}")

    set(abs_dir "")
    foreach(d ${dir})
        set(abs_dir ${abs_dir} ${search_dir}/${d})
    endforeach()
    
    set(${outputs} ${${outputs}} ${abs_dir} PARENT_SCOPE)
endfunction()

function(fetch_include_recursively root_dir)
    if(IS_DIRECTORY ${root_dir})
        #message(STATUS "root_dir = ${root_dir}")
        include_directories(${root_dir})
    endif()

    file(GLOB all_sub RELATIVE ${root_dir} ${root_dir}/*)
    foreach(sub ${all_sub})
        if(IS_DIRECTORY ${root_dir}/${sub})
            fetch_include_recursively(${root_dir}/${sub})
        endif()
    endforeach()
endfunction()