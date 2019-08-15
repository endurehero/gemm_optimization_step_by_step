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

function(fetch_files_with_suffix_recursively search_dir sufffix outputs)
    set(abs_dir "")
    file(GLOB_RECURSE ${abs_dir} ${search_dir} "*.${suffix}")
    set(${outputs} ${${outputs}} ${abs_dir})
endfunction()