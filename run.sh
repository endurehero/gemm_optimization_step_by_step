#!/bin/bash

EXEC_FILE="./output/GEMM"

if [ ! -f ${EXEC_FILE} ]; then
    echo "no valid exec file to execut!"
    exit 1
fi

#verify
${EXEC_FILE} ./dat/test1.dat -verify GPU_CUBLAS GPU_RAW

#perf
${EXEC_FILE} ./dat/test1.dat -perf ./output/perf1.dat GPU_CUBLAS GPU_OPT1 GPU_OPT2