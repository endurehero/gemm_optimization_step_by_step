#!/bin/bash

if [ ! -d "./build" ]; then
    mkdir ./build
fi

cd ./build
rm -rf *

cmake \
    -DENABLE_DEBUG=NO \
    -DBUILD_SHARED=YES \
    -DUSE_CBLAS=YES \
    -DUSE_CPU_RAW=NO \
    -DUSE_CPU_OPT1=NO \
    -DUSE_CPU_OPT2=NO \
    -DUSE_CPU_OPT3=NO \
    -DUSE_CPU_OPT4=NO \
    -DUSE_CPU_OPT5=NO \
    -DUSE_CPU_OPT6=NO \
    -DUSE_CPU_OPT7=NO \
    -DUSE_CPU_OPT8=NO \
    -DUSE_CPU_OPT9=NO \
    -DUSE_GPU=YES \
        -DUSE_CUBLAS=YES \
        -DUSE_GPU_RAW=NO \
        -DUSE_GPU_OPT1=NO \
        -DUSE_GPU_OPT2=NO \
        -DUSE_TENSOR_CORE=YES \
    ..
make