#!/bin/bash

if [ -d "./gemm_build" ]; then
    rm -rf ./gemm_build
fi

if [ -d "./output" ]; then
    rm -rf ./output
fi

mkdir ./output
mkdir ./gemm_build

cd ./gemm_build

cmake \
    -DENABLE_DEBUG=YES \
    -DBUILD_SHARED=YES \
    -DUSE_CBLAS=YES \
    -DUSE_OPENMP=YES \
    -DUSE_THREADS=YES \
    -DUSE_GPU=YES \
        -DUSE_CUBLAS=YES \
        -DUSE_TENSOR_CORE=NO \
    ..
make
