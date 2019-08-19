#!/bin/bash

if [ ! -d "./build" ]; then
    mkdir ./build
fi

cd ./build
rm -rf *

cmake \
    -DENABLE_DEBUG=YES \
    -DBUILD_SHARED=YES \
    -DUSE_GPU=NO \
    -DUSE_RAW=NO \
    -DUSE_REDUCE_INDEX_OVERHEAD=NO \
    -DUSE_COL_UNROLL=NO \
    -DUSE_SCALAR_REGISTER=NO \
    -DUSE_BLOCK4X4=NO \
    -DUSE_BLOCK4X4_REG=NO \
    -DUSE_BLOCK4X4_VREG=NO \
    -DUSE_BLOCK4X4_CACHED_VREG=YES \
    ..
make