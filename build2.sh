#!/bin/bash
set -e
mkdir -p build
cd build
nvcc ../src/second.cu -o main2 -O3 -arch=native -std=c++20
cp main2 ..