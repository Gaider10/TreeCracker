#!/bin/bash
set -e
mkdir -p build
cd build
nvcc ../src/main.cu -o main -O3 -arch=native -std=c++20
cp main ..