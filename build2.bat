@echo off
setlocal
mkdir build 2> nul
cd build
nvcc ../src/second.cu -o main2.exe -O3 -arch=native -std=c++20
copy /Y main2.exe ..