@echo off
setlocal
mkdir build 2> nul
cd build
nvcc ../src/main.cu -o main.exe -O3 -arch=native -std=c++20
copy /Y main.exe ..