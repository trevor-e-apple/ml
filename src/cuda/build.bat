@echo off

call .\setup.bat

set BuildFolder=..\..\target
IF NOT EXIST %BuildFolder% mkdir %BuildFolder%
pushd %BuildFolder%

rem nvcc --shared matrix.cu -o matrix.dll
nvcc ..\knn.cu -o knn.dll --debug --device-debug -l Shlwapi

popd