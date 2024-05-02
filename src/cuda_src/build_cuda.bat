@echo off

call .\setup.bat

set batch_dir=%cd%
set BuildFolder=.\build
IF NOT EXIST %BuildFolder% mkdir %BuildFolder%
pushd %BuildFolder%

nvcc --shared %batch_dir%\cuda.cu -o cuda.dll -l Shlwapi

popd