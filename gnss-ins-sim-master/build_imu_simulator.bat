@echo off
echo ========================================
echo Building IMU Simulator Executable
echo ========================================

echo.
echo Step 1: Installing requirements...
pip install pyinstaller numpy matplotlib scipy

echo.
echo Step 2: Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist __pycache__ rmdir /s /q __pycache__

echo.
echo Step 3: Building executable...
pyinstaller --onefile ^
    --name imu_simulator ^
    --add-data "gnss_ins_sim;gnss_ins_sim" ^
    --add-data "demo_motion_def_files;demo_motion_def_files" ^
    --hidden-import numpy ^
    --hidden-import matplotlib ^
    --hidden-import scipy ^
    --hidden-import gnss_ins_sim ^
    --hidden-import gnss_ins_sim.sim ^
    --hidden-import gnss_ins_sim.sim.imu_model ^
    --hidden-import gnss_ins_sim.sim.ins_sim ^
    --hidden-import gnss_ins_sim.geoparams ^
    --collect-all gnss_ins_sim ^
    imu_simulator.py

echo.
echo ========================================
echo Build complete!
echo Executable location: dist\imu_simulator.exe
echo ========================================
echo.
echo Usage examples:
echo   imu_simulator.exe -t demo_motion_def_files\motion_def-static.csv
echo   imu_simulator.exe -t path\to\trajectory.csv -o output_data
echo   imu_simulator.exe -t trajectory.csv -a high-accuracy --fs 100
echo.
pause