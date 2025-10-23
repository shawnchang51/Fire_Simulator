@echo off
REM Example script to run Monte Carlo queue system
REM
REM Usage:
REM   1. Put your JSON config files in a folder (e.g., ./configs)
REM   2. Edit the parameters below
REM   3. Run this script

echo ========================================
echo Monte Carlo Queue System - Quick Start
echo ========================================
echo.

REM ====== CONFIGURE THESE ======
set QUEUE_FOLDER=example_configs
set NUM_RUNS=50
set PARALLEL=--parallel
set OUTPUT_DIR=monte_carlo_results
REM ==============================

echo Queue Folder: %QUEUE_FOLDER%
echo Runs per config: %NUM_RUNS%
echo Output Directory: %OUTPUT_DIR%
echo.

REM Check if queue folder exists
if not exist "%QUEUE_FOLDER%" (
    echo ERROR: Folder '%QUEUE_FOLDER%' not found!
    echo.
    echo Please create the folder and add JSON configuration files.
    echo Example: mkdir %QUEUE_FOLDER%
    echo          copy example_configuration.json %QUEUE_FOLDER%\
    pause
    exit /b 1
)

REM Count JSON files
set COUNT=0
for %%F in (%QUEUE_FOLDER%\*.json) do set /a COUNT+=1

if %COUNT% EQU 0 (
    echo ERROR: No JSON files found in '%QUEUE_FOLDER%'!
    echo.
    echo Please add at least one JSON configuration file.
    echo Example: copy example_configuration.json %QUEUE_FOLDER%\
    pause
    exit /b 1
)

echo Found %COUNT% configuration file(s) in %QUEUE_FOLDER%
echo.

REM Run the queue
python monte_carlo_queue.py --queue-folder %QUEUE_FOLDER% --runs %NUM_RUNS% %PARALLEL% --output %OUTPUT_DIR%

echo.
echo ========================================
echo Queue processing complete!
echo Results saved to: %OUTPUT_DIR%
echo ========================================
pause
