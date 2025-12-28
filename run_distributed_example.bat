@echo off
REM Example: Distributed Training Data Generation Across 3 Machines
REM
REM This script shows how to run the training data generation on 3 machines.
REM Each machine processes a different range of floor plans, then results are combined.

echo ========================================
echo Distributed Training Data Generation V5
echo ========================================
echo.
echo This is an EXAMPLE script showing commands for 3 machines.
echo DO NOT run this script directly - copy commands to each machine.
echo.
echo ========================================
echo STEP 1: Run evaluation on each machine
echo ========================================
echo.
echo Machine 1 (Plans 0-333):
echo python generate_training_data_v5.py --plan-start-idx 0 --plan-end-idx 334 --evaluation-only --output-dir training_data_v5_m1 --workers 8
echo.
echo Machine 2 (Plans 334-666):
echo python generate_training_data_v5.py --plan-start-idx 334 --plan-end-idx 667 --evaluation-only --output-dir training_data_v5_m2 --workers 8
echo.
echo Machine 3 (Plans 667-999):
echo python generate_training_data_v5.py --plan-start-idx 667 --plan-end-idx 1000 --evaluation-only --output-dir training_data_v5_m3 --workers 8
echo.
echo ========================================
echo STEP 2: Transfer files to one machine
echo ========================================
echo.
echo Copy these files from each machine:
echo - training_data_v5_m1\simulation_results.jsonl
echo - training_data_v5_m2\simulation_results.jsonl
echo - training_data_v5_m3\simulation_results.jsonl
echo - training_data_v5_m1\config.json (any one machine)
echo.
echo ========================================
echo STEP 3: Combine results on central machine
echo ========================================
echo.
echo mkdir training_data_v5_combined
echo type training_data_v5_m1\simulation_results.jsonl training_data_v5_m2\simulation_results.jsonl training_data_v5_m3\simulation_results.jsonl ^> training_data_v5_combined\simulation_results.jsonl
echo copy training_data_v5_m1\config.json training_data_v5_combined\config.json
echo.
echo ========================================
echo STEP 4: Run pairing phase
echo ========================================
echo.
echo python run_pairing_phase.py --input-file training_data_v5_combined\simulation_results.jsonl --output-dir training_data_v5_final --config-json training_data_v5_combined\config.json
echo.
echo ========================================
echo STEP 5: Verify outputs
echo ========================================
echo.
echo Check training_data_v5_final directory for:
echo - train_pairs.jsonl
echo - val_pairs.jsonl
echo - test_pairs.jsonl
echo - raw_pairs.jsonl
echo - metadata.json
echo.
echo ========================================
echo Done!
echo ========================================

pause
