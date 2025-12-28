@echo off
REM ============================================================
REM Fire Simulator V5 - Distributed Evaluation
REM Total num-floor-plans = 8 (split into 4 chunks)
REM Left-closed, Right-open ranges
REM ============================================================

set PYTHON=python
set SCRIPT=generate_training_data_v5.py
set RESPLAN=C:\dev\Fire_Simulator\ResPlan\ResPlan.pkl
set SEED=523

REM ---------- Task 1: plans [1, 3) ----------
%PYTHON% %SCRIPT% ^
  --resplan-path %RESPLAN% ^
  --num-floor-plans 8 ^
  --exit-configs-per-plan 2 ^
  --door-configs-per-exit 2 ^
  --monte-carlo-runs 1 ^
  --workers 6 ^
  --output-dir ./test_v5_dis_01 ^
  --seed %SEED% ^
  --plan-start-idx 1 ^
  --plan-end-idx 3 ^
  --evaluation-only

REM ---------- Task 2: plans [3, 5) ----------
%PYTHON% %SCRIPT% ^
  --resplan-path %RESPLAN% ^
  --num-floor-plans 8 ^
  --exit-configs-per-plan 2 ^
  --door-configs-per-exit 2 ^
  --monte-carlo-runs 1 ^
  --workers 6 ^
  --output-dir ./test_v5_dis_02 ^
  --seed %SEED% ^
  --plan-start-idx 3 ^
  --plan-end-idx 5 ^
  --evaluation-only

REM ---------- Task 3: plans [5, 7) ----------
%PYTHON% %SCRIPT% ^
  --resplan-path %RESPLAN% ^
  --num-floor-plans 8 ^
  --exit-configs-per-plan 2 ^
  --door-configs-per-exit 2 ^
  --monte-carlo-runs 1 ^
  --workers 6 ^
  --output-dir ./test_v5_dis_03 ^
  --seed %SEED% ^
  --plan-start-idx 5 ^
  --plan-end-idx 7 ^
  --evaluation-only

REM ---------- Task 4: plans [7, 9) ----------
%PYTHON% %SCRIPT% ^
  --resplan-path %RESPLAN% ^
  --num-floor-plans 8 ^
  --exit-configs-per-plan 2 ^
  --door-configs-per-exit 2 ^
  --monte-carlo-runs 1 ^
  --workers 6 ^
  --output-dir ./test_v5_dis_04 ^
  --seed %SEED% ^
  --plan-start-idx 7 ^
  --plan-end-idx 9 ^
  --evaluation-only

echo.
echo All tasks completed.
pause
