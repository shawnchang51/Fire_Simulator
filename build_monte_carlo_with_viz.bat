@echo off
REM Nuitka build script for monte_carlo.py WITH visualization support
REM This creates a larger .exe but includes pygame, scipy, matplotlib

echo Building monte_carlo.exe with FULL visualization support...
echo This may take 10-15 minutes and create a larger file...

python -m nuitka ^
    --standalone ^
    --onefile ^
    --windows-console-mode=force ^
    --enable-plugin=numpy ^
    --enable-plugin=multiprocessing ^
    --include-package=d_star_lite ^
    --include-module=simulation ^
    --include-module=fire_model_float ^
    --include-module=fire_model_realistic ^
    --include-module=fire_model_aggressive ^
    --include-module=fire_monitor ^
    --include-module=door_graph ^
    --include-module=pygame_visualizer ^
    --include-module=matlab_visualizer ^
    --include-module=snapshot_ainmator ^
    --follow-import-to=pygame ^
    --follow-import-to=scipy ^
    --follow-import-to=matplotlib ^
    --assume-yes-for-downloads ^
    --output-dir=build ^
    --output-filename=monte_carlo_full.exe ^
    monte_carlo.py

echo.
echo Build complete! Check the build folder for monte_carlo_full.exe
pause
