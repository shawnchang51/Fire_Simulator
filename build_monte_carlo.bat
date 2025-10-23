@echo off
REM Nuitka build script for monte_carlo.py
REM This creates a standalone .exe file with all dependencies

echo Building monte_carlo.exe with Nuitka...
echo This may take several minutes...

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
    --nofollow-import-to=pygame ^
    --nofollow-import-to=scipy ^
    --nofollow-import-to=matplotlib ^
    --assume-yes-for-downloads ^
    --output-dir=build ^
    --output-filename=monte_carlo.exe ^
    monte_carlo.py

echo.
echo Build complete! Check the build folder for monte_carlo.exe
pause
