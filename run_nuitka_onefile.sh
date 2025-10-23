#!/bin/bash
# Single-File Nuitka Compilation for monte_carlo.py
# This creates a single executable file (slower startup but easier to distribute)

echo "Starting Nuitka single-file compilation..."
echo "This may take 10-20 minutes depending on your server specs."
echo "Note: --onefile has slower startup but creates a single portable executable."
echo ""

python -m nuitka \
  --standalone \
  --onefile \
  --enable-plugin=multiprocessing \
  --include-package=d_star_lite \
  --include-module=simulation \
  --include-module=fire_model_float \
  --include-module=fire_model_realistic \
  --include-module=fire_model_aggressive \
  --include-module=fire_monitor \
  --include-module=door_graph \
  --nofollow-import-to=pygame \
  --nofollow-import-to=scipy \
  --nofollow-import-to=matplotlib \
  --assume-yes-for-downloads \
  --output-dir=build \
  --lto=yes \
  --python-flag=no_site \
  monte_carlo.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Compilation successful!"
    echo ""
    echo "Single executable created at: ./build/monte_carlo.bin"
    echo ""
    echo "To run the compiled version:"
    echo "  ./build/monte_carlo.bin --runs 1000 --parallel"
    echo ""
    echo "Example with all options:"
    echo "  ./build/monte_carlo.bin --runs 1000 --parallel --processes \$(nproc) --config example_configuration.json"
    echo ""
    echo "Note: First run may be slower as it extracts dependencies to /tmp"
else
    echo ""
    echo "✗ Compilation failed. Check the error messages above."
    exit 1
fi
