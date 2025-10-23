#!/bin/bash
# Server-Optimized Nuitka Compilation for monte_carlo.py
# This script compiles monte_carlo.py for maximum performance on Linux servers

echo "Starting Nuitka compilation..."
echo "This may take 5-15 minutes depending on your server specs."
echo ""

nuitka \
  --standalone \
  --follow-imports \
  --lto=yes \
  --python-flag=no_site \
  --assume-yes-for-downloads \
  --remove-output \
  --nofollow-import-to=pygame \
  --nofollow-import-to=matplotlib.backends.backend_tkagg \
  monte_carlo.py

if [ $? -eq 0 ]; then
    echo ""
    echo " Compilation successful!"
    echo ""
    echo "To run the compiled version:"
    echo "  ./monte_carlo.dist/monte_carlo --runs 1000 --parallel"
    echo ""
    echo "Example with all options:"
    echo "  ./monte_carlo.dist/monte_carlo --runs 1000 --parallel --processes \$(nproc) --config example_configuration.json"
else
    echo ""
    echo " Compilation failed. Check the error messages above."
    exit 1
fi
