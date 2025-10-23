#!/bin/bash
# Helper script to choose between Nuitka compilation methods

echo "Monte Carlo Nuitka Compilation Options"
echo "======================================="
echo ""
echo "1) Standalone (Recommended for servers)"
echo "   - Faster startup time"
echo "   - Creates monte_carlo.dist/ directory"
echo "   - Better for repeated runs"
echo ""
echo "2) Single File (Portable)"
echo "   - Single executable file"
echo "   - Slower startup (extracts to /tmp each run)"
echo "   - Easier to distribute"
echo ""
read -p "Choose compilation method (1 or 2): " choice

case $choice in
    1)
        echo ""
        echo "Running standalone compilation..."
        ./run_nuitka.sh
        ;;
    2)
        echo ""
        echo "Running single-file compilation..."
        ./run_nuitka_onefile.sh
        ;;
    *)
        echo "Invalid choice. Please run again and choose 1 or 2."
        exit 1
        ;;
esac
