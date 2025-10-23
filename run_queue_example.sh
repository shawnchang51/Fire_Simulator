#!/bin/bash
# Example script to run Monte Carlo queue system
#
# Usage:
#   1. Put your JSON config files in a folder (e.g., ./configs)
#   2. Edit the parameters below
#   3. Run: chmod +x run_queue_example.sh && ./run_queue_example.sh

echo "========================================"
echo "Monte Carlo Queue System - Quick Start"
echo "========================================"
echo ""

# ====== CONFIGURE THESE ======
QUEUE_FOLDER="example_configs"
NUM_RUNS=50
PARALLEL="--parallel"
OUTPUT_DIR="monte_carlo_results"
# ==============================

echo "Queue Folder: $QUEUE_FOLDER"
echo "Runs per config: $NUM_RUNS"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Check if queue folder exists
if [ ! -d "$QUEUE_FOLDER" ]; then
    echo "ERROR: Folder '$QUEUE_FOLDER' not found!"
    echo ""
    echo "Please create the folder and add JSON configuration files."
    echo "Example: mkdir $QUEUE_FOLDER"
    echo "         cp example_configuration.json $QUEUE_FOLDER/"
    exit 1
fi

# Count JSON files
COUNT=$(ls -1 "$QUEUE_FOLDER"/*.json 2>/dev/null | wc -l)

if [ $COUNT -eq 0 ]; then
    echo "ERROR: No JSON files found in '$QUEUE_FOLDER'!"
    echo ""
    echo "Please add at least one JSON configuration file."
    echo "Example: cp example_configuration.json $QUEUE_FOLDER/"
    exit 1
fi

echo "Found $COUNT configuration file(s) in $QUEUE_FOLDER"
echo ""

# Run the queue
python monte_carlo_queue.py --queue-folder "$QUEUE_FOLDER" --runs $NUM_RUNS $PARALLEL --output "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Queue processing complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"
