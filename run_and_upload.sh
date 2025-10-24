#!/bin/bash
# Automated Monte Carlo Runner with Upload
# Runs simulations for multiple configurations and uploads results to transfer.sh

set -e  # Exit on any error

# ============================================================
# CONFIGURATION - EDIT THESE VARIABLES
# ============================================================

# Simulation executable (Python script or compiled binary)
SIM_EXECUTABLE="python monte_carlo.py"

# Directory containing JSON configuration files
JSON_DIR="configs"

# Output directory (will be created if doesn't exist)
OUTPUT_DIR="monte_carlo_results"

# Number of simulation runs per configuration
NUM_RUNS=100

# ============================================================
# END CONFIGURATION
# ============================================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to compress and upload results
compress_and_upload() {
    local config_name=$1
    local result_folder=$2
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local archive_name="${config_name}_${timestamp}.tar.gz"

    print_info "Compressing ${result_folder}..."
    tar -czf "${archive_name}" -C "${OUTPUT_DIR}" "$(basename ${result_folder})"

    if [ $? -ne 0 ]; then
        print_error "Failed to compress ${result_folder}"
        return 1
    fi

    print_success "Created archive: ${archive_name}"

    # Get file size for logging
    local size=$(du -h "${archive_name}" | cut -f1)
    print_info "Archive size: ${size}"

    print_info "Uploading ${archive_name} to transfer.sh via rclone..."
    rclone copy "${archive_name}" transfer.sh:

    if [ $? -eq 0 ]; then
        print_success "Upload completed: ${archive_name}"

        # Clean up local archive
        print_info "Cleaning up local archive..."
        rm "${archive_name}"
        print_success "Local archive removed"
    else
        print_error "Upload failed for ${archive_name}"
        print_warning "Keeping local archive: ${archive_name}"
        return 1
    fi
}

# Main script starts here
echo "============================================================"
echo "  Monte Carlo Simulation Runner with Auto-Upload"
echo "============================================================"
echo ""

# Check if JSON directory exists
if [ ! -d "${JSON_DIR}" ]; then
    print_error "JSON directory not found: ${JSON_DIR}"
    print_info "Please create the directory and add JSON configuration files."
    print_info "Example: mkdir ${JSON_DIR}"
    exit 1
fi

# Find all JSON files in the directory
print_info "Scanning ${JSON_DIR} for JSON files..."
JSON_FILES=()
while IFS= read -r -d '' file; do
    JSON_FILES+=("$file")
done < <(find "${JSON_DIR}" -maxdepth 1 -name "*.json" -type f -print0 | sort -z)

# Check if any JSON files were found
if [ ${#JSON_FILES[@]} -eq 0 ]; then
    print_error "No JSON files found in ${JSON_DIR}"
    print_info "Please add at least one JSON configuration file to the directory."
    exit 1
fi

print_success "Found ${#JSON_FILES[@]} JSON file(s) in ${JSON_DIR}"

echo ""
print_info "Configuration:"
echo "  - Executable: ${SIM_EXECUTABLE}"
echo "  - JSON directory: ${JSON_DIR}"
echo "  - Runs per config: ${NUM_RUNS}"
echo "  - Output directory: ${OUTPUT_DIR}"
echo "  - Total configs: ${#JSON_FILES[@]}"
echo ""

# List all configs to be processed
print_info "Configs to process:"
for json_file in "${JSON_FILES[@]}"; do
    echo "  - $(basename ${json_file})"
done
echo ""

# Check if rclone is available
if ! command -v rclone &> /dev/null; then
    print_error "rclone is not installed or not in PATH"
    print_info "Please install rclone: https://rclone.org/downloads/"
    exit 1
fi

# Check if transfer.sh remote is configured
if ! rclone listremotes | grep -q "transfer.sh:"; then
    print_error "rclone remote 'transfer.sh' is not configured"
    print_info "Please configure rclone with transfer.sh remote"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Track overall progress
TOTAL_CONFIGS=${#JSON_FILES[@]}
CURRENT_CONFIG=0
FAILED_CONFIGS=()

# Main loop - process each configuration
for json_file in "${JSON_FILES[@]}"; do
    CURRENT_CONFIG=$((CURRENT_CONFIG + 1))

    echo ""
    echo "============================================================"
    echo "  Processing Config ${CURRENT_CONFIG}/${TOTAL_CONFIGS}: ${json_file}"
    echo "============================================================"
    echo ""

    # Check if config file exists
    if [ ! -f "${json_file}" ]; then
        print_error "Configuration file not found: ${json_file}"
        FAILED_CONFIGS+=("${json_file} (file not found)")
        continue
    fi

    # Extract config name without extension for naming
    config_basename=$(basename "${json_file}" .json)

    # Run the simulation
    print_info "Starting simulation..."
    echo ""

    START_TIME=$(date +%s)

    ${SIM_EXECUTABLE} --runs ${NUM_RUNS} --parallel --config "${json_file}" --output "${OUTPUT_DIR}"
    SIM_EXIT_CODE=$?

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""

    # Check if simulation completed successfully
    if [ ${SIM_EXIT_CODE} -eq 0 ]; then
        print_success "Simulation completed in ${DURATION} seconds"

        # Find the most recent output folder for this config
        # Output format: {config_name}_{timestamp}
        RESULT_FOLDER=$(ls -dt ${OUTPUT_DIR}/${config_basename}_* 2>/dev/null | head -n 1)

        if [ -z "${RESULT_FOLDER}" ]; then
            print_error "Could not find result folder for ${config_basename}"
            FAILED_CONFIGS+=("${json_file} (results not found)")
            continue
        fi

        print_info "Results folder: ${RESULT_FOLDER}"

        # Compress and upload
        if compress_and_upload "${config_basename}" "${RESULT_FOLDER}"; then
            print_success "Config ${CURRENT_CONFIG}/${TOTAL_CONFIGS} completed and uploaded"
        else
            print_error "Upload failed for ${json_file}"
            FAILED_CONFIGS+=("${json_file} (upload failed)")
        fi
    else
        print_error "Simulation failed for ${json_file} (exit code: ${SIM_EXIT_CODE})"
        FAILED_CONFIGS+=("${json_file} (simulation failed)")
    fi

    echo ""
done

# Final summary
echo "============================================================"
echo "  FINAL SUMMARY"
echo "============================================================"
echo ""
print_info "Total configurations processed: ${TOTAL_CONFIGS}"

if [ ${#FAILED_CONFIGS[@]} -eq 0 ]; then
    print_success "All simulations completed and uploaded successfully!"
else
    print_warning "Failed configurations: ${#FAILED_CONFIGS[@]}"
    for failed in "${FAILED_CONFIGS[@]}"; do
        echo "  - ${failed}"
    done
fi

echo ""
echo "============================================================"
echo "  Script completed at $(date)"
echo "============================================================"

# Exit with error if any configs failed
if [ ${#FAILED_CONFIGS[@]} -gt 0 ]; then
    exit 1
fi
